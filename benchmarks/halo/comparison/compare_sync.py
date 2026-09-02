from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloPlan


# ---------------------------------------------------------------------------
# EntityMPI loading (lightweight: bypasses fealpy.distributed.__init__ which
# imports the mesh/plotting stack and matplotlib).
# ---------------------------------------------------------------------------
def load_entity_mpi(fealpy_root: Path):
    root = Path(fealpy_root)
    sys.path.insert(0, str(root))
    import fealpy  # noqa: F401  (fealpy/__init__ -> logs -> tqdm only)
    import fealpy.backend  # noqa: F401  (numpy backend loads lazily on first bm access)

    pkg_name = "fealpy.distributed"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(root / "fealpy" / "distributed")]
        sys.modules[pkg_name] = pkg

    spec = importlib.util.spec_from_file_location(
        pkg_name + ".entity_mpi",
        root / "fealpy" / "distributed" / "entity_mpi.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Measurement:
    median_seconds: float
    p10_seconds: float
    p90_seconds: float
    max_seconds: float
    payload_bytes: int
    effective_bandwidth_bytes_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HaloPlan.reduce_and_broadcast vs EntityMPI.sync_add comparison"
    )
    parser.add_argument("--topology", choices=("ring", "grid"), default="ring")
    parser.add_argument("--entities-list", type=str, default="100000",
                        help="ring: owned entities E per rank; grid: owned side length n.")
    parser.add_argument("--halo-list", type=str, default="1024",
                        help="Halo width H per direction.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fealpy-path", type=Path, default=Path(r"D:\suanhai_repo\fealpy"))
    parser.add_argument("--no-assert", action="store_true",
                        help="Skip the numerical-equivalence assertion.")
    return parser.parse_args()


def _csv_ints(text: str) -> list[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def synchronized_max(comm: MPI.Comm, elapsed: float) -> float:
    return float(comm.allreduce(elapsed, op=MPI.MAX))


def make_payload(local_count: int, rank: int, dtype: np.dtype[Any]) -> np.ndarray:
    values = np.arange(local_count, dtype=dtype)
    return values + dtype.type(rank)


def factor_grid(size: int) -> tuple[int, int]:
    """Split ``size`` into ``(Px, Py)`` with ``Py`` the largest factor <= sqrt(size)."""
    for py in range(math.isqrt(size), 0, -1):
        if size % py == 0:
            return size // py, py
    return size, 1


# ---------------------------------------------------------------------------
# Metadata generators.  Each returns (global_ids, owners, sharing_pairs) where
# sharing_pairs[i] is either None or an (index_self, index_other) tuple for the
# peer rank i.  Both the tribbie plan (from_global_ids) and the EntityMPI plan
# are built from the SAME metadata, so they describe the same distributed
# entity set, owners, and sharing relationships.
# ---------------------------------------------------------------------------
def generate_ring(rank: int, size: int, entities: int, halo: int):
    """1-D periodic ring: each rank owns E entities, shares H with each neighbor.

    Local layout: [0,H) left ghosts (owned by prev), [H,E+H) owned,
    [E+H,E+2H) right ghosts (owned by next).
    """
    if size < 2:
        raise ValueError("ring requires size >= 2")
    if entities < 2 * halo:
        raise ValueError("entities must be at least twice halo")

    prev = (rank - 1) % size
    nxt = (rank + 1) % size

    left_gids = prev * entities + (entities - halo) + np.arange(halo, dtype=np.int64)
    owned_gids = rank * entities + np.arange(entities, dtype=np.int64)
    right_gids = nxt * entities + np.arange(halo, dtype=np.int64)
    global_ids = np.concatenate([left_gids, owned_gids, right_gids])
    owners = np.concatenate([
        np.full(halo, prev, dtype=np.int32),
        np.full(entities, rank, dtype=np.int32),
        np.full(halo, nxt, dtype=np.int32),
    ])

    # Full bidirectional sharing with a neighbour: my left ghosts [0,H) and my
    # owned leftmost [H,2H) are shared with prev; my owned rightmost [E,E+H)
    # and my right ghosts [E+H,E+2H) are shared with next.  This mirrors
    # dist_from_masks, which exposes BOTH owned-shared and ghost-shared entities.
    pairs: list[Any] = [None] * size
    if size == 2:
        peer = 1 - rank
        index_self = np.concatenate([
            np.arange(2 * halo, dtype=np.intp),
            np.arange(entities, entities + 2 * halo, dtype=np.intp),
        ])
        index_other = np.concatenate([
            np.arange(entities, entities + 2 * halo, dtype=np.intp),
            np.arange(2 * halo, dtype=np.intp),
        ])
        pairs[peer] = (index_self, index_other)
    else:
        pairs[prev] = (
            np.arange(2 * halo, dtype=np.intp),
            np.arange(entities, entities + 2 * halo, dtype=np.intp),
        )
        pairs[nxt] = (
            np.arange(entities, entities + 2 * halo, dtype=np.intp),
            np.arange(2 * halo, dtype=np.intp),
        )
    return global_ids, owners, pairs


def generate_grid(rank: int, size: int, n: int, halo: int):
    """2-D periodic uniform grid (torus): Px x Py ranks, each owns n x n nodes.

    Local block is (n+2H) x (n+2H) with periodic wrap.  Sharing pairs cover the
    8 neighbours (4 edges + 4 diagonals); diagonals are required so that corner
    nodes shared by 4 ranks are fully summed by sync_add.
    """
    if size < 4:
        raise ValueError("grid requires size >= 4")
    if n < 2 or halo < 1 or halo >= n:
        raise ValueError("grid requires n >= 2 and 1 <= halo < n")

    px, py = factor_grid(size)
    if n + 2 * halo > px * n or n + 2 * halo > py * n:
        raise ValueError(
            f"grid block width {n + 2 * halo} exceeds periodic extent for {px}x{py} ranks "
            "(halo too large; a wrapped block would duplicate global ids)"
        )
    gx_rank, gy_rank = rank // py, rank % py
    GX = px * n
    GY = py * n
    L = n + 2 * halo

    li = np.arange(L, dtype=np.int64)
    gi = (gx_rank * n - halo + li) % GX
    gj = (gy_rank * n - halo + li) % GY
    # Broadcasting (no full meshgrid) keeps the peak memory low for large payloads.
    global_ids = (gi[:, None] * GY + gj[None, :]).ravel()
    own_i = ((gi // n) * py).astype(np.int32)
    own_j = (gj // n).astype(np.int32)
    owners = (own_i[:, None] + own_j[None, :]).ravel()

    base_i = (gx_rank * n - halo) % GX
    base_j = (gy_rank * n - halo) % GY

    collected: dict[int, list[np.ndarray]] = {}
    for dnx in (-1, 0, 1):
        for dny in (-1, 0, 1):
            if dnx == 0 and dny == 0:
                continue
            npx = (gx_rank + dnx) % px
            npy = (gy_rank + dny) % py
            nrank = npx * py + npy

            glo = max(gx_rank * n - halo, (gx_rank + dnx) * n - halo)
            ghi = min(gx_rank * n + n + halo, (gx_rank + dnx) * n + n + halo)
            jlo = max(gy_rank * n - halo, (gy_rank + dny) * n - halo)
            jhi = min(gy_rank * n + n + halo, (gy_rank + dny) * n + n + halo)
            if glo >= ghi or jlo >= jhi:
                continue

            ogi = np.arange(glo, ghi, dtype=np.int64)
            ogj = np.arange(jlo, jhi, dtype=np.int64)
            OGI, OGJ = np.meshgrid(ogi, ogj, indexing="ij")

            gi_mod = OGI % GX
            gj_mod = OGJ % GY
            base_i2 = (npx * n - halo) % GX
            base_j2 = (npy * n - halo) % GY

            index_self = (((gi_mod - base_i) % GX) * L + ((gj_mod - base_j) % GY)).ravel()
            index_other = (((gi_mod - base_i2) % GX) * L + ((gj_mod - base_j2) % GY)).ravel()

            collected.setdefault(nrank, ([], []))
            collected[nrank][0].append(index_self)
            collected[nrank][1].append(index_other)

    pairs: list[Any] = [None] * size
    for nrank, (self_list, other_list) in collected.items():
        pairs[nrank] = (np.concatenate(self_list), np.concatenate(other_list))
    return global_ids, owners, pairs


def sync_add_no_copy(empi, array: np.ndarray) -> np.ndarray:
    """EntityMPI.sync_add without the full-array copy (in-place reduce).

    ``sync`` copies every sent block before the alltoall, so accumulating into
    ``array`` afterwards is safe and performs no full-array copy.
    """
    for data in empi.sync(array):
        if data is not None:
            np.add.at(array, data.indices, data.data)
    return array


def assert_equivalence(
    reduce_plan: HaloPlan,
    broadcast_plan: HaloPlan,
    empi,
    local_count: int,
    rank: int,
) -> None:
    base = make_payload(local_count, rank, np.dtype(np.float64))
    expected = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, base.copy())
    got_copy = empi.sync_add(base.copy())
    got_no_copy = sync_add_no_copy(empi, base.copy())
    if not np.allclose(expected, got_copy):
        raise AssertionError("sync_add (copy) mismatch vs reduce_and_broadcast")
    if not np.allclose(expected, got_no_copy):
        raise AssertionError("sync_add (no-copy) mismatch vs reduce_and_broadcast")


def time_fn(
    comm: MPI.Comm,
    fn: Callable[[], Any],
    payload_bytes: int,
    warmup: int,
    repeats: int,
) -> Measurement:
    for _ in range(warmup):
        fn()
        comm.Barrier()
    samples: list[float] = []
    for _ in range(repeats):
        comm.Barrier()
        started = time.perf_counter()
        fn()
        samples.append(synchronized_max(comm, time.perf_counter() - started))
    median = statistics.median(samples)
    return Measurement(
        median_seconds=median,
        p10_seconds=float(np.percentile(samples, 10)),
        p90_seconds=float(np.percentile(samples, 90)),
        max_seconds=max(samples),
        payload_bytes=payload_bytes,
        effective_bandwidth_bytes_per_second=payload_bytes / median if median else float("inf"),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    EntityMPI = load_entity_mpi(args.fealpy_path).EntityMPI
    SharingPair = sys.modules["fealpy.distributed.entity_mpi"].SharingPair

    entities_list = _csv_ints(args.entities_list)
    halo_list = _csv_ints(args.halo_list)
    if len(entities_list) != len(halo_list):
        raise ValueError("--entities-list and --halo-list must have the same length (paired element-wise)")
    dtype = np.dtype(np.float64)

    configs: list[dict[str, Any]] = []
    for entities, halo in zip(entities_list, halo_list):
        if args.topology == "ring":
            if size < 2 or entities < 2 * halo:
                continue
            local_count = entities + 2 * halo
            global_ids, owners, raw_pairs = generate_ring(rank, size, entities, halo)
        else:
            if size < 4 or entities < 2 or halo < 1 or halo >= entities:
                continue
            local_count = (entities + 2 * halo) ** 2
            global_ids, owners, raw_pairs = generate_grid(rank, size, entities, halo)

        pairs = [None if rp is None else SharingPair(rp[0], rp[1]) for rp in raw_pairs]

        reduce_plan, broadcast_plan = HaloPlan.from_global_ids(
            comm, global_ids, owners, direction="two_way"
        )
        empi = EntityMPI(pairs=pairs, comm=comm)
        del global_ids, owners, raw_pairs

        correct = True
        if not args.no_assert and size >= 2:
            try:
                assert_equivalence(reduce_plan, broadcast_plan, empi, local_count, rank)
            except AssertionError:
                correct = False

        payload_bytes = int(local_count * dtype.itemsize)
        halo_entities = int(reduce_plan.total_send_count)

        metrics: dict[str, dict[str, Any]] = {}
        for case in ("reduce_and_broadcast", "sync_add", "sync_add_no_copy"):
            payload = make_payload(local_count, rank, dtype)
            if case == "reduce_and_broadcast":
                fn = lambda p=payload: HaloPlan.reduce_and_broadcast(  # noqa: E731
                    reduce_plan, broadcast_plan, p
                )
            elif case == "sync_add":
                fn = lambda p=payload: empi.sync_add(p)  # noqa: E731
            else:
                fn = lambda p=payload: sync_add_no_copy(empi, p)  # noqa: E731
            metrics[case] = asdict(time_fn(comm, fn, payload_bytes, args.warmup, args.repeats))

        base_median = metrics["reduce_and_broadcast"]["median_seconds"]
        speedup_copy = metrics["sync_add"]["median_seconds"] / base_median if base_median else float("inf")
        speedup_no_copy = metrics["sync_add_no_copy"]["median_seconds"] / base_median if base_median else float("inf")

        configs.append({
            "topology": args.topology,
            "entities": entities,
            "halo": halo,
            "local_count": local_count,
            "payload_bytes": payload_bytes,
            "halo_entities": halo_entities,
            "correct": correct,
            "metrics": metrics,
            "speedup_copy": speedup_copy,
            "speedup_no_copy": speedup_no_copy,
        })
        reduce_plan.close()
        broadcast_plan.close()

    result: dict[str, object] = {
        "schema": "tribbie.halo.comparison.v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mpi_size": size,
            "fealpy_path": str(args.fealpy_path),
        },
        "parameters": {
            "topology": args.topology,
            "entities_list": entities_list,
            "halo_list": halo_list,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "configs": configs,
        "notes": [
            "Compares HaloPlan.reduce_and_broadcast (two_way from_global_ids) with EntityMPI.sync_add (copy and no-copy variants).",
            "Both paths are mathematically equivalent: every copy of every shared entity ends up with the global sum of all contributions.",
            "Construction time is excluded and not compared (from_global_ids discovery vs sharing_pairs are not comparable).",
            "speedup_copy = sync_add_median / reduce_and_broadcast_median; speedup_no_copy = sync_add_no_copy_median / reduce_and_broadcast_median (>1 means reduce_and_broadcast is faster).",
            "sync_add (copy) is fealpy's native out-of-place API (full-array copy); sync_add_no_copy applies the same reduction in-place.",
            "tracemalloc is disabled: the full-array copy is a real cost being measured, and tracemalloc would distort it.",
            "Timing samples are communicator-wide maximum rank times.",
        ],
    }
    if rank == 0 and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
