from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
import tracemalloc
import types
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan


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
    temporary_memory_peak_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HaloPlan.exchange vs EntityMPI.sync comparison")
    parser.add_argument("--entities-list", type=str, default="10000,100000,1000000")
    parser.add_argument("--halo-list", type=str, default=None,
                        help="Explicit halo entities per direction (csv). If omitted, H=E//16.")
    parser.add_argument("--components-list", type=str, default="1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fealpy-path", type=Path, default=Path(r"D:\suanhai_repo\fealpy"))
    parser.add_argument("--no-assert", action="store_true",
                        help="Skip the numerical-equivalence assertion.")
    return parser.parse_args()


def _csv_ints(text: str) -> list[int]:
    return [int(token) for token in text.split(",") if token.strip()]


def synchronized_max(comm: MPI.Comm, elapsed: float) -> float:
    return float(comm.allreduce(elapsed, op=MPI.MAX))


def make_payload(shape: tuple[int, ...], rank: int, dtype: np.dtype[Any]) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return values + dtype.type(rank)


def payload_shape(entities: int, components: int) -> tuple[int, ...]:
    return (entities,) if components == 1 else (entities, components)


def build_topology(rank: int, size: int, entities: int, halo: int):
    """Return (halo_edges, sharing_pairs) for a 1-D periodic halo exchange.

    Every rank shares its left ``[0, halo)`` block with the previous rank and
    its right ``[entities-halo, entities)`` block with the next rank.  The two
    representations are aligned so that ``exchange(op="sum")`` equals
    ``sync_add`` and ``exchange(op="replace")`` equals ``sync`` + index apply.
    """
    if entities < 2 * halo:
        raise ValueError("entities must be at least twice halo_entities")
    left = np.arange(halo, dtype=np.intp)
    right = np.arange(entities - halo, entities, dtype=np.intp)
    prev = (rank - 1) % size
    nxt = (rank + 1) % size

    if size == 1:
        return [], [None]

    if size == 2:
        peer = 1 - rank
        edges = [HaloEdge(peer, left, left), HaloEdge(peer, right, right)]
        merged = np.concatenate([left, right])
        pairs = [None, None]
        pairs[peer] = (merged, merged)  # (index_self, index_other)
        return edges, pairs

    edges = [HaloEdge(prev, left, left), HaloEdge(nxt, right, right)]
    pairs = [None] * size
    pairs[prev] = (left, right)
    pairs[nxt] = (right, left)
    return edges, pairs


def apply_replace(empi, array: np.ndarray) -> np.ndarray:
    """EntityMPI 'replace' equivalent, in-place (mirrors exchange(op='replace')).

    ``sync`` copies every sent block before the ``alltoall``, so scattering into
    ``array`` afterwards is safe and does no full-array copy — the same array
    work HaloPlan.exchange performs.
    """
    for data in empi.sync(array):
        if data is not None:
            array[data.indices] = data.data
    return array


def apply_sum(empi, array: np.ndarray) -> np.ndarray:
    """EntityMPI 'sum' equivalent, in-place (mirrors exchange(op='sum'))."""
    for data in empi.sync(array):
        if data is not None:
            np.add.at(array, data.indices, data.data)
    return array


def assert_equivalence(comm: MPI.Comm, plan: HaloPlan, empi, entities: int, components: int, rank: int) -> None:
    shape = payload_shape(entities, components)
    base = make_payload(shape, rank, np.dtype(np.float64))
    lhs = plan.exchange(base.copy(), op="sum")
    rhs = apply_sum(empi, base.copy())
    if not np.allclose(lhs, rhs):
        raise AssertionError("sum result mismatch between HaloPlan.exchange and EntityMPI.sync + apply")
    lhs = plan.exchange(base.copy(), op="replace")
    rhs = apply_replace(empi, base.copy())
    if not np.allclose(lhs, rhs):
        raise AssertionError("replace result mismatch between HaloPlan.exchange and EntityMPI.sync")


def time_fn(comm: MPI.Comm, fn: Callable[[], Any], payload_bytes: int, warmup: int, repeats: int) -> Measurement:
    for _ in range(warmup):
        fn()
        comm.Barrier()
    samples: list[float] = []
    tracemalloc.start()
    for _ in range(repeats):
        comm.Barrier()
        started = time.perf_counter()
        fn()
        samples.append(synchronized_max(comm, time.perf_counter() - started))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    median = statistics.median(samples)
    return Measurement(
        median_seconds=median,
        p10_seconds=float(np.percentile(samples, 10)),
        p90_seconds=float(np.percentile(samples, 90)),
        max_seconds=max(samples),
        payload_bytes=payload_bytes,
        effective_bandwidth_bytes_per_second=payload_bytes / median if median else float("inf"),
        temporary_memory_peak_bytes=int(peak),
    )


def measure_case(
    comm: MPI.Comm,
    plan: HaloPlan,
    empi,
    shape: tuple[int, ...],
    rank: int,
    payload_bytes: int,
    warmup: int,
    repeats: int,
) -> dict[str, dict[str, Any]]:
    dtype = np.dtype(np.float64)
    metrics: dict[str, dict[str, Any]] = {}
    for op in ("replace", "sum"):
        halo_payload = make_payload(shape, rank, dtype)
        metrics[f"halo_plan_{op}"] = asdict(time_fn(
            comm, lambda p=halo_payload, o=op: plan.exchange(p, op=o), payload_bytes, warmup, repeats,
        ))
        empi_payload = make_payload(shape, rank, dtype)
        if op == "replace":
            fn = lambda p=empi_payload: apply_replace(empi, p)  # noqa: E731
        else:
            fn = lambda p=empi_payload: apply_sum(empi, p)  # noqa: E731
        metrics[f"entity_mpi_{op}"] = asdict(time_fn(comm, fn, payload_bytes, warmup, repeats))
    return metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    EntityMPI = load_entity_mpi(args.fealpy_path).EntityMPI
    SharingPair = sys.modules["fealpy.distributed.entity_mpi"].SharingPair

    entities_list = _csv_ints(args.entities_list)
    components_list = _csv_ints(args.components_list)
    if args.halo_list is not None:
        halo_list = _csv_ints(args.halo_list)
    else:
        halo_list = [max(1, e // 16) for e in entities_list]

    configs: list[dict[str, Any]] = []
    for entities in entities_list:
        for halo in halo_list:
            if entities < 2 * halo:
                continue
            for components in components_list:
                shape = payload_shape(entities, components)
                edges, raw_pairs = build_topology(rank, size, entities, halo)
                pairs = [
                    None if raw is None else SharingPair(raw[0], raw[1])
                    for raw in raw_pairs
                ]
                comm.Barrier()
                t0 = time.perf_counter()
                plan = HaloPlan.from_edges(comm, edges, entity_count=entities)
                halo_build = synchronized_max(comm, time.perf_counter() - t0)
                t0 = time.perf_counter()
                empi = EntityMPI(pairs=pairs, comm=comm)
                empi_build = synchronized_max(comm, time.perf_counter() - t0)

                correct = True
                if not args.no_assert and size >= 2:
                    try:
                        assert_equivalence(comm, plan, empi, entities, components, rank)
                    except AssertionError:
                        correct = False
                elif size == 1:
                    correct = True

                send_entities = plan.total_send_count
                payload_bytes = int(send_entities * np.prod(shape[1:] or (1,)) * np.dtype(np.float64).itemsize)
                index_bytes_sync = int(send_entities * 8)  # int64 index tensor shipped by sync

                metrics = measure_case(comm, plan, empi, shape, rank, payload_bytes, args.warmup, args.repeats)

                halo_med = metrics[f"halo_plan_replace"]["median_seconds"]
                empi_med = metrics[f"entity_mpi_replace"]["median_seconds"]
                speedup_replace = empi_med / halo_med if halo_med else float("inf")
                halo_med = metrics[f"halo_plan_sum"]["median_seconds"]
                empi_med = metrics[f"entity_mpi_sum"]["median_seconds"]
                speedup_sum = empi_med / halo_med if halo_med else float("inf")

                configs.append({
                    "entities": entities,
                    "halo": halo,
                    "components": components,
                    "send_entities": send_entities,
                    "payload_bytes": payload_bytes,
                    "index_bytes_sync": index_bytes_sync,
                    "correct": correct,
                    "halo_plan_build_seconds_max_rank": halo_build,
                    "entity_mpi_build_seconds_max_rank": empi_build,
                    "metrics": metrics,
                    "speedup_replace": speedup_replace,
                    "speedup_sum": speedup_sum,
                })
                plan.close()

    result: dict[str, object] = {
        "schema": "tribbie.halo.comparison.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mpi_size": size,
            "fealpy_path": str(args.fealpy_path),
        },
        "parameters": {
            "entities_list": entities_list,
            "halo_list": halo_list,
            "components_list": components_list,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "configs": configs,
        "notes": [
            "Timing samples are communicator-wide maximum rank times.",
            "HaloPlan.exchange uses point-to-point typed Isend/Irecv; EntityMPI.sync uses dense comm.alltoall over pickled SparseData1D objects.",
            "Both sides apply the received halo in-place (no full-array copy), so the measured difference isolates the communication path.",
            "payload_bytes counts useful payload only; index_bytes_sync is the extra int64 index tensor shipped by sync each call.",
            "EntityMPI.sync_add (the out-of-place reduce API) additionally copies the full array and is not measured here; its extra cost grows with E.",
            "speedup_* = entity_mpi_median / halo_plan_median (>1 means HaloPlan is faster).",
            "Construction (from_edges / EntityMPI init) is excluded from exchange timings.",
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
