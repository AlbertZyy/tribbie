from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from tribbie.halo import HaloEdge, HaloPlan


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
    parser = argparse.ArgumentParser(description="Halo communication microbenchmark")
    parser.add_argument("--entities", type=int, default=4096)
    parser.add_argument("--halo-entities", type=int, default=1024)
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compute", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def ring_edges(comm: MPI.Comm, halo_entities: int) -> list[HaloEdge]:
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size == 1:
        return []
    if halo_entities < 1:
        raise ValueError("halo_entities must be positive")
    previous = (rank - 1) % size
    following = (rank + 1) % size
    first = np.arange(halo_entities, dtype=np.intp)
    second = np.arange(halo_entities, 2 * halo_entities, dtype=np.intp)
    if previous == following:
        return [HaloEdge(previous, first, second), HaloEdge(previous, second, first)]
    return [HaloEdge(previous, first, second), HaloEdge(following, second, first)]


def payload_shape(entities: int, halo_entities: int, components: int) -> tuple[int, ...]:
    if entities < 2 * halo_entities:
        raise ValueError("entities must be at least twice halo_entities for the ring benchmark")
    if components < 1:
        raise ValueError("components must be positive")
    return (entities,) if components == 1 else (entities, components)


def make_payload(shape: tuple[int, ...], rank: int, dtype: np.dtype[Any]) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    return values + dtype.type(rank)


def synchronized_max(comm: MPI.Comm, elapsed: float) -> float:
    return float(comm.allreduce(elapsed, op=MPI.MAX))


def measure_plan(
    plan: HaloPlan,
    comm: MPI.Comm,
    shape: tuple[int, ...],
    rank: int,
    op: str,
    nonblocking: bool,
    warmup: int,
    repeats: int,
    compute: int,
) -> Measurement:
    payload = make_payload(shape, rank, np.dtype(np.float64))
    for _ in range(warmup):
        if nonblocking:
            request = plan.begin_exchange(payload, op=op)
            if compute:
                np.sin(payload[:compute], out=payload[:compute])
            request.wait()
        else:
            plan.exchange(payload, op=op)
        comm.Barrier()
    samples = []
    tracemalloc.start()
    for _ in range(repeats):
        comm.Barrier()
        started = time.perf_counter()
        if nonblocking:
            request = plan.begin_exchange(payload, op=op)
            if compute:
                np.sin(payload[:compute], out=payload[:compute])
            request.wait()
        else:
            plan.exchange(payload, op=op)
        elapsed = synchronized_max(comm, time.perf_counter() - started)
        samples.append(elapsed)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    payload_bytes = int(plan.total_send_count * np.prod(shape[1:] or (1,)) * np.dtype(np.float64).itemsize)
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


def measure_baseline(
    comm: MPI.Comm,
    edges: list[HaloEdge],
    shape: tuple[int, ...],
    rank: int,
    warmup: int,
    repeats: int,
) -> Measurement:
    payload = make_payload(shape, rank, np.dtype(np.float64))
    for _ in range(warmup):
        _baseline_exchange(comm, edges, payload)
        comm.Barrier()
    samples = []
    for _ in range(repeats):
        comm.Barrier()
        started = time.perf_counter()
        _baseline_exchange(comm, edges, payload)
        samples.append(synchronized_max(comm, time.perf_counter() - started))
    payload_bytes = int(sum(edge.send_indices.size for edge in edges) * np.prod(shape[1:] or (1,)) * 8)
    median = statistics.median(samples)
    return Measurement(
        median_seconds=median,
        p10_seconds=float(np.percentile(samples, 10)),
        p90_seconds=float(np.percentile(samples, 90)),
        max_seconds=max(samples),
        payload_bytes=payload_bytes,
        effective_bandwidth_bytes_per_second=payload_bytes / median if median else float("inf"),
        temporary_memory_peak_bytes=0,
    )


def _baseline_exchange(comm: MPI.Comm, edges: list[HaloEdge], payload: np.ndarray) -> None:
    receives = []
    sends = []
    for edge in edges:
        send_buffer = np.ascontiguousarray(payload[edge.send_indices]).copy()
        recv_buffer = np.empty((edge.recv_indices.size, *payload.shape[1:]), dtype=payload.dtype)
        receives.append((edge, recv_buffer))
        sends.append((edge, send_buffer))
    requests = [comm.Irecv(buffer, source=edge.peer, tag=91) for edge, buffer in receives]
    requests.extend(comm.Isend(buffer, dest=edge.peer, tag=91) for edge, buffer in sends)
    MPI.Request.Waitall(requests)
    for edge, buffer in receives:
        payload[edge.recv_indices] = buffer


def run(args: argparse.Namespace) -> dict[str, object]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    shape = payload_shape(args.entities, args.halo_entities, args.components)
    edges = ring_edges(comm, args.halo_entities)
    comm.Barrier()
    build_started = time.perf_counter()
    plan = HaloPlan.from_edges(comm, edges, entity_count=args.entities)
    build_seconds = synchronized_max(comm, time.perf_counter() - build_started)
    measurements: dict[str, dict[str, Any]] = {}
    for op in ("replace", "sum"):
        for nonblocking in (False, True):
            key = f"plan_{op}_{'nonblocking' if nonblocking else 'blocking'}"
            measurements[key] = asdict(measure_plan(plan, comm, shape, rank, op, nonblocking, args.warmup, args.repeats, args.compute))
    measurements["baseline_typed_nonblocking"] = asdict(measure_baseline(comm, edges, shape, rank, args.warmup, args.repeats))
    peak = max(measurement["temporary_memory_peak_bytes"] for measurement in measurements.values())
    result = {
        "schema": "tribbie.halo.performance.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "mpi_size": comm.Get_size()},
        "parameters": {"entities": args.entities, "halo_entities": args.halo_entities, "components": args.components, "warmup": args.warmup, "repeats": args.repeats, "compute": args.compute},
        "plan": {"build_seconds_max_rank": build_seconds, "neighbors": plan.neighbors, "send_count": plan.total_send_count, "recv_count": plan.total_recv_count},
        "metrics": measurements,
        "temporary_memory_peak_bytes_max_rank": peak,
        "notes": ["Timing samples are communicator-wide maximum rank times.", "The baseline uses typed Irecv/Isend buffers without plan metadata.", "Construction metadata exchange is excluded from runtime exchange timings."],
    }
    if rank == 0 and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plan.close()
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
