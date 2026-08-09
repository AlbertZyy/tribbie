from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Halo benchmark JSON file as Markdown")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    lines = [
        "# Halo Performance Report",
        "",
        f"- Timestamp (UTC): `{data['timestamp_utc']}`",
        f"- MPI processes: `{data['environment']['mpi_size']}`",
        f"- Platform: `{data['environment']['platform']}`",
        f"- Python: `{data['environment']['python']}`",
        "",
        "## Parameters",
        "",
    ]
    for key, value in data["parameters"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Plan",
        "",
        f"- Build time (max rank): `{data['plan']['build_seconds_max_rank']:.6e}` s",
        f"- Neighbors: `{data['plan']['neighbors']}`",
        f"- Send entities: `{data['plan']['send_count']}`",
        f"- Receive entities: `{data['plan']['recv_count']}`",
        "",
        "## Measurements",
        "",
        "| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, measurement in data["metrics"].items():
        lines.append(
            f"| `{name}` | {measurement['median_seconds']:.6e} | "
            f"{measurement['p10_seconds']:.6e} | {measurement['p90_seconds']:.6e} | "
            f"{measurement['max_seconds']:.6e} | {measurement['payload_bytes']} | "
            f"{measurement['effective_bandwidth_bytes_per_second']:.6e} | {measurement['temporary_memory_peak_bytes']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Timing values are communicator-wide maximum rank times.",
        "- `plan_*` cases use the current reusable-buffer Halo implementation.",
        "- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.",
        "- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.",
        "",
        "## Raw Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in data.get("notes", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
