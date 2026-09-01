from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a HaloPlan-vs-EntityMPI comparison JSON as Markdown")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    env = data["environment"]
    params = data["parameters"]
    lines = [
        "# HaloPlan.exchange vs EntityMPI.sync — Comparison Report",
        "",
        f"- Timestamp (UTC): `{data['timestamp_utc']}`",
        f"- MPI processes: `{env['mpi_size']}`",
        f"- Platform: `{env['platform']}`",
        f"- Python: `{env['python']}`",
        f"- fealpy path: `{env['fealpy_path']}`",
        "",
        "## Parameters",
        "",
        f"- entities_list: `{params['entities_list']}`",
        f"- halo_list: `{params['halo_list']}`",
        f"- components_list: `{params['components_list']}`",
        f"- warmup: `{params['warmup']}`, repeats: `{params['repeats']}`",
        "",
        "## Results",
        "",
        "`speedup = entity_mpi_median / halo_plan_median`; values > 1 mean HaloPlan is faster.",
        "",
        "| E | H | C | send | payload (B) | sync idx (B) | correct | replace: halo/empi (s) | replace speedup | sum: halo/empi (s) | sum speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in data["configs"]:
        m = c["metrics"]
        lines.append(
            f"| {c['entities']} | {c['halo']} | {c['components']} | {c['send_entities']} | "
            f"{c['payload_bytes']} | {c['index_bytes_sync']} | {c['correct']} | "
            f"{_fmt(m['halo_plan_replace']['median_seconds'])} / {_fmt(m['entity_mpi_replace']['median_seconds'])} | "
            f"{c['speedup_replace']:.2f}x | "
            f"{_fmt(m['halo_plan_sum']['median_seconds'])} / {_fmt(m['entity_mpi_sum']['median_seconds'])} | "
            f"{c['speedup_sum']:.2f}x |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- HaloPlan transfers raw typed NumPy buffers with point-to-point `Isend`/`Irecv` to real neighbors only.",
        "- EntityMPI.sync runs a dense `comm.alltoall` over pickled `SparseData1D` objects (data + an int64 index tensor per message).",
        "- `payload_bytes` counts useful payload only; `sync idx (B)` is the additional index tensor bytes EntityMPI ships every call.",
        "- Timing is end-to-end exchange time (communicator-wide max rank); construction is excluded.",
        "- `tracemalloc` peak reflects Python allocations only, not NumPy/MPI native buffers.",
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in data.get("notes", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
