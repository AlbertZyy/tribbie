from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a reduce-and-broadcast vs sync_add comparison JSON as Markdown"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    env = data["environment"]
    params = data["parameters"]
    lines = [
        "# reduce-and-broadcast vs sync_add — Comparison Report",
        "",
        f"- Timestamp (UTC): `{data['timestamp_utc']}`",
        f"- MPI processes: `{env['mpi_size']}`",
        f"- Topology: `{params['topology']}`",
        f"- Platform: `{env['platform']}`",
        f"- Python: `{env['python']}`",
        f"- fealpy path: `{env['fealpy_path']}`",
        "",
        "## Parameters",
        "",
        f"- entities_list (E for ring, n for grid): `{params['entities_list']}`",
        f"- halo_list: `{params['halo_list']}`",
        f"- warmup: `{params['warmup']}`, repeats: `{params['repeats']}`",
        "",
        "## Results",
        "",
        "`speedup = sync_median / reduce_and_broadcast_median`; values > 1 mean reduce_and_broadcast is faster.",
        "",
        "| entity | halo | local | payload (B) | halo (entities) | correct | reduce (s) | sync_add copy (s) | speedup copy | sync_add no-copy (s) | speedup no-copy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in data["configs"]:
        m = c["metrics"]
        lines.append(
            f"| {c['entities']} | {c['halo']} | {c['local_count']} | {c['payload_bytes']} | "
            f"{c['halo_entities']} | {c['correct']} | "
            f"{_fmt(m['reduce_and_broadcast']['median_seconds'])} | "
            f"{_fmt(m['sync_add']['median_seconds'])} | {c['speedup_copy']:.2f}x | "
            f"{_fmt(m['sync_add_no_copy']['median_seconds'])} | {c['speedup_no_copy']:.2f}x |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `reduce_and_broadcast` (two_way `from_global_ids`) does ghost->owner sum then owner->ghost replace via point-to-point typed buffers.",
        "- `sync_add` (copy) is fealpy's native out-of-place API: full-array copy + dense alltoall of pickled `SparseData1D` + index add.",
        "- `sync_add` (no-copy) applies the same reduction in-place, isolating the full-array copy from the communication path.",
        "- `speedup_copy - speedup_no_copy` (via the medians) reflects the cost of the full-array copy at that payload.",
        "- Construction time is excluded and not compared (from_global_ids discovery vs sharing_pairs are not comparable).",
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in data.get("notes", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
