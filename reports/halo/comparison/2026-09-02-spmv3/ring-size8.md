# reduce-and-broadcast vs sync_add — Comparison Report

- Timestamp (UTC): `2026-09-02T03:20:12.270456+00:00`
- MPI processes: `8`
- Topology: `ring`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`
- fealpy path: `D:\suanhai_repo\fealpy`

## Parameters

- entities_list (E for ring, n for grid): `[11718, 117188, 1171876]`
- halo_list: `[391, 3906, 39062]`
- warmup: `3`, repeats: `10`

## Results

`speedup = sync_median / reduce_and_broadcast_median`; values > 1 mean reduce_and_broadcast is faster.

| entity | halo | local | payload (B) | halo (entities) | correct | reduce (s) | sync_add copy (s) | speedup copy | sync_add no-copy (s) | speedup no-copy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11718 | 391 | 12500 | 100000 | 782 | True | 5.240e-05 | 3.084e-04 | 5.88x | 2.858e-04 | 5.45x |
| 117188 | 3906 | 125000 | 1000000 | 7812 | True | 2.500e-04 | 2.388e-03 | 9.55x | 1.897e-03 | 7.59x |
| 1171876 | 39062 | 1250000 | 10000000 | 78124 | True | 3.682e-03 | 2.279e-02 | 6.19x | 2.099e-02 | 5.70x |

## Interpretation

- `reduce_and_broadcast` (two_way `from_global_ids`) does ghost->owner sum then owner->ghost replace via point-to-point typed buffers.
- `sync_add` (copy) is fealpy's native out-of-place API: full-array copy + dense alltoall of pickled `SparseData1D` + index add.
- `sync_add` (no-copy) applies the same reduction in-place, isolating the full-array copy from the communication path.
- `speedup_copy - speedup_no_copy` (via the medians) reflects the cost of the full-array copy at that payload.
- Construction time is excluded and not compared (from_global_ids discovery vs sharing_pairs are not comparable).

## Notes

- Compares HaloPlan.reduce_and_broadcast (two_way from_global_ids) with EntityMPI.sync_add (copy and no-copy variants).
- Both paths are mathematically equivalent: every copy of every shared entity ends up with the global sum of all contributions.
- Construction time is excluded and not compared (from_global_ids discovery vs sharing_pairs are not comparable).
- speedup_copy = sync_add_median / reduce_and_broadcast_median; speedup_no_copy = sync_add_no_copy_median / reduce_and_broadcast_median (>1 means reduce_and_broadcast is faster).
- sync_add (copy) is fealpy's native out-of-place API (full-array copy); sync_add_no_copy applies the same reduction in-place.
- tracemalloc is disabled: the full-array copy is a real cost being measured, and tracemalloc would distort it.
- Timing samples are communicator-wide maximum rank times.
