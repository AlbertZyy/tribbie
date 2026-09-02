# reduce-and-broadcast vs sync_add — Comparison Report

- Timestamp (UTC): `2026-09-02T03:21:35.529028+00:00`
- MPI processes: `8`
- Topology: `grid`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`
- fealpy path: `D:\suanhai_repo\fealpy`

## Parameters

- entities_list (E for ring, n for grid): `[112, 354, 1118]`
- halo_list: `[7, 22, 70]`
- warmup: `3`, repeats: `10`

## Results

`speedup = sync_median / reduce_and_broadcast_median`; values > 1 mean reduce_and_broadcast is faster.

| entity | halo |   local | payload (B) | halo (entities) | correct | reduce (s) | sync_add copy (s) | speedup copy | sync_add no-copy (s) | speedup no-copy |
| -----: | ---: | ------: | ----------: | --------------: | ------: | ---------: | ----------------: | -----------: | -------------------: | --------------: |
|    112 |    7 |   15876 |      127008 |            3332 |    True |  2.856e-04 |         4.439e-04 |        1.55x |            9.807e-04 |           3.43x |
|    354 |   22 |  158404 |     1267232 |           33088 |    True |  8.235e-04 |         8.805e-03 |       10.69x |            1.097e-02 |          13.33x |
|   1118 |   70 | 1582564 |    12660512 |          332640 |    True |  3.269e-02 |         1.557e-01 |        4.76x |            2.517e-01 |           7.70x |

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
