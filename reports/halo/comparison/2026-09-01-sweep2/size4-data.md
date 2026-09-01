# HaloPlan.exchange vs EntityMPI.sync — Comparison Report

- Timestamp (UTC): `2026-09-01T02:01:30.648961+00:00`
- MPI processes: `4`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`
- fealpy path: `D:\suanhai_repo\fealpy`

## Parameters

- entities_list: `[10000, 100000, 1000000]`
- halo_list: `[625, 6250, 62500]`
- components_list: `[1]`
- warmup: `5`, repeats: `20`

## Results

`speedup = entity_mpi_median / halo_plan_median`; values > 1 mean HaloPlan is faster.

| E | H | C | send | payload (B) | sync idx (B) | correct | replace: halo/empi (s) | replace speedup | sum: halo/empi (s) | sum speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 625 | 1 | 1250 | 10000 | 10000 | True | 9.400e-05 / 1.640e-04 | 1.74x | 1.016e-04 / 3.003e-04 | 2.96x |
| 100000 | 625 | 1 | 1250 | 10000 | 10000 | True | 9.185e-05 / 2.207e-04 | 2.40x | 1.079e-04 / 2.578e-04 | 2.39x |
| 100000 | 6250 | 1 | 12500 | 100000 | 100000 | True | 1.135e-04 / 4.392e-04 | 3.87x | 2.091e-04 / 1.368e-03 | 6.54x |
| 1000000 | 625 | 1 | 1250 | 10000 | 10000 | True | 1.545e-04 / 3.187e-04 | 2.06x | 1.572e-04 / 3.501e-04 | 2.23x |
| 1000000 | 6250 | 1 | 12500 | 100000 | 100000 | True | 2.584e-04 / 3.458e-04 | 1.34x | 2.595e-04 / 8.391e-04 | 3.23x |
| 1000000 | 62500 | 1 | 125000 | 1000000 | 1000000 | True | 8.028e-04 / 6.943e-03 | 8.65x | 9.385e-04 / 1.136e-02 | 12.10x |

## Interpretation

- HaloPlan transfers raw typed NumPy buffers with point-to-point `Isend`/`Irecv` to real neighbors only.
- EntityMPI.sync runs a dense `comm.alltoall` over pickled `SparseData1D` objects (data + an int64 index tensor per message).
- `payload_bytes` counts useful payload only; `sync idx (B)` is the additional index tensor bytes EntityMPI ships every call.
- Timing is end-to-end exchange time (communicator-wide max rank); construction is excluded.
- `tracemalloc` peak reflects Python allocations only, not NumPy/MPI native buffers.

## Notes

- Timing samples are communicator-wide maximum rank times.
- HaloPlan.exchange uses point-to-point typed Isend/Irecv; EntityMPI.sync uses dense comm.alltoall over pickled SparseData1D objects.
- Both sides apply the received halo in-place (no full-array copy), so the measured difference isolates the communication path.
- payload_bytes counts useful payload only; index_bytes_sync is the extra int64 index tensor shipped by sync each call.
- EntityMPI.sync_add (the out-of-place reduce API) additionally copies the full array and is not measured here; its extra cost grows with E.
- speedup_* = entity_mpi_median / halo_plan_median (>1 means HaloPlan is faster).
- Construction (from_edges / EntityMPI init) is excluded from exchange timings.
