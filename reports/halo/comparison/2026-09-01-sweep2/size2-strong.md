# HaloPlan.exchange vs EntityMPI.sync — Comparison Report

- Timestamp (UTC): `2026-09-01T02:01:21.443555+00:00`
- MPI processes: `2`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`
- fealpy path: `D:\suanhai_repo\fealpy`

## Parameters

- entities_list: `[500000]`
- halo_list: `[31250]`
- components_list: `[1]`
- warmup: `5`, repeats: `20`

## Results

`speedup = entity_mpi_median / halo_plan_median`; values > 1 mean HaloPlan is faster.

| E | H | C | send | payload (B) | sync idx (B) | correct | replace: halo/empi (s) | replace speedup | sum: halo/empi (s) | sum speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500000 | 31250 | 1 | 62500 | 500000 | 500000 | True | 5.731e-04 / 2.673e-03 | 4.67x | 3.044e-04 / 4.369e-03 | 14.36x |

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
