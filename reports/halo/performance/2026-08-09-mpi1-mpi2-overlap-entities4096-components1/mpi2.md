# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:32:40.263101+00:00`
- MPI processes: `2`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`

## Parameters

- `entities`: `4096`
- `components`: `1`
- `warmup`: `3`
- `repeats`: `10`
- `compute`: `0`

## Plan

- Build time (max rank): `7.415500e-03` s
- Neighbors: `[1]`
- Send entities: `2`
- Receive entities: `2`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 4.985000e-05 | 4.352000e-05 | 7.554000e-05 | 8.040000e-05 | 16 | 3.209629e+05 | 4857 |
| `plan_replace_nonblocking` | 4.520000e-05 | 4.379000e-05 | 5.277000e-05 | 6.060000e-05 | 16 | 3.539823e+05 | 5529 |
| `plan_sum_blocking` | 4.875000e-05 | 4.769000e-05 | 6.156000e-05 | 7.110000e-05 | 16 | 3.282051e+05 | 4689 |
| `plan_sum_nonblocking` | 4.925000e-05 | 4.878999e-05 | 5.711001e-05 | 6.080000e-05 | 16 | 3.248731e+05 | 5529 |
| `baseline_typed_nonblocking` | 5.700000e-06 | 5.589999e-06 | 6.190003e-06 | 7.000010e-06 | 16 | 2.807018e+06 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
