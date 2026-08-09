# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:40:47.966683+00:00`
- MPI processes: `2`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`

## Parameters

- `entities`: `4096`
- `halo_entities`: `1024`
- `components`: `1`
- `warmup`: `3`
- `repeats`: `10`
- `compute`: `1024`

## Plan

- Build time (max rank): `6.314300e-03` s
- Neighbors: `[1]`
- Send entities: `2048`
- Receive entities: `2048`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 5.630001e-05 | 5.456999e-05 | 6.658000e-05 | 7.990001e-05 | 16384 | 2.910124e+08 | 17416 |
| `plan_replace_nonblocking` | 6.855000e-05 | 6.532000e-05 | 7.959001e-05 | 8.400000e-05 | 16384 | 2.390080e+08 | 18088 |
| `plan_sum_blocking` | 5.975000e-05 | 5.796000e-05 | 7.934001e-05 | 9.140001e-05 | 16384 | 2.742092e+08 | 17248 |
| `plan_sum_nonblocking` | 7.310000e-05 | 7.141001e-05 | 7.913000e-05 | 8.659999e-05 | 16384 | 2.241313e+08 | 18088 |
| `baseline_typed_nonblocking` | 1.675000e-05 | 1.626999e-05 | 1.881999e-05 | 2.260000e-05 | 16384 | 9.781495e+08 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
