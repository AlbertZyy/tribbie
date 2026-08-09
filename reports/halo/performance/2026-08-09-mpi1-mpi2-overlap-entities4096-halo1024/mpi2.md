# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:40:46.962671+00:00`
- MPI processes: `2`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`

## Parameters

- `entities`: `4096`
- `halo_entities`: `1024`
- `components`: `1`
- `warmup`: `3`
- `repeats`: `10`
- `compute`: `0`

## Plan

- Build time (max rank): `6.082200e-03` s
- Neighbors: `[1]`
- Send entities: `2048`
- Receive entities: `2048`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 7.020000e-05 | 5.998999e-05 | 1.023700e-04 | 1.111000e-04 | 16384 | 2.333903e+08 | 17416 |
| `plan_replace_nonblocking` | 6.460000e-05 | 5.798000e-05 | 9.626000e-05 | 1.265000e-04 | 16384 | 2.536223e+08 | 18088 |
| `plan_sum_blocking` | 5.865000e-05 | 5.700000e-05 | 7.296000e-05 | 9.690000e-05 | 16384 | 2.793521e+08 | 17248 |
| `plan_sum_nonblocking` | 5.990000e-05 | 5.922000e-05 | 7.205001e-05 | 7.700000e-05 | 16384 | 2.735225e+08 | 18088 |
| `baseline_typed_nonblocking` | 1.629999e-05 | 1.578999e-05 | 1.871001e-05 | 2.150000e-05 | 16384 | 1.005154e+09 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
