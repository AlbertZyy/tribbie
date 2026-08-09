# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:32:39.393768+00:00`
- MPI processes: `1`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`

## Parameters

- `entities`: `4096`
- `components`: `1`
- `warmup`: `3`
- `repeats`: `10`
- `compute`: `0`

## Plan

- Build time (max rank): `7.400000e-05` s
- Neighbors: `[]`
- Send entities: `0`
- Receive entities: `0`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 1.965000e-05 | 1.919000e-05 | 3.285001e-05 | 4.140000e-05 | 0 | 0.000000e+00 | 5041 |
| `plan_replace_nonblocking` | 2.345000e-05 | 2.228001e-05 | 3.494001e-05 | 4.519999e-05 | 0 | 0.000000e+00 | 4745 |
| `plan_sum_blocking` | 1.995000e-05 | 1.976000e-05 | 2.336000e-05 | 2.840000e-05 | 0 | 0.000000e+00 | 4769 |
| `plan_sum_nonblocking` | 2.200000e-05 | 2.170000e-05 | 2.475000e-05 | 2.789999e-05 | 0 | 0.000000e+00 | 4769 |
| `baseline_typed_nonblocking` | 7.999988e-07 | 7.000039e-07 | 1.109994e-06 | 1.199995e-06 | 0 | 0.000000e+00 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
