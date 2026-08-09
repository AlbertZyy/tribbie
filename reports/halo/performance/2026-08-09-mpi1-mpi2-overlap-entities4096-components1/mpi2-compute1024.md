# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:32:41.051685+00:00`
- MPI processes: `2`
- Platform: `Windows-11-10.0.26200-SP0`
- Python: `3.12.12`

## Parameters

- `entities`: `4096`
- `components`: `1`
- `warmup`: `3`
- `repeats`: `10`
- `compute`: `1024`

## Plan

- Build time (max rank): `6.255600e-03` s
- Neighbors: `[1]`
- Send entities: `2`
- Receive entities: `2`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 5.280000e-05 | 4.507999e-05 | 9.333001e-05 | 9.720000e-05 | 16 | 3.030303e+05 | 4857 |
| `plan_replace_nonblocking` | 6.210001e-05 | 5.131000e-05 | 1.129100e-04 | 2.147000e-04 | 16 | 2.576489e+05 | 5529 |
| `plan_sum_blocking` | 4.900000e-05 | 4.807000e-05 | 5.728001e-05 | 6.700000e-05 | 16 | 3.265306e+05 | 4689 |
| `plan_sum_nonblocking` | 5.515001e-05 | 5.367000e-05 | 6.842999e-05 | 7.500000e-05 | 16 | 2.901178e+05 | 5529 |
| `baseline_typed_nonblocking` | 5.599999e-06 | 5.499998e-06 | 6.070001e-06 | 7.600000e-06 | 16 | 2.857143e+06 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
