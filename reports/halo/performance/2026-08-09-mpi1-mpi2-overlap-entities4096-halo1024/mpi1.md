# Halo Performance Report

- Timestamp (UTC): `2026-08-09T06:40:46.261522+00:00`
- MPI processes: `1`
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

- Build time (max rank): `7.390000e-05` s
- Neighbors: `[]`
- Send entities: `0`
- Receive entities: `0`

## Measurements

| Case | Median (s) | P10 (s) | P90 (s) | Max (s) | Payload (B) | Effective bandwidth (B/s) | Peak temp memory (B) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `plan_replace_blocking` | 1.990001e-05 | 1.939000e-05 | 3.419000e-05 | 4.400000e-05 | 0 | 0.000000e+00 | 5041 |
| `plan_replace_nonblocking` | 2.175001e-05 | 2.158000e-05 | 3.362000e-05 | 4.370000e-05 | 0 | 0.000000e+00 | 4745 |
| `plan_sum_blocking` | 1.950000e-05 | 1.919000e-05 | 2.273000e-05 | 2.660000e-05 | 0 | 0.000000e+00 | 4769 |
| `plan_sum_nonblocking` | 2.165000e-05 | 2.139000e-05 | 2.420001e-05 | 2.779999e-05 | 0 | 0.000000e+00 | 4769 |
| `baseline_typed_nonblocking` | 7.999988e-07 | 7.000039e-07 | 1.199996e-06 | 1.200009e-06 | 0 | 0.000000e+00 | 0 |

## Interpretation

- Timing values are communicator-wide maximum rank times.
- `plan_*` cases use the current reusable-buffer Halo implementation.
- `baseline_typed_nonblocking` uses direct typed `Irecv`/`Isend` buffers and does not use Python object payload communication.
- The benchmark records end-to-end exchange time; it does not claim an independently instrumented pack/MPI/unpack decomposition.

## Raw Notes

- Timing samples are communicator-wide maximum rank times.
- The baseline uses typed Irecv/Isend buffers without plan metadata.
- Construction metadata exchange is excluded from runtime exchange timings.
