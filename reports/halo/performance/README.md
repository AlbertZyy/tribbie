# Halo Performance Reports

本目录保存可复现的 Halo 性能结果。每个运行目录同时保存原始 JSON 与 Markdown 报告；不要只保留渲染后的摘要。

## Runs

- `2026-08-09-mpi1-mpi2-overlap-entities4096-halo1024/`
  - `mpi1.json` / `mpi1.md`：单进程空通信基线。
  - `mpi2.json` / `mpi2.md`：双进程环形通信。
  - `mpi2-compute1024.json` / `mpi2-compute1024.md`：双进程环形通信与可调本地计算负载。

报告生成程序：`benchmarks/halo/render_report.py`。
