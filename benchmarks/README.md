# Benchmarks

仓库按模块归档基准程序与报告：

```text
benchmarks/<module>/             # 可执行基准程序
reports/<module>/performance/   # 原始 JSON 与 Markdown 报告
```

当前 Halo 基准位于 `benchmarks/halo/`，报告位于 `reports/halo/performance/`。

## Halo

```bash
uv run python benchmarks/halo/microbenchmark.py \
  --entities 4096 --halo-entities 1024 --components 1 \
  --warmup 3 --repeats 10 \
  --output reports/halo/performance/<run-id>/mpi1.json

mpiexec -n 2 uv run python benchmarks/halo/microbenchmark.py \
  --entities 4096 --halo-entities 1024 --components 1 \
  --warmup 3 --repeats 10 \
  --output reports/halo/performance/<run-id>/mpi2.json

uv run python benchmarks/halo/render_report.py \
  reports/halo/performance/<run-id>/mpi2.json \
  reports/halo/performance/<run-id>/mpi2.md
```

`--entities` 是本地 payload 长度，`--halo-entities` 是每个方向的 Halo 实体数。报告同时记录阻塞/非阻塞、`replace`/`sum`、typed-buffer baseline，以及可选的 `--compute` 重叠负载。
