# Halo Performance Protocol

## Program layout

- `benchmarks/halo/microbenchmark.py`：执行 Halo 交换微基准。
- `benchmarks/halo/render_report.py`：把机器可读 JSON 渲染为 Markdown。
- `reports/halo/performance/<run-id>/`：按模块和运行参数归档原始结果与报告。

其他模块应使用相同的 `benchmarks/<module>/` 与 `reports/<module>/performance/` 结构，不把模块专属报告放在仓库根目录。

## Metrics

记录计划构建时间、端到端交换时间、发送/接收字节数、有效带宽、采样分位数和临时内存峰值。并行时间取 communicator 内最慢 rank。公式为 `B_p = H_p K s` 和 `W_p = B_p / T_p`。

当前基准同时测量：

- 当前可复用 buffer 的 `replace`/`sum` 阻塞实现；
- 当前可复用 buffer 的 `replace`/`sum` 非阻塞实现；
- 逐邻居 typed-buffer `Irecv`/`Isend` + `Waitall` 对照基线；
- 可选 `--compute N` 的非阻塞本地计算负载。

## Reproducibility

每个配置包含预热和重复采样，并保存 timestamp、Python、平台、MPI 进程数、payload 参数、Halo 实体数和原始 JSON。示例：

```bash
mpiexec -n 2 uv run python benchmarks/halo/microbenchmark.py \
  --entities 4096 --halo-entities 1024 --components 1 \
  --warmup 3 --repeats 10 --output reports/halo/performance/<run-id>/mpi2.json
uv run python benchmarks/halo/render_report.py \
  reports/halo/performance/<run-id>/mpi2.json \
  reports/halo/performance/<run-id>/mpi2.md
```

## Current report

真实运行报告归档于：

`reports/halo/performance/2026-08-09-mpi1-mpi2-overlap-entities4096-halo1024/`

该运行覆盖 MPI 1、MPI 2，以及 MPI 2 加 `--compute 1024`。环境为 Python 3.12.12、Microsoft MPI 10.1.12498.52、Windows 11 主机；绝对时间仅作为本机基线，不作为跨平台验收阈值。

## Interpretation and limitations

- 运行期仅传输真实邻居的 typed NumPy payload；全局编号和索引不进入运行期消息。
- 当前报告的时间是端到端交换时间；没有声称通过内部 instrumentation 精确分离打包、MPI 等待和解包时间。
- 阶段 F 将重复索引 `sum` 解包从 Python 逐项循环优化为 `numpy.add.at`；在本报告的双进程、2048 接收实体配置中，`sum` 中位数约为 `5.9e-5` s，已接近 `replace` 的 `7.0e-5` s。该数值是本机样本，不是跨平台保证。
- baseline 不使用 Python 对象 payload 通信。
- 当前基准使用环形拓扑；弱/强扩展和多 payload/dtype 的完整矩阵应在目标硬件上继续采集。
- `tracemalloc` 只反映 Python 分配，不等价于 NumPy/MPI 原生分配峰值。
- 当前平台没有跨节点绑定与网络隔离实验，因此报告不外推网络性能结论。
