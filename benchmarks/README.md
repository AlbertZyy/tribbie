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

## Halo comparison（`HaloPlan.exchange` vs `EntityMPI.sync`）

跨实现对比基准，用于回答「tribbie 的 `HaloPlan.exchange` 与 fealpy 的 `EntityMPI.sync` 哪个更快」。程序位于 `benchmarks/halo/comparison/`：

- `benchmarks/halo/comparison/compare_sync.py`：单次 `mpiexec` 启动内对当前 `size` 循环数据规模轴（`--entities-list` × `--halo-list` × `--components-list`），先做数值等价断言再计时，输出 `reports/halo/comparison/<run-id>/size<N>-<tag>.json`。
- `benchmarks/halo/comparison/render_comparison.py`：把 JSON 渲染成含加速比的 Markdown。
- `benchmarks/halo/comparison/run_comparison.ps1`：按弱扩展 / 强扩展 / 数据规模三种场景遍历 `size ∈ {2,4,8,16}` 并调用 `mpiexec`。

依赖与前置：

```powershell
# 1) 环境依赖（一次性）：scipy 与 tqdm 供 fealpy 的 numpy 后端/日志使用
uv pip install --python .venv\Scripts\python.exe scipy tqdm

# 2) fealpy 通过 --fealpy-path 指向源码树（默认 D:\suanhai_repo\fealpy），
#    compare_sync.py 内部用轻量 shim 只加载 entity_mpi.py，不触发 mesh/matplotlib。

# 3) 多进程启动需要 MPI 进程管理器（二选一）：
#    (a) 管理员启动 MS-MPI Launch Service：
#         Start-Service MsMpiLaunchSvc
#    (b) 普通用户后台跑 smpd 守护进程（无需管理员）：
#         & "C:\Program Files\Microsoft MPI\Bin\smpd.exe" -d
```

运行（弱扩展 / 强扩展 / 数据规模三场景，`size ∈ {2,4,8,16}`）：

```powershell
powershell -ExecutionPolicy Bypass -File benchmarks/halo/comparison/run_comparison.ps1 -RunId <run-id>
```

单次手动示例（`size=4`、`E=10^5`、`H=1024`）：

```powershell
mpiexec -n 4 .venv\Scripts\python.exe benchmarks/halo/comparison/compare_sync.py `
  --entities-list 100000 --halo-list 1024 --components-list 1 `
  --warmup 5 --repeats 20 `
  --output reports/halo/comparison/<run-id>/size4-weak.json

.venv\Scripts\python.exe benchmarks/halo/comparison/render_comparison.py `
  reports/halo/comparison/<run-id>/size4-weak.json `
  reports/halo/comparison/<run-id>/size4-weak.md
```

**语义对齐**：两侧构造「1-D 周期链、每 rank 左右各共享 `H` 个 halo 实体」的同一问题，并断言 `exchange(op="sum") == sync`+in-place add、`exchange(op="replace") == sync`+in-place scatter；两侧都就地 apply（无整数组拷贝），隔离出纯通信路径差异。`speedup = entity_mpi / halo_plan`，`>1` 表示 `HaloPlan` 更快。
