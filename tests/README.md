# Tests

阶段 A 建立测试入口与契约测试；通信行为从阶段 B 开始实现。

## 本地

```bash
uv run pytest tests/unit -q
```

## MPI

```bash
mpiexec -n 1 uv run pytest tests/mpi -q
mpiexec -n 2 uv run pytest tests/mpi -q
mpiexec -n 4 uv run pytest tests/mpi -q
```

MPI 测试必须配置外部超时。阶段 A 的 MPI 测试只验证入口和 communicator 可用，不宣称 Halo 通信正确性。
