# Testing Protocol

## Unit matrix

阶段 A 覆盖公共异常层级、请求 pending/completed 状态、重复 `wait()`、计划元数据只读性、关闭计划行为，以及通信方法明确延期。后续单元测试补充 dtype 映射、索引验证、buffer 尺寸和归并。

## MPI matrix

入口必须支持 `mpiexec -n 1/2/4 python -m pytest tests/mpi`。阶段 B 起覆盖单进程、对称/非对称双进程、链、环、星形、不连通图、多方共享、零实体和非连续 ID；每种拓扑覆盖阻塞/非阻塞、两种操作、原地/非原地、标量/多分量及必需 dtype。

MPI 测试必须设置超时，并在失败信息中保留 rank、阶段、peer、请求类型与通信计数。CI 至少运行 1、2、4 进程。

## Determinism and properties

使用 `value(global_id, rank, component)` 计算期望值。属性测试随机化邻接、ID、owner、本地排列、副本数和 payload 形状，并比较阻塞/非阻塞结果及多轮 buffer 复用。
