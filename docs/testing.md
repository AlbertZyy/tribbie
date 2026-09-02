# Testing Protocol

## Unit matrix

阶段 A 覆盖公共异常层级与请求状态。阶段 B 增加直接边、peer/index/计数验证、元数据只读性、关闭行为、payload 校验、空通信、原地打包和固定形状 buffer 复用。阶段 C 增加 `test()`/`wait()`、重复等待、非法并发、请求存活期间关闭、计算重叠和阻塞/非阻塞结果一致性。 阶段 E 增加全局 ID 的 int64/唯一性/owner 校验、乱序非连续 ID、多方共享、构建期映射和跨 rank 失败一致性测试。

## MPI matrix

入口必须支持 `mpiexec -n 1/2/4 python -m pytest tests/mpi`。阶段 B 起覆盖单进程、对称/非对称双进程、链、环、星形、不连通图、多方共享、零实体和非连续 ID；每种拓扑覆盖阻塞/非阻塞、两种操作、原地/非原地、标量/多分量及必需 dtype。

MPI 测试必须设置超时，并在失败信息中保留 rank、阶段、peer、请求类型与通信计数。CI 至少运行 1、2、4 进程。

## Determinism and properties

使用 `value(global_id, rank, component)` 计算期望值。属性测试随机化邻接、ID、owner、本地排列、副本数和 payload 形状，并比较阻塞/非阻塞结果及多轮 buffer 复用。

## Distribution matrix

`distribution` 模块覆盖 `tests/unit/test_distribution.py` 与 `tests/mpi/test_distribution.py`。

单元测试覆盖：

- `DistributionVersion` 冻结/相等/哈希/`zero()`；
- 通用布局的 owned/ghost 分类、`owned_size/ghost_size/local_size/global_size`、`local_to_global`/`global_to_local`/`owner`/`is_owned`/`is_ghost` 查询与越界异常；
- 校验错误：负/超界/非 int64 global id、重复 id、owners 非法（长度/负/非整数）、`global_size<0`、`self_rank<0`；
- 只读性（返回视图不可写、构造期拷贝隔离）与 `to_dict`/`from_dict` 往返；
- 预留 sorted 布局：`layout=OWNED_FIRST_GHOST_SORTED` 与 `from_owned_first_ghost_sorted` 抛 `UnsupportedLayoutError`、`is_owned_first` 为 `False`；
- 单进程 `make_halo_plan` 三个方向返回空计划；`self_rank` 与 comm rank 不符抛 `RankMismatchError`。

MPI 测试覆盖（`mpiexec -n 1/2/4`）：

- 2-rank 双向 `two_way` 后 `reduce_and_broadcast` 结果一致；
- 混合 owned/ghost 与非连续/乱序 id 的正确性；
- 3-rank 多方共享（owner + 两个 ghost）；
- 空分布/空 ghost 进程；
- `validation="full"` 检测跨 rank owner 不一致（所有 rank 一致抛错）；
- `self_rank` 与 comm rank 不符的集体失败（不挂死）。
