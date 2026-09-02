# Distribution Overview

`distribution` 模块为**任意抽象编号空间**提供持久化的分布式元数据：它只回答“每个本地对象是谁、归谁所有、在全局编号空间中的位置”，不回答“如何通信”。通信执行计划由 `halo` 模块的 `HaloPlan` 负责，二者通过 `make_halo_plan` 桥接。

核心对象 `IndexDistribution` 面向一种编号空间（顶点、边、面、单元、自由度、矩阵行、粒子、约束变量……），不耦合网格、DOF、有限元等具体概念，也不持有 `MPI.Comm`（owner rank 以普通整数 `self_rank` 记录）。

## 1. 数学定义

### 1.1 数据模型

设进程 $p \in \{0,\dots,P-1\}$ 持有 $n_p$ 个本地实体，本地索引空间 $I_p = \{0,\dots,n_p-1\}$。对每个本地索引 $i$：

- **全局编号**：$g_i = \text{global\_ids}[i] \in \mathbb{Z}_{64}$，跨进程唯一且 $0 \le g_i < G$（$G$ 为 `global_size`）；
- **owner**：$o_i = \text{owners}[i] \in \mathbb{Z}_{32}$，即该实体的权威 rank。

owned/ghost 分类完全由 owner 与本地 rank 的关系决定：

$$
i \text{ 是 owned} \iff o_i = p, \qquad i \text{ 是 ghost} \iff o_i \neq p.
$$

查询接口即上述语义的直接翻译：

| 方法 | 数学含义 |
| --- | --- |
| `local_to_global(i)` | $i \mapsto g_i$ |
| `global_to_local(g)` | 逆映射（偏函数）：存在时返回唯一本地索引，否则 `None` |
| `owner(i)` | $i \mapsto o_i$ |
| `is_owned(i)` / `is_ghost(i)` | $o_i = p$ / $o_i \neq p$ |
| `owned_size` / `ghost_size` / `local_size` | $\#\{i: o_i = p\}$ / $\#\{i: o_i \neq p\}$ / $n_p$ |
| `global_size` | $G$ |

### 1.2 两种本地布局

- **`DistributionLayout.GENERAL`**（本次实现）：保留调用方给定的本地顺序，owned 与 ghost 可任意交错。此时 `HaloPlan` 通过 `send_indices` 与 `recv_indices` 完成双向通信，通信双方对称。

- **`DistributionLayout.OWNED_FIRST_GHOST_SORTED`**（预留，未实现）：owned 位于 `[0, owned_size)`，ghost 紧随其后并按 owner rank、再按局部顺序排序。此时 `HaloPlan` 只需 `send_indices` 做 owner→ghost、只需 `recv_indices` 做 ghost→owner，通信双方不再对称。该布局依赖 `HaloPlan` 侧的非对称支持，故仅保留接口（`DistributionLayout` 枚举、`is_owned_first` 属性、`IndexDistribution.from_owned_first_ghost_sorted(...)` 构造桩），请求时抛 `UnsupportedLayoutError`。

### 1.3 元数据 → 计划的桥接

`make_halo_plan(distribution, comm, *, direction="two_way", validation="basic")` 是唯一需要 communicator 的入口，其数学含义等价于：

$$
\text{HaloPlan} = f(\text{IndexDistribution}, \text{comm}, \text{direction}).
$$

它先**集体**校验 `distribution.self_rank == comm.Get_rank()`（保证 owned/ghost 在正确的 rank 空间下解释），再把 `distribution.global_ids` / `distribution.owners` 交给 `HaloPlan.from_global_ids` 做分布式发现。`direction` 取值与 `HaloPlan.from_global_ids` 一致：

| `direction` | 返回 | 边方向 | 常用操作 |
| --- | --- | --- | --- |
| `"two_way"`（默认） | `(ghost_to_owner, owner_to_ghost)` | 一次发现产出双向边 | `reduce_and_broadcast` |
| `"owner_to_ghost"` | 单个 `HaloPlan` | owner 只发、ghost 只收 | `replace` |
| `"ghost_to_owner"` | 单个 `HaloPlan` | ghost 只发、owner 只收 | `sum` |

## 2. 接口 Reference（Overview）

| 接口 | 签名 | 说明 |
| --- | --- | --- |
| `IndexDistribution` | `IndexDistribution(global_ids, owners, *, global_size, self_rank, version=None, layout=GENERAL)` | 通用布局的持久元数据；构造期校验并拷贝数组 |
| `IndexDistribution.from_owned_first_ghost_sorted` | 预留构造 | 抛 `UnsupportedLayoutError` |
| 只读属性 | `self_rank` `global_size` `local_size` `owned_size` `ghost_size` `version` `layout` `is_owned_first` | 结构语义；数组视图不可写 |
| 数组属性 | `global_ids` `owners` `owned_global_ids` `ghost_global_ids` `ghost_owners` | 只读视图（`int64` / `int32`） |
| 查询 | `is_owned(i)` `is_ghost(i)` `local_to_global(i)` `global_to_local(g)` `owner(i)` | 全部无通信；越界抛 `InvalidIndexError` |
| 序列化 | `to_dict()` / `IndexDistribution.from_dict(data)` | checkpoint 独立保存/恢复分布 |
| `make_halo_plan` | `make_halo_plan(distribution, comm, *, direction="two_way", validation="basic")` | 元数据 → `HaloPlan` 桥接；集体调用 |
| `IndexDistribution.make_halo_plan` | `distribution.make_halo_plan(comm, *, direction="two_way", validation="basic")` | 上述函数的便捷方法 |
| `DistributionVersion` | `DistributionVersion(topology=0, numbering=0, ghost_layout=0)` / `.zero()` | 冻结值类型；重分区/重编号/ghost 重建时递增对应字段 |

异常（均继承 `DistributionError`）：`InvalidGlobalIdError`、`DuplicateGlobalIdError`、`InvalidOwnerError`、`InvalidIndexError`、`RankMismatchError`、`UnsupportedLayoutError`。`ghost_owners` 越出 comm size、跨 rank owner 不一致等依赖 communicator 的校验由 `make_halo_plan` 委派给 halo 的既有校验（`validation="full"` 时检测全局 owner 一致性，所有 rank 一致抛错）。

## 3. 使用范例

以下示例需以多进程方式运行，例如 `mpiexec -n 2 python 示例.py`。

### 示例：同一分布生成双向 plan 并归约广播

情景：rank 0 拥有全局实体 `9000000000007` 与 `41`，rank 1 持有这两个实体的 ghost 副本。用同一个 `IndexDistribution` 生成 `(ghost_to_owner, owner_to_ghost)` 双向计划，把两个进程的局部贡献归约到 owner 再广播回所有副本。

```python
from mpi4py import MPI
import numpy as np

from tribbie.distribution import IndexDistribution, make_halo_plan
from tribbie.halo import HaloPlan

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ids = np.array([9000000000007, 41], dtype=np.int64)
owners = np.array([0, 0], dtype=np.int32)          # owner 均为 rank 0
dist = IndexDistribution(ids, owners, global_size=10**13, self_rank=rank)

# 一次发现产出双向计划，顺序与 reduce_and_broadcast 期望一致
reduce_plan, broadcast_plan = make_halo_plan(dist, comm, direction="two_way")

local = np.array([100.0 + rank, 200.0 + rank])
total = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, local)

# 两个 rank 都得到 owner 归约结果 [301.0, 301.0]
assert np.array_equal(total, [301.0, 301.0])
```

## 4. 进一步阅读

- 元数据与执行计划的边界、责任归属表与两种布局的设计说明：[`kb/development/distribution_metadata.md`](../../../kb/development/distribution_metadata.md)
- `HaloPlan` 的执行语义与 `reduce_and_broadcast`：[`docs/references/halo/overview_halo.md`](../halo/overview_halo.md)
- 测试矩阵：[`docs/testing.md`](../../testing.md)
