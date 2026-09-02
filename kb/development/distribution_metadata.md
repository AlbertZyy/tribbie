# Distribution Metadata 与执行计划边界

本文记录 `tribbie.distribution` 模块的设计依据与取舍，作为“持久结构语义（metadata）”与“可派生执行计划（HaloPlan）”之间边界的开发备忘。

## 1. 核心分离

- **Distribution Metadata** 描述数据“如何分布”，是**持久的结构语义**；
- **HaloPlan** 描述一次特定通信“如何执行”，是**可派生、可缓存的执行计划**。

$$
\text{HaloPlan} = f(\text{Distribution Metadata}, \text{CommContext}, \text{Exchange Semantics})
$$

判据：**即使删除所有 `HaloPlan`，系统仍应完整知道每个局部对象是谁、归谁所有、它在全局分布中的位置**；只是暂时不知道如何高效通信。

## 2. 责任归属表

| 内容 | Distribution Metadata | HaloPlan |
| --- | :---: | :---: |
| owned/ghost 分类 | 是 | 否 |
| global ID | 是 | 否 |
| owner rank | 是 | 否 |
| global size | 是 | 否 |
| 分布版本 | 是 | 引用/记录（延后） |
| peer 列表 | 可派生 | 是 |
| send/recv local indices | 可派生 | 是 |
| buffer/count/offset | 否 | 是 |
| MPI request/tag | 否 | 是 |
| I/O 所需身份信息 | 是 | 否 |

## 3. 关键设计决策

1. **方案 B（纯 metadata，无内嵌 communicator）**：`IndexDistribution` 只持久 `self_rank: int`，通信由 `make_halo_plan(distribution, comm, ...)` 显式传入。优点：metadata 可序列化、可单元测试、可 checkpoint；rank 空间校验（owner 越界、owner 一致性）推迟到 plan 构建期由 halo 集体完成。
2. **owned/ghost 由 `owners` 数组承载**：`owners[i] == self_rank` 即 owned，否则 ghost。通用布局不强制 owned-first、不重排，保留调用方本地顺序。
3. **不存储 per-entity 变长 `sharing_ranks`**：内存开销大。owned 实体被哪些 rank 引用由 `HaloPlan` 的 send peer 信息表示；仅网格重分区/拓扑修改等场景才需额外维护共享进程集合。
4. **最小持久字段**：`global_ids`（或可计算的 owned 区间）、`owners`（每个 ghost 的 owner，owned 隐式 `self_rank`）、`global_size`、`self_rank`、`version`。其余（peer 列表、收发索引、count/displacement、buffer、datatype）均可通过通信重建。

## 4. 两种本地布局

- **`GENERAL`（本次实现）**：调用方顺序，`HaloPlan` 通过 `send_indices` + `recv_indices` 双向通信。
- **`OWNED_FIRST_GHOST_SORTED`（预留）**：owned 在前，ghost 按 owner rank + 局部顺序排序；`HaloPlan` 只需 `send_indices`（owner→ghost）或 `recv_indices`（ghost→owner），通信双方不再对称。因 `HaloPlan` 侧尚未支持非对称索引，本次仅预留 `DistributionLayout` 枚举、`is_owned_first` 属性与 `from_owned_first_ghost_sorted` 构造桩，请求即抛 `UnsupportedLayoutError`。

## 5. 版本与失效（延后）

分布可能因重分区、自适应加密、ghost 重建、DOF 重编号、约束变化、communicator 改变而失效。`IndexDistribution.version` 已作为持久字段保存。将 `version` 绑定到 `HaloPlan`（如给 `HaloPlan` 增加不透明 `source_version` 字段，并在执行前校验）需要 halo 侧改动，与 sorted 布局一并延后；当前约定为“plan 仅在 `distribution.version` 不变期间有效”。

## 6. 上层组合（不在本模块范围）

`IndexDistribution` 只面向一种编号空间。网格/DOF 分布应作为上层组合：

- `MeshDistribution`：持有顶点/边/面/单元的多个 `IndexDistribution` + ghost 深度 + 分区版本；
- `DofDistribution`：`IndexDistribution` + cell-dof 映射 + 方向 + 约束。

本模块不引入 `MeshDistribution`/`DofDistribution`/`CombineOperation`/GPU staging/MPI datatype 等执行层或具体对象概念。
