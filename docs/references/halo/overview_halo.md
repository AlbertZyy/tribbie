# Halo Overview

halo 模块为任意**分布式编号集**提供静态计划的稀疏 Halo 数据交换：通信关系（共享实体、owner、收发索引）在计划构建期确定并缓存，运行期只与真实邻居传输数值载荷。本页从数学角度定义 `HaloPlan.exchange` / `HaloPlan.begin_exchange` 的执行过程，给出面向用户的接口 Reference，并附两个使用范例。

## 1. 数学定义

### 1.1 记号与数据模型

设共有 $P$ 个进程，rank $p \in \{0, 1, \dots, P-1\}$。

- **本地实体集**：$I_p = \{0, 1, \dots, n_p-1\}$，进程 $p$ 持有 $n_p$ 个本地实体；
- **全局编号**：$g$ 是跨进程唯一的标识符，其 owner 为 rank $o(g)$；同一实体可被多个进程持有（owner 一份权威值，其余为 ghost 副本）；
- **载荷数组**：$A \in \mathbb{K}^{\,n_p \times k_1 \times \cdots \times k_d}$，第一维对应本地实体，其余维度是每个实体的 payload；$\mathbb{K} \in \{\texttt{int32}, \texttt{int64}, \texttt{float32}, \texttt{float64}, \texttt{complex64}, \texttt{complex128}\}$；
- **目标数组**：$B$，与 $A$ 形状、dtype 相同；$B = A$（即 `dst=None`）表示原地交换。

### 1.2 通信计划：静态有向边

一个 `HaloPlan` 固定一组有向边。对每个真实邻居 $q$：

$$
\mathcal{E}_p = \{\, e = (q,\ S_{p,q},\ R_{p,q}) : q \in \text{neighbors}(p) \,\},
$$

其中：

- $S_{p,q} = (s_1, \dots, s_{m_{p,q}})$：**发送索引**，从本进程源数组打包的位置；
- $R_{p,q} = (r_1, \dots, r_{\ell_{p,q}})$：**接收索引**，接收数据写入本进程目标数组的位置。

计划构建期集体校验如下**一致性不变量**（`from_edges` / `from_global_ids` 均执行）：

$$
m_{p,q} = \ell_{q,p} \qquad \forall\, (p, q),
$$

即 $p$ 发给 $q$ 的实体数等于 $q$ 从 $p$ 接收的实体数，且顺序对齐：$p$ 发出的第 $i$ 个元素到达 $q$ 后成为其第 $i$ 个接收槽。

`from_global_ids(comm, global_ids, owners, *, direction=...)` 在构建期完成**分布式 owner 查询**：ghost 把自己的全局编号一次性 typed `Alltoallv` 发给对应 owner，owner 用 `searchsorted` 映射为本地索引，得到两个方向共用的共享关系。`direction` 决定返回什么：

- `"owner_to_ghost"`（默认，owner 只发送、ghost 只接收）：

$$
\begin{aligned}
\text{owner 边}: &\quad e = (p,\ S_{o,p} \ni \text{idx}_o(g),\ R_{o,p} = \varnothing), \\[2pt]
\text{ghost 边}: &\quad e = (o,\ S_{p,o} = \varnothing,\ R_{p,o} \ni \text{idx}_p(g)).
\end{aligned}
$$

- `"ghost_to_owner"`（ghost 只发送、owner 只接收，即把上式中的 $S$ 与 $R$ 互换）。
- `"two_way"`（默认）：一次发现同时返回 `(ghost_to_owner, owner_to_ghost)` 两个方向。

运行期不再传输任何全局编号或索引，只交换数值载荷。

### 1.3 `exchange` / `begin_exchange` 的数学过程

一次交换对每条边依次执行三步，整体写作：

$$
B = \text{Scatter}\bigl(\text{Transmit}\bigl(\text{Gather}(A,\ \mathcal{E}_p)\bigr),\ \mathcal{E}_p,\ \oplus\bigr),
\qquad \oplus \in \{\text{replace},\ \text{sum}\}.
$$

**第 1 步 — 稀疏收集（Gather）**：对每条边，从源数组按行选取子数组

$$
G_{p,q} = A[S_{p,q}], \qquad (G_{p,q})_i = A_{s_i}, \quad i = 1, \dots, m_{p,q}.
$$

**第 2 步 — 传输（Transmit）**：把 $G_{p,q}$ 发送给 $q$，同时接收 $\tilde G_{p,q} = G_{q,p}$（对端以同样规则收集其自身源数组）。每条边投递一个 typed `Irecv` 与一个 typed `Isend`（tag 通过 `comm.Dup()` 复制的通信子隔离），只与真实邻居通信；通信量即收集后的载荷，不携带索引。

**第 3 步 — 散射与归并（Scatter with $\oplus$）**：把收到的 $\tilde G_{p,q}$ 写入目标数组 $B$ 的接收位置：

- **replace**（覆盖）：

  $$
  B_{r_i} \leftarrow (\tilde G_{p,q})_i, \qquad i = 1, \dots, \ell_{p,q}.
  $$

  单条边内接收索引必须互异，否则交换前抛 `ReplaceConflictError`；跨边写同一位置由调用方保证不发生（实现按边序依次写入）。

- **sum**（累加，目标原值保留）：

  $$
  B_r \leftarrow B_r + \sum_{i\,:\, r_i = r} (\tilde G_{p,q})_i, \qquad \forall r.
  $$

  重复接收索引按贡献逐个累加（`np.add.at`），不依赖高级索引赋值的未定义行为；如需从零归并，调用方先清零目标位置。

**原地语义**：`dst=None` 时 $B = A$。由于第 1 步的收集先于任何接收写入（接收先落入独立缓冲区，全部传输完成后再散射），原地交换安全，不会读到被覆盖的源数据。

**阻塞与非阻塞**：`exchange` 与 `begin_exchange` 执行同一个数学过程，区别仅在时序——`begin_exchange` 投递第 1、2 步的请求后立即返回 `HaloRequest`，第 3 步推迟到请求完成：

- `request.wait()`：阻塞至所有 MPI 请求完成，然后执行散射并返回 $B$；
- `request.test()`：非阻塞轮询，完成时同样执行散射；
- 请求存活期间调用方不得修改参与发送的 $A$、$B$ 或计划内部缓冲区。

### 1.4 归约与广播的复合

`HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, values)` 串行复合两个方向计划（可用 `from_global_ids(..., direction="two_way")` 或 `from_global_ids(direction=...)` 构造，也可用 `from_edges` 显式构造）：

$$
R = \text{reduce\_plan.exchange}(values,\ op=\texttt{"sum"}), \qquad
B = \text{broadcast\_plan.exchange}(R,\ op=\texttt{"replace"}),
$$

数学效果为

$$
\text{owner} \leftarrow \text{owner} + \sum_{g \neq o} c_g, \qquad
\text{ghost} \leftarrow \text{owner 的归约结果},
$$

其中 $c_g$ 是各 ghost 的局部贡献。典型构造（示例 2）：`reduce_plan = from_global_ids(..., direction="ghost_to_owner")`、`broadcast_plan = from_global_ids(..., direction="owner_to_ghost")`，或默认 `direction="two_way"` 一次得到 `(reduce_plan, broadcast_plan)`；也可用 `from_edges` 显式给边。该助手不推断 owner、不构造全局索引映射。

## 2. 接口 Reference（Overview）

| 接口                              | 签名                                                                                                                       | 对应的数学过程                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `HaloEdge`                      | `HaloEdge(peer, send_indices, recv_indices)`                                                                             | 一条有向边 $e = (q, S_{p,q}, R_{p,q})$                     |
| `HaloPlan.from_global_ids`      | `from_global_ids(comm, global_ids, owners, *, direction="two_way", validation="basic")` | 由全局编号分布式发现共享关系，按 `direction` 返回 owner→ghost / ghost→owner 单方向计划，或 `two_way` 返回 `(ghost_to_owner, owner_to_ghost)` |
| `HaloPlan.from_edges`           | `from_edges(comm, edges, *, validation="basic", entity_count=None)`                                                      | 由显式边集构造计划：按 peer 合并、确定性排序、集体校验 $m_{p,q} = \ell_{q,p}$ |
| `plan.exchange`                 | `exchange(src, dst=None, *, op="replace") -> dst`                                                                        | 阻塞执行 Gather → Transmit → Scatter($\oplus$) 并返回 $B$    |
| `plan.begin_exchange`           | `begin_exchange(src, dst=None, *, op="replace") -> HaloRequest`                                                          | 同 `exchange` 的数学过程，但投递非阻塞请求后立即返回                      |
| `HaloRequest.wait`              | `wait(timeout=None) -> dst`                                                                                              | 阻塞至传输完成，执行散射后返回 $B$                                   |
| `HaloRequest.test`              | `test() -> (bool, result)`                                                                                               | 非阻塞轮询传输是否完成，完成则执行散射                                   |
| `HaloRequest.completed`         | 只读属性                                                                                                                     | 请求是否已观察到终态                                            |
| `HaloPlan.reduce_and_broadcast` | `reduce_and_broadcast(reduce_plan, broadcast_plan, values, *, dst=None)`                                                 | owner ← owner + Σ贡献；ghost ← owner（§1.4）               |
| `plan.close`                    | `close()`                                                                                                                | 释放计划复制的通信子；存在未完成请求时拒绝关闭                               |
| 元数据属性                           | `edges`, `neighbors`, `send_counts`, `recv_counts`, `total_send_count`, `total_recv_count`, `supported_dtypes`, `closed` | 计划静态信息：邻居、每邻居/总计收发实体数、支持 dtype、关闭状态                   |

其他要点：

- `validation`：`"basic"` 做本地校验与分布式 owner 查询；`"full"` 额外跨进程核对每个全局编号的 owner 一致且 owner 确实持有该编号；`"none"` 跳过完整一致性检查；
- `sum` 保留目标原值（`dst[index] += value`）；`replace` 覆盖目标；
- 同一计划同一时刻只允许一个未完成请求（否则抛 `ConcurrentRequestError`）；dtype / 形状 / 实体数不匹配、已关闭计划被使用等均有专用异常（均继承 `HaloError`）；
- 单进程（comm 大小 1）与空编号集：计划为空或走本地拷贝分支，语义仍然成立。

## 3. 使用范例

以下示例需以多进程方式运行，例如 `mpiexec -n 3 python 示例.py`。

### 示例 1：分布式顶点位移同步（owner → ghost，`replace`，`from_global_ids`）

情景：三个进程共享全局顶点 `42`，其权威值由 rank 0 持有。每轮迭代后，把 owner 的位移同步到所有副本，保证后续计算读到的边界值一致。

```python
from mpi4py import MPI
import numpy as np

from tribbie.halo import HaloPlan

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 本地实体：全局编号 42，owner 为 rank 0（三个进程各持有一份副本）
ids = np.array([42], dtype=np.int64)
owners = np.array([0], dtype=np.int32)
plan = HaloPlan.from_global_ids(comm, ids, owners, validation="full")

# 每轮迭代后：把 owner 的权威位移同步到所有副本（原地 replace）
displacement = np.array([1.0 * rank], dtype=np.float64)  # rank0: 0.0, rank1: 1.0, ...
synced = plan.exchange(displacement, op="replace")

# 三个进程现在都持有 owner 的值 0.0
assert synced[0] == 0.0
plan.close()
```

### 示例 2：两进程贡献归约并广播（ghost → owner `sum` + owner → ghost `replace`，`from_edges`）

情景：rank 0 拥有实体、rank 1 持有副本，双方各自产生局部贡献（例如有限元装配的局部残差）。需要把贡献归约到 owner 求和，再把结果广播回所有副本。

```python
from mpi4py import MPI
import numpy as np

from tribbie.halo import HaloEdge, HaloPlan

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
peer = 1 - rank
owner = 0

# 归约方向：ghost(1) 只发送贡献，owner(0) 只接收并累加
reduce_edge = HaloEdge(peer, [0], []) if rank != owner else HaloEdge(peer, [], [0])
reduce_plan = HaloPlan.from_edges(comm, [reduce_edge], entity_count=1)

# 广播方向：owner(0) 只发送归约结果，ghost(1) 只接收
broadcast_edge = HaloEdge(peer, [], [0]) if rank != owner else HaloEdge(peer, [0], [])
broadcast_plan = HaloPlan.from_edges(comm, [broadcast_edge], entity_count=1)

local = np.array([float(rank + 1)])  # rank0 贡献 1.0，rank1 贡献 2.0
total = HaloPlan.reduce_and_broadcast(reduce_plan, broadcast_plan, local)
assert total[0] == 3.0               # 双方都得到 1.0 + 2.0
```

> 说明：`from_edges` 校验收发计数对称（$m_{p,q} = \ell_{q,p}$），因此单方向的归约/广播计划需要像上面这样把发送边与接收边分别放在两端；若两进程做对称的 peer exchange，也可直接使用双向边 `HaloEdge(peer, [0], [0])`。等价地，`reduce_plan, broadcast_plan = HaloPlan.from_global_ids(comm, ids, owners, validation="full")`（`ids`/`owners` 同示例 1，默认 `two_way`）可一次发现得到同样两个方向。

## 4. 进一步阅读

- 详细 API 契约与参数、异常约定：[`docs/api.md`](../../api.md)
- `replace` / `sum` / 各通信方向（reduce-to-owner、owner-to-ghost、reduce-and-broadcast、peer exchange）的语义说明：[`docs/semantics.md`](../../semantics.md)
