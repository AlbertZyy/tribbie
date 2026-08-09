# Halo Design Contract

## Scope

Halo 是面向通用分布式编号集的通信组件，不包含网格、节点、单元、自由度或有限元语义。每个本地实体由数组位置标识，并由 `global_ids` 映射到全局唯一编号。

## Data model

- `global_ids`: 本地实体的一维 64-bit 编号数组，进程内不得重复。
- `owners`: 可选的一维 rank 数组；提供时长度必须与 `global_ids` 相同，rank 必须属于 communicator。
- `HaloEdge`: `peer`、`send_indices`、`recv_indices`；索引在计划构建期固定。
- `HaloPlan`: 静态邻接、索引、计数和可复用 buffer 的所有者；运行期只接收 payload 数组。

阶段 B 通过 `HaloEdge` 构建直接边计划。构建期复制 communicator 以隔离消息上下文，collective 核对对端发送/接收计数；运行期只与真实邻居交换 typed-buffer payload。固定 dtype 与 payload 形状的发送、接收 buffer 跨轮复用。阶段 C 的请求持有 MPI 请求、源/目标数组及这些 buffer，直到 `test()` 或 `wait()` 确认完成。 阶段 E 的 `from_global_ids()` 要求显式 `owners`；非 owner 在构建期向 owner 请求全局编号映射，owner 返回本地索引，随后转换为 owner-to-ghost 直接边。

## Owner and ghost semantics

Owner 是某实体的权威 rank；ghost 是其他 rank 上的副本。Owner-to-ghost 通常使用 `replace`，Ghost-to-owner 通常使用 `sum`。Reduce-and-broadcast 是先归并到 owner，再由 owner 广播；Peer exchange 不要求 owner。

## Request lifecycle

`begin_exchange()` 返回请求。请求处于 pending 或 completed 状态；`test()` 返回 `(completed, result)`，未完成时结果为 `None`；`wait()` 返回最终目标数组。完成请求允许重复 `wait()`，但一次请求不得重新执行。非阻塞请求完成前，源、目标和 buffer 必须保持有效。阶段 A 的 pending 请求仅用于验证状态契约。

## Communication direction and tags

阶段 B 采用真实邻居的点对点 typed-buffer 通信。计划构建期交换索引，运行期不传输全局编号或本地索引。tag 必须由计划上下文隔离，避免同一 communicator 上的不同计划串包。

## Validation levels

`none`、`basic`、`full` 分别表示最低运行检查、本地与计数检查、跨进程一致性检查。跨进程验证失败必须尽量让所有 rank 一致结束当前操作，避免部分 rank 等待。
