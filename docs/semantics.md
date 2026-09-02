# Halo Semantics Contract

## `replace`

Owner 发出的值覆盖 ghost 的 `dst[recv_indices]`。阶段 B 必须拒绝无法确定优先级的多源写入。

## `sum`

所有接收贡献执行 `dst[index] += value`。目标原值保留；调用方若需要从零归并，必须先清零。实现不得依赖重复高级索引赋值的未定义行为。

## Directional modes

- Reduce-to-owner：ghost contributions → owner，使用 `sum`。
- Owner-to-ghost：owner value → ghost，使用 `replace`。
- Reduce-and-broadcast：先 Reduce-to-owner，再 Owner-to-ghost。
- Peer exchange：调用方提供有向边，不要求 owner。

这些语义在阶段 D 的单元及 MPI 测试中验证；reduce-and-broadcast 可用调用方显式提供的两个计划（`from_edges`），也可由 `from_global_ids(..., direction="ghost_to_owner")` 与 `from_global_ids(..., direction="owner_to_ghost")`（或默认 `two_way` 一次得到两个方向）构造，后者仍不推断 owner。
