# Halo API Contract

阶段 C 已实现直接边的阻塞与非阻塞 `replace`；`sum` 保留给阶段 D。

## `HaloPlan`

- `HaloPlan.from_global_ids(comm, global_ids, owners, *, validation="basic")`
- `HaloPlan.from_edges(comm, edges, *, validation="basic")`
- `plan.exchange(src, dst=None, *, op="replace")`
- `plan.begin_exchange(src, dst=None, *, op="replace")`
- `plan.close()`

`dst=None` 表示原地更新。数据形状为 `(local_entity_count, ...)`，支持标量与多分量 payload。阶段 B 的 `replace` 覆盖接收位置；`sum` 将在阶段 D 实现。计划元数据和内部索引不得暴露为可变别名。

## `HaloRequest`

- `wait(timeout=None)` 返回最终目标数组。
- `test()` 返回 `(bool, result)`。
- `completed` 属性报告终态。

完成请求允许重复 `wait()`。计划关闭、非法并发、未完成请求释放等行为使用专用异常表示。请求完成前不得修改源、目标或计划 buffer；请求完成后计划可重新发起交换。

## Errors

所有领域异常继承 `HaloError`：无效编号、owner、peer、索引、通信数量或 payload；多源 `replace` 冲突；关闭计划使用；非法请求状态；不支持数组或 dtype。
