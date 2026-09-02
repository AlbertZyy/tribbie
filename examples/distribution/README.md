# Distribution Examples

本目录演示 `tribbie.distribution` 的持久分布式元数据与 `halo.HaloPlan` 执行计划的分离，以及二者的桥接。

## roundtrip.py

用同一个 `IndexDistribution` 生成双向 halo 计划，完成“贡献归约到 owner 再广播回所有副本”（`reduce_and_broadcast`）。

```bash
mpiexec -n 3 uv run python examples/distribution/roundtrip.py
```

示例要点：

- `IndexDistribution` 只描述“谁拥有哪个全局实体”，不持有 communicator；
- `make_halo_plan(dist, comm, direction="two_way")` 返回 `(ghost_to_owner, owner_to_ghost)`，顺序与 `HaloPlan.reduce_and_broadcast` 期望一致；
- 同一分布可反复用于生成不同方向、不同验证等级的 plan。
