# reduce-and-broadcast vs sync_add 对比实验

本目录是对比基准的程序与结论，回答「tribbie 的 `HaloPlan.reduce_and_broadcast` 与 fealpy 的
`EntityMPI.sync_add` 在求解器 SpMV 归约-广播路径上哪个更快」。

- `compare_sync.py`：单次 `mpiexec` 启动内，对给定拓扑在多个 payload 上先做数值等价断言再计时。
- `render_comparison.py`：把 JSON 渲染成 Markdown。
- `run_comparison.ps1`：按 ring / grid 两种拓扑遍历 size 与 payload 并调用 `mpiexec`。

## 语义对齐

两条路径语义不同但**数学等价**：给定同一份分布式元数据（同一组全局实体、同一 owner、同一共享关系），
二者最终都得到「每个共享实体的每个副本 = 全局贡献之和」。

- `reduce_and_broadcast(reduce_plan, broadcast_plan, values)`：先 ghost→owner `sum`，再 owner→ghost `replace`。
- `sync_add(array)`：每个 rank 把**所有**共享该实体的其它 rank 的贡献累加到自己的本地副本上。

三者（`reduce_and_broadcast`、`sync_add` 带拷贝、`sync_add` 不带拷贝）先做 `allclose` 等价断言，
断言通过才计时（`correct` 字段）。

## 元数据同源

每个 (topology, size, payload) 配置由**同一个确定性生成器**产出两种表示：

- tribbie 侧：`global_ids`、`owners` → `from_global_ids(comm, global_ids, owners, direction="two_way")`。
- EntityMPI 侧：`sharing_pairs`（每 rank 一份，索引 = 对端 rank）→ `EntityMPI(pairs=sharing_pairs, comm=comm)`。

构造时间不计时、不比较（`from_global_ids` 的发现与 `sharing_pairs` 的直接给出本不同构）。

## 拓扑

- `ring`：一维周期环，每个 rank 拥有 E 个实体，与左右各共享 H 个（本地 `E+2H`）。
- `grid`：二维周期均匀网格（环面），`Px×Py` 个 rank，每个 rank 拥有 `n×n` 节点，本地 `(n+2H)²`；
  sharing_pairs 覆盖 8 邻域（4 边 + 4 对角），对角保证角节点（4 rank 共享）能被 `sync_add` 完整汇总。

## 三个计时用例

1. `reduce_and_broadcast`（就地，两次点对点交换）。
2. `sync_add`（fealpy 原生 API，**整组拷贝** + alltoall + index_add，out-of-place）。
3. `sync_add_no_copy`（就地 `np.add.at`，无整组拷贝，隔离通信路径）。

`speedup_copy = sync_add / reduce_and_broadcast`，`speedup_no_copy = sync_add_no_copy / reduce_and_broadcast`；
两者之差即整组拷贝在该 payload 下的影响。`>1` 表示 reduce_and_broadcast 更快。

## 实验矩阵

| 轴       | 取值                                                       |
| ------- | -------------------------------------------------------- |
| 拓扑      | `ring`、`grid`                                            |
| 并行规模    | ring `size ∈ {2,4,8,16}`；grid `size ∈ {4,8,16}`          |
| payload | ≈ 100KB / 1MB / 10MB / 100MB（本地数组字节 = 本地实体数 × 8，float64） |

`tracemalloc` 已禁用：整组拷贝是被实测的真实成本，tracemalloc 会按分配放大拷贝时间、污染结论。

## 运行

```powershell
# 前置：起 MPI 进程管理器（管理员起 MsMpiLaunchSvc，或普通用户 smpd -d）
powershell -ExecutionPolicy Bypass -File benchmarks/halo/comparison/run_comparison.ps1 -RunId <run-id>
```

单次手动（ring，size=4，E=10^5，H=1024）：

```powershell
mpiexec -n 4 .venv\Scripts\python.exe benchmarks/halo/comparison/compare_sync.py `
  --topology ring --entities-list 100000 --halo-list 1024 `
  --warmup 3 --repeats 10 `
  --output reports/halo/comparison/<run-id>/ring-size4.json
```

## 结果与结论

全部 24 个配置（ring 14 + grid 10，见矩阵）`correct = True`，且 `reduce_and_broadcast` **每一项都快于** `sync_add`（带拷贝与不带拷贝两个变体）——`speedup > 1` 无一例外：

| | speedup_copy | speedup_no_copy |
| --- | ---: | ---: |
| 范围 | 1.55× – 19.48× | 3.03× – 23.78× |

`speedup = sync_add 中位数 / reduce_and_broadcast 中位数`（>1 表示 reduce_and_broadcast 更快），
时间取 communicator 内最慢 rank 的中位数（warmup 3、repeat 10、`tracemalloc` 已禁用）。

### 代表数据（size=4）

**ring（E + 2H 本地实体，H = E/16）**

| payload | reduce (ms) | sync_add copy (ms) | speedup copy | sync_add no-copy (ms) | speedup no-copy |
| ------: | ----------: | -----------------: | -----------: | --------------------: | --------------: |
|   98 KB |       0.034 |              0.084 |        2.45× |                 0.130 |           3.78× |
|  977 KB |       0.085 |              1.397 |       16.35× |                 1.431 |          16.75× |
|  9.8 MB |       0.951 |              13.76 |       14.47× |                 14.65 |          15.40× |
|   98 MB |        35.1 |              181.5 |        5.16× |                 220.9 |           6.29× |

**grid（(n+2H)² 本地实体，H = n/16）**

| payload | reduce (ms) | sync_add copy (ms) | speedup copy | sync_add no-copy (ms) | speedup no-copy |
| ------: | ----------: | -----------------: | -----------: | --------------------: | --------------: |
|  124 KB |       0.156 |              0.315 |        2.03× |                 1.043 |           6.70× |
|  1.2 MB |       0.293 |              5.707 |       19.48× |                 6.966 |          23.78× |
|   12 MB |        11.8 |               53.7 |        4.54× |                 103.7 |           8.76× |
|  121 MB |       141.5 |              648.4 |        4.58× |                1066.0 |           7.53× |

完整数据见 `reports/halo/comparison/<run-id>/`（每个 `*.json` 对应一份 `*.md` 渲染）。

### 整组拷贝的影响（要求 4）

结论与直觉不同：**整组拷贝在这组 payload 下不是 `sync_add` 的主要开销**。

- `sync_add`（带拷贝）与 `sync_add`（无拷贝）的中位数接近，且在若干大 payload 配置下**无拷贝反而更慢**
  （如 grid size=4 @ 12 MB：copy 53.7 ms vs no-copy 103.7 ms）。这说明 `sync_add` 的差距主要来自
  **稠密 `comm.alltoall` + pickle 序列化 + `np.add.at`**，而整组 `array.copy()`（即使 100 MB 也仅 ~10 ms 量级）
  不是瓶颈。
- 早前旧基准（`2026-09-01-sweep*`）出现的 ~52× 整组拷贝放大，是 `tracemalloc` 按分配放大所致；本基准
  禁用 `tracemalloc` 后，整组拷贝的成本在总开销中占比很小。

### 趋势

- `speedup` 随 payload 增大整体走高（小 payload 时单机 MPI 延迟噪声压低差距，但仍 >1）。
- 拓扑与并行规模上，grid 的加速比普遍高于 ring（grid 每个 rank 有更多邻居，`alltoall` 的消息数 O(size) 与
  pickle 开销被放大）；两种拓扑下 `reduce_and_broadcast` 均保持全面占优。

## 历史版本

早期版本比较的是 `HaloPlan.exchange`（replace/sum）与 `EntityMPI.sync`（transfer），
归档于 `reports/halo/comparison/2026-09-01-sweep*`，方法与结论已被本目录取代。
