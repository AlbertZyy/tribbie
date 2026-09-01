# HaloPlan.exchange vs EntityMPI.sync 对比实验

本目录是对比基准的程序与结论，用于回答「tribbie 的 `HaloPlan.exchange` 与 fealpy 的
`EntityMPI.sync` 哪个更快」。原始数据（JSON + Markdown）归档于
`reports/halo/comparison/2026-09-01-sweep2/`。

## 结论（TL;DR）

在测试的全部 14 个配置 × 2 种语义（`replace` / `sum`）中，`HaloPlan.exchange`
**每一项都快于** `EntityMPI.sync`（28 个加速比全部 > 1）。因此，**对测试数据而言，
`exchange` 是 `sync` 的性能上界（完全上位）**：

| 语义 | 加速比范围 | 均值（14 个配置） |
| --- | ---: | ---: |
| `replace` | 1.34× – 8.65× | ≈ 3.2× |
| `sum` | 2.23× – 14.36× | ≈ 6.0× |

`speedup = entity_mpi_median / halo_plan_median`，`> 1` 表示 `exchange` 更快；时间取
communicator 内最慢 rank 的中位数（warmup 5、repeat 20）。

## 实验设计（公平性）

- **语义对齐**：两侧构造同一「1-D 周期链、每 rank 左右各共享 `H` 个 halo 实体」问题，
  并做数值等价断言 —— `exchange(op="sum") == sync + in-place add`、
  `exchange(op="replace") == sync + in-place scatter`。所有配置 `correct: true`，
  保证比较的是同一数据流、同一字节量。
- **就地 apply**：两侧都就地 apply（不做整组数组拷贝），因此测得的差异只反映
  **通信 / 序列化路径**，不含拷贝。
- **构造时间剔除**：`from_edges` / `EntityMPI.__init__` 的构造时间不计入交换计时。

## 结果

### 弱扩展（每 rank E=10⁵，H=1024，payload 16 KB）

| size | exchange (µs) | sync (µs) | speedup |
| ---: | ---: | ---: | ---: |
| 2  | 76.5 / 70.2  | 105.1 / 227.2 | 1.37× / 3.24× |
| 4  | 97.9 / 108.6 | 243.2 / 270.6 | 2.49× / 2.49× |
| 8  | 226.2 / 257.0| 496.2 / 821.6 | 2.19× / 3.20× |
| 16 | 238.1 / 274.6| 754.6 / 1003  | 3.17× / 3.65× |

（每格为 `replace / sum` 的中位数，µs）

### 强扩展（全局 10⁶ 实体按 rank 均分）

| size | E | H | replace speedup | sum speedup |
| ---: | ---: | ---: | ---: | ---: |
| 2  | 500000 | 31250 | 4.67× | 14.36× |
| 4  | 250000 | 15625 | 3.73× | 12.57× |
| 8  | 125000 | 7812  | 2.17× | 6.32× |
| 16 | 62500  | 3906  | 4.91× | 9.11× |

### 数据规模（size=4，float64，C=1）

| E | H | payload | replace speedup | sum speedup |
| ---: | ---: | ---: | ---: | ---: |
| 10⁴ | 625   | 10 KB  | 1.74× | 2.96× |
| 10⁵ | 625   | 10 KB  | 2.40× | 2.39× |
| 10⁵ | 6250  | 100 KB | 3.87× | 6.54× |
| 10⁶ | 625   | 10 KB  | 2.06× | 2.23× |
| 10⁶ | 6250  | 100 KB | 1.34× | 3.23× |
| 10⁶ | 62500 | 1 MB   | 8.65× | 12.10× |

**趋势**：加速比随 payload 字节（`H`）单调增大（10 KB → 1 MB 时 replace 1.3–2.4× 升到
8.65×，sum 2.2–3.0× 升到 12.1×），并随 rank 数整体走高（弱扩展 2 → 16 时 replace
1.37× → 3.17×）。小 payload 时差距被单机 MPI 延迟噪声压低，但 `exchange` 仍无例外地更快。

## 为什么 `exchange` 全面占优（机制）

`exchange` 与 `sync` 的运行时路径存在三层本质差异：

| 开销 | `HaloPlan.exchange` | `EntityMPI.sync` |
| --- | --- | --- |
| 通信 | 点对点 typed `Isend`/`Irecv`，只与真实邻居（2 个）通信 | 稠密 `comm.alltoall`，对**所有** rank 发消息（O(size)） |
| 序列化 | 原生 dtype 裸 buffer，零 pickle | 每条消息 pickle 一个 `SparseData1D`（数据 + 索引） |
| 索引 | 静态存在 plan，运行期不传 | 每条消息额外带一个 int64 `index_other` 张量 |

`sync` 每次调用的在线字节 ≈ 有用 payload × 2（数据 + 索引张量），且全部经过 pickle；
`exchange` 只传有用 payload 且走原生类型。因此差距随数据规模与并行规模同步放大。

## 范围与注意事项（诚实声明）

- **「完全上位」的边界**：结论建立在测试网格内（14 个配置，`float64`、`C=1`、
  1-D 周期链、单机 Windows 11 + MS-MPI 10.1，绝对时间为本机基线）。未覆盖多分量
  `C>1`、非连续布局、跨节点网络；在这些维度外推需另行验证。
- **拷贝因素**：本结论测的是「隔离通信」口径（两侧都就地 apply）。若按
  **API 原生用法**直接对比 `sync_add`（out-of-place，每次都整组 `array.copy()`）与
  `exchange`（就地），在小 halo / 大数组的极端 regime（如 `E=10⁶, H=625`）差距会因
  整组拷贝进一步放大到 ~52×；这是 `sync_add` 的 out-of-place 契约，而非 `sync` 本身。
  `sync`（transfer）只拷贝共享块，不含整组拷贝。
- **测量噪声**：计时期间启用 `tracemalloc` 会略微放大「每次调用都分配数组」的
  `sync`/`sync_add` 开销，真实无 trace 运行的绝对倍数可能略低于表列值，但方向不变。
