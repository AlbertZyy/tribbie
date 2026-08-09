# Benchmark Protocol

阶段 A 冻结测量协议，不报告未经执行的性能数字。

## Metrics

记录计划构建、每轮打包、MPI 等待、解包/归并、总交换时间、发送/接收字节数、临时内存峰值、分配次数及非阻塞隐藏比例。并行总耗时取最慢 rank。

有效 payload 为 `B_p = H_p K s`，有效带宽为 `W_p = B_p / T_p`。隐藏比例按任务书中的通信、计算与重叠时间公式计算。

## Matrix

改变邻居数、每邻居实体数、payload 分量、dtype、本地索引连续性、`replace`/`sum` 及阻塞/非阻塞。每个配置执行预热和多次重复，报告中位数、低/高分位和跨 rank 最大值。

## Scaling and baselines

执行弱扩展、强扩展和可调计算负载的非阻塞重叠测试。基线至少包括逐邻居 typed-buffer 阻塞实现与 `Isend`/`Irecv` + `Waitall`；不使用 Python 对象通信作为性能基线。
