# OOM 分析实验报告 Evidence Map

| ID | 来源 | 等级 | 支持内容 | 不能支持 | 用途 | 风险 |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | 用户材料：`pasted-text.txt` | L1 | 32 MiB 初版推导、三组测试数值、作业输出、阶段性结论 | 未提供的节点型号、精确命令、`pct_01` 原始 `sacct` 行 | 摘要、实验、结果、结论 | 三组测试记录完整度不同 |
| E2 | `docs/timed-output-optimization-handoff.md` | L1 | 历史 OOM 状态、故障边界、优化设计、未完成验证项 | OOM 的唯一单行根因 | 背景、故障分析、局限性 | 历史原始轨迹已删除 |
| E3 | `reacnetgenerator/_path.py` at `b332ce99` | L1 | memmap、紧凑 dtype、route 索引任务、`chunksize=1`、受限在途任务 | 生产规模性能 | 方法 | 无 |
| E4 | `reacnetgenerator/_reaction.py` at `b332ce99` | L1 | transition 索引任务、局部 Counter、乱序完成 | 生产规模性能 | 方法 | 无 |
| E5 | `reacnetgenerator/_timedoutput.py` at `b332ce99` | L1 | HDF5 逻辑结构、增量写入、状态和发布语义 | 长期格式选型优越性 | 方法、完整性分析 | 无格式基准对照 |
| E6 | `reacnetgenerator/utils.py` at `b332ce99` | L1 | `chunksize` 和 `max_inflight` 约束 | 特定平台的队列内存精确值 | 方法 | 无 |
| E7 | `git diff 02a12117..b332ce99` | L1 | 旧数组型任务与新索引型任务之间的实现差异 | 历史 3 TB 峰值的精确组成 | 根因机制分析 | 机制解释而非逐分配剖析 |
| E8 | 由 E1 数值直接计算的倍率 | L1 派生 | 帧数、时间、内存、文件、区间和事件倍率 | 测试范围外的外推 | 结果、讨论 | 限于三个规模点 |

## 结论边界

- 可写为实测：三组作业完成；退出码、HDF5 状态、时间、MaxRSS、文件大小及数据行数。
- 可写为实现事实：memmap、索引任务、`chunksize=1`、受限 `max_inflight`、区间和 Counter 写出。
- 只能写为机制判断：旧实现的矩阵、IPC、预取和 Python 对象共同构成内存放大路径。
- 不可写为已证实：历史 OOM 的唯一单行根因；相对于 3 TB 的精确优化倍数；HMM 或生产全量轨迹已通过。
