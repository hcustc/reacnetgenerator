# 时序分子与反应事件输出：优化设计交接

更新日期：2026-08-01  
工作分支：`perf/optimize-timed-molecule-output`

## 1. 一句话结论

两个新增功能需要保留，但不应各自重复扫描和展开整条轨迹。ReacNetGenerator 现在复用原有分子识别、HMM 和反应检测结果，流式写出紧凑的 HDF5 中间文件；ReacNet-Scope 再按需将其转换或导入 SQL。HDF5 已是当前实现，但尚未通过原 OOM 规模证明为长期最优格式。真正决定能否解决 OOM 的是数据表示和计算流程，而不只是把 SQLite 换成 HDF5。

## 2. 两个新增功能的目的

### 分子时间信息

为每个分子实例保留：它是什么物种、由哪些原子和键组成、在哪些帧或原始 timestep 中存在。用途是让 ReacNet-Scope 能回到具体时刻和具体原子，而不只是看到总体反应网络。

原始接口和输出：

- `--show-molecule-time`：输出 `<input>.molecules.csv`。
- `--molecule-frame ...`：只输出指定的分析帧，并自动启用该功能。
- `--molecule-timestep ...`：只输出指定的原始 timestep，并自动启用该功能。

### 逐时刻反应事件

保留每一次相邻分析帧之间发生的反应，包括 transition index、反应物和产物。用途是支持按时间查看、筛选和回溯反应，而不是只得到总次数。

原始接口和输出：

- `--reaction-event`：输出 `<input>.reactionevent.csv`。

## 3. 与 ReacNetGenerator 原有功能的关系

这两个功能不是新的化学识别算法，而是把原有计算中已经出现、但随后被汇总或丢弃的时序信息保存下来。

- `.moname` 已保存分子实例的物种名、原子和键，但不保存完整存在时间。
- `.route` 已保存每个原子的物种变化路径，并用“发生变化的帧”压缩连续不变区间。
- `.reactionabcd` 保存 `A+B->C+D` 的总次数，但丢失每次反应发生的时间。
- `atomeach[N,T]` 和 `conflict[N,T]` 是 PATH 阶段已有的中间状态，反应汇总和逐时刻反应事件都基于它们。

因此，两个新增功能应共享以下计算：分子实例识别、SMILES/VF2 归一化、HMM 后的存在区间、相邻帧变化检测和反应物/产物归一化。不能为两种输出各跑一遍完整分析。

若中间文件保存了分子实例、原子集合、存在区间和冲突信息，逐时刻反应事件原则上可以由它推出；`.reactionabcd` 又可以由逐时刻事件聚合得到。反方向不成立：只有 `.reactionabcd` 无法恢复时间，只有 `.route` 也无法可靠恢复多分子反应的分组、键和冲突信息。

## 4. 为什么不直接扩写 `.reactionabcd` 和 `.route`

这两个文件是面向人和既有工具的文本摘要，语义稳定且已有下游依赖。直接塞入详细时序数据会带来四个问题：

1. 破坏已有格式兼容性。
2. 重复记录物种名、原子和键，文件会急剧膨胀。
3. `.route` 是按原子组织，反应事件是按帧和反应组织，无法自然表示完整反应分组。
4. ReacNet-Scope 的随机查询仍需重新解析和建立索引。

合理做法是继续生成原有文本摘要，同时增加一个机器可读、可压缩、可流式访问的时序中间文件。

## 5. ReacNetGenerator 与 ReacNet-Scope 的职责

ReacNetGenerator 的定位是批量读取轨迹、完成化学识别并输出可复现结果，不应承担完整的交互数据库服务。它负责：

- 生成唯一、紧凑、带版本和来源信息的中间文件。
- 保证峰值内存受块大小和并行度控制，而不是随轨迹总长度或事件总数线性增长。
- 保留足够信息，让下游无需重新读取原始大轨迹。

ReacNet-Scope 负责：

- 按自己的查询模型把中间数据导入 SQLite 或其他数据库。
- 创建面向界面的索引、缓存和派生表。
- 做筛选、联表、分页和交互查询。

这样可以避免让所有 ReacNetGenerator 用户为 SQL 查询能力支付存储和构建索引的成本，也允许 Scope 后续独立演进数据库结构。

## 6. 存储格式结论

### SQLite

优点是 Python 自带、事务和原子发布成熟、关系约束清楚、可以直接查询。缺点是 B-tree、行记录和索引有额外空间与写入成本，而且这些查询能力主要属于 ReacNet-Scope。SQLite 并不“太重”到不可用，但未必符合 ReacNetGenerator 的中间格式定位。

### HDF5

适合分块数值数组、压缩、局部读取和单文件交付，较符合科学计算中间文件。问题是字符串和变长嵌套结构不如关系表自然，`h5py/HDF5` 增加二进制依赖，写入并发和损坏恢复也需要额外设计。

### 其他候选

- Parquet/Arrow：列式压缩和表格扫描优秀，但增加较重依赖，多张逻辑表的单文件组织和增量写入不如 HDF5/SQLite 直接。
- Zarr：分块和云存储友好，但通常产生目录和大量小文件，不利于当前单文件交付。
- NPZ 或自定义压缩二进制：简单或最紧凑，但流式追加、局部读取、版本兼容和维护成本较差。

当前结论是：可以把 HDF5 作为替换 SQLite 的当前设计方向，但必须用相同逻辑数据和真实查询负载做基准后再定案。不要先写死存储后端；计算层应输出统一记录流，SQLite、HDF5 或其他实现只是可替换的 sink。

## 7. 建议的逻辑数据模型

无论最终使用哪种物理格式，都先固定以下逻辑结构：

- `metadata`：格式版本、ReacNetGenerator 版本、状态、参数和输入来源。
- `sources`：多个输入文件的顺序与路径。
- `frames`：连续分析帧、来源文件内帧号、原始 timestep。
- `molecules`：分子实例 ID、物种 ID、原子列表、键列表。
- `molecule_ranges`：`molecule_id, start_frame, end_frame`。
- `reaction_types`：反应 ID、反应物、产物及总次数。
- `reaction_events`：`transition_index, reaction_id, count`。

关键压缩规则：

- 连续 991 帧存在的同一分子写成一个 `[start,end]` 区间，而不是 991 行。
- 重复字符串改为整数 ID 和字典表。
- 同一 transition 中相同反应写一条带 `count` 的记录。
- 原子和键优先使用扁平数值数组加 offsets，避免大量 Python 字符串和 HDF5 变长对象。
- 不把完整 `atomeach[N,T] + conflict[N,T]` 原样作为最终文件格式。

## 8. OOM 事实与判断

大轨迹作业的已知信息：

- Step 1 完成，用时 66532.411 s。
- Step 2（HMM）完成，用时 375.110 s。
- Step 3 尚未完成便发生 OOM。
- `.molecules.csv` 已生成，`.reactionevent.csv` 为空。
- Slurm step `1064886.0` 状态为 `OUT_OF_MEMORY`，`MaxRSS=3000.29G`，`MaxVMSize=3008.41G`，节点约 3 TB 内存。
- 原始大轨迹已经删除，暂时无法原样复现。

这说明 OOM 发生在 PATH 阶段，而且很可能位于分子时间文件完成之后、反应事件完成之前。仅凭现有日志还不能把根因唯一归到某一行代码，但可以排除“只是输出文本文件太大”这一单一解释。

原有功能也会构造 `atomeach[N,T] + conflict[N,T]`，所以不开两个开关时能够完成并不矛盾：基础矩阵可能已经接近内存上限，新增的逐事件对象、排序/有序并行结果、队列预取、聚合列表或数据库 staging 又制造了额外峰值。96 个进程还会放大序列化、任务队列和工作数组副本。

当前工作区已把 `atomeach` 从固定 `int64` 改为可容纳分子 ID 的最小无符号整数，并把 `conflict` 改为 `bool`。两者的逻辑大小由约 `16*N*T` 字节降到约 `2~9*N*T` 字节，并存放在临时 `memmap` 中而不是 Python/NumPy 常驻堆内存。逻辑矩阵和临时磁盘仍随总帧数线性增长，因此需要验证磁盘容量和页面换入性能，但不再由每个 worker 各持有或通过任务队列复制完整行/列。

## 9. 已实施与仍需验证的内存优化

1. 已实施：分子存在信息写成区间；原子 route 只返回变化点，不展开逐帧 Python 行。
2. 已实施：反应事件在 worker 内按 transition 聚合为 `Counter`，主进程收到后立即写出。
3. 已实施：`atomeach/conflict` 使用临时磁盘映射，worker 初始化时只读映射，任务只传整数索引。
4. 已实施：Step 3 的 SMILES、route 和 reaction 任务使用 `chunksize=1`，并把 in-flight 数量限制为约 `2*nproc`。
5. 已实施：HDF5 writer 使用追加块、压缩、flush、文件锁和同目录原子发布。
6. 仍需验证：在 1、8、32、96 进程下测峰值 RSS、临时空间和吞吐；进程越多不一定越快。
7. 后续候选：若 `memmap` 的跨列访问或临时空间成为瓶颈，再改为带一帧重叠的 frame-block 流水线。

可直接学习的原有做法包括：LZ4 压缩临时块、生成器式读取、`WriteBuffer`、HMM 存在信号压缩、`.route` 只记录变化点、`.reactionabcd` 用 Counter 聚合，以及 `--stepinterval`/筛选参数减少分析规模。需要检查的是这些机制是否在调用处又被 `list(...)`、有序 multiprocessing 缓冲或大矩阵复制抵消。

## 10. 当前分支的真实状态

当前工作区已经把原 SQLite 原型替换为 HDF5，并完成三项 Step 3 内存修复：

- 两个旧 CSV 被合并为默认 `<input>.timeline.h5`。
- 分子 worker 不再返回展开的 NumPy frame 数组，而是透传压缩 frame block；主进程仅在写入时解压并合并为闭区间，HDF5 保存区间。
- 反应 worker 只接收 `transition_index`，返回 `(transition_index, Counter)`；允许乱序完成，HDF5 用按 transition 索引的 `block_start/block_length` 指向紧凑事件块。
- `atomeach` 和 `conflict` 改为临时 `memmap`；route worker 只接收 atom index，reaction worker 只接收 transition index，均使用 `chunksize=1` 和受限的 in-flight 数量。
- HDF5 使用同目录临时文件、文件锁、状态属性和原子发布；失败构建不会覆盖正式文件。

涉及的主要文件为 `reacnetgenerator/_timedoutput.py`、`_path.py`、`_reaction.py`、`_detect.py`、`reacnetgen.py`、`commandline.py`、`tools.py` 和 `tests/test_timedoutput.py`。这些修改消除了已定位的 IPC 放大和完整矩阵在每个进程中常驻的问题，但仍需用新的大轨迹验证峰值 RSS、临时磁盘和运行时间。

## 11. 测试与选型计划

`/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 可以用于功能回归、输出一致性和初步性能测试，但小轨迹通过不能证明大轨迹不会 OOM。建议测试矩阵如下：

- 四种模式：两个功能都关闭、仅分子时间、仅反应事件、两者都开启。
- 多种并行度：1、8、32、96。
- 记录每个 Step 3 子阶段的 wall time、峰值 RSS、临时空间和最终文件大小。
- 比较旧 CSV、HDF5 实现和原 SQLite 原型；必要时加入 Parquet 作为列式基线。
- 验证 `.moname`、`.route`、`.reactionabcd` 与原实现完全一致，并验证从详细事件聚合出的 `.reactionabcd` 一致。
- 对 ReacNet-Scope 的典型操作计时：按帧查分子、按原子查轨迹、按反应查发生时间、全量导入 SQL。

由于原始 OOM 数据已删除，还需要一份新的大轨迹，或由 `rp3.lammpstrj` 构造可控放大的压力数据，验证内存是否真正受块大小约束。

## 12. 完成标准与下一步

完成标准：

- 峰值内存主要由块大小和进程数决定，不再随总帧数或事件总数无界增长。
- 同时开启两个功能时复用同一遍计算，内存和耗时不应近似两项相加。
- 原有文本文件的语义和格式保持兼容。
- 中间文件可校验、可版本化、失败时不覆盖已有完整结果。
- ReacNet-Scope 能流式导入 SQL，而无需重新读取原始轨迹。

建议后续顺序：先完成小轨迹功能回归；再增加 Step 3 分阶段 RSS 和临时磁盘监测；随后用同一份放大轨迹比较旧实现、HDF5 实现和原 SQLite 原型；最后根据 ReacNet-Scope 的真实查询负载确认长期格式。
