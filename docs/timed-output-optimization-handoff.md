# 时序分子与反应事件输出：优化设计交接

更新日期：2026-08-08\
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
1. 重复记录物种名、原子和键，文件会急剧膨胀。
1. `.route` 是按原子组织，反应事件是按帧和反应组织，无法自然表示完整反应分组。
1. ReacNet-Scope 的随机查询仍需重新解析和建立索引。

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

### 当前分支的因果边界

将分支点 `02a12117` 与当前已提交基线 `b332ce99` 逐行比较后，结论是“当前分支放大并
暴露了既有主进程瓶颈”，而不是“串行 fan-in 完全由当前分支新建”。在 `02a12117` 中，
SMILES worker 已返回 `(name, atoms, bonds, frames)`，单个父进程依次执行 fallback/miso、
`.moname` 和 timeline spool；因此 worker 先完成、父进程消费跟不上这一结构此前就存在。
`b332ce99` 新增 HDF5 后，又在同一个父循环中逐 molecule 调用 `add_molecule()`；最初版本
会对 molecule ID、species ID、atoms、offset、bonds 和 ranges 等 dataset 反复
`resize + write`。这足以加重已有串行/I/O 临界路径，与作业现象方向一致。

但原始全轨迹已删除，也没有保存作业 `1089747` 的精确 commit、分阶段 profile、affinity
和 HDF5 writer 属性，因此不能把 48 小时和约 2.1 核唯一归因于 `b332ce99`。CPU binding、
cgroup、实际 worker 选择或正停留在其他串行子阶段仍需生产日志排除。当前工作区同时修复
了两类问题：批量/有界 HDF5 writer 消除分支新增的细粒度写放大，SMILES/route/reaction
调度和解码复用则缩短更早就存在的父进程 fan-in。

当前工作区已把 `atomeach` 从固定 `int64` 改为可容纳分子 ID 的最小无符号整数，并把 `conflict` 从 1 byte/cell 的 `bool` 改为 1 bit/cell 的行优先位矩阵。两者的临时文件大小由约 `16*N*T` 字节降到约 `(molecule_id_bytes + 1/8)*N*T` 字节，并存放在临时 `memmap` 中而不是 Python/NumPy 常驻堆内存。逻辑矩阵和临时磁盘仍随总帧数线性增长，因此需要验证磁盘容量和页面换入性能，但不再由每个 worker 各持有或通过任务队列复制完整行/列。

## 9. 已实施与仍需验证的内存优化

1. 已实施：分子存在信息写成区间；原子 route 只返回变化点，不展开逐帧 Python 行。
1. 已实施：反应事件在 worker 内按 transition 聚合为 `Counter`，主进程收到后立即写出。
1. 已实施：`atomeach/conflict` 使用临时磁盘映射，worker 初始化时只读映射，任务只传整数索引。
1. 已实施：`conflict` 以 1 bit/cell 保存，matrix 使用已有的有界 overlap mask 批量 pack/OR，reaction 按列和 atom block 懒解码。5,120 万逻辑 cell 压力下文件固定降为 1/8，整体 MaxRSS 下降约 17%--21%，全列扫描缩短约 53%--57%；逐 atom 写入的失败原型已由结构回归排除。
1. 已实施：matrix 构建对全真 signal block 使用连续 frame slice，不再创建最多约 8 MiB/block 的 `intp` frame index；64 项前缀先排除 empty/sparse block。500 万帧 dense A/B 的 wall/RSS 成对中位约降低 62.6%/54.6%，稀疏和空块无系统性 wall 回退。
1. 已实施：no-HMM matrix 自适应复用 molecule 临时记录中已有的 frame index；只有其未压缩 payload 不大于完整 bool signal 时才直写，否则保留 signal/dense-slice 路径。500 万帧稀疏 A/B 的 wall/RSS 成对中位约降低 71.4%/84.4%，密集回退无系统性 wall 回退；no-HMM 只读取 atom/frame 字段，HMM 只读取 atom 字段。
1. 已实施：matrix 的 empty signal block 在继续扫描前释放 block view，防止最后一块为空时把上一条完整 signal 保留到下一条解码；弱引用回归锁定相邻 signal 不重叠。
1. 已实施：no-HMM filter 与 matrix 共用 frame 内存阈值；direct 记录不再展开、压缩或写入 origin signal，全 direct 时不创建 Pool，遇到 dense 后才惰性切回并行。no-HMM 还复用 Step 1 molecule 文件，不再写相同副本。5,615 条形态的强制 Pool/父进程 A/B wall 成对中位约降低 98.6%；真实样例 Step 2 为 0.063 秒、origin 0 字节、避免 3.139 MiB 副本。
1. 已实施：HMM 与 no-HMM dense fallback 按 frame 数自适应 Pool 批次；百万帧时 64 核由默认 `chunksize/max_inflight=100/9600` 收紧为 `1/128`，短轨迹仍保持默认批次。no-HMM fallback 强制保序，使 origin signal 与复用的 Step 1 molecule 文件对齐；真实双 worker 逆序完成回归锁定该契约。最终有界保序路径在 40 条 × 100 万帧反馈环中的 wall/RSS 成对中位约降低 21.7%/73.2%，真实 HMM 40 条 × 10 万帧 wall 成对中位约降低 15.5%。
1. 已实施：Step 3 的 SMILES、route 和 reaction 均使用有界 in-flight；route 保持每 atom 一项，SMILES 只对廉价 fork 记录做有限分批，reaction 则按 transition 数、原子扫描量和修改事件密度在 `1--32` 之间自适应 chunksize。
1. 已实施：`nproc=1` 且不需要 worker initializer 时直接同步执行，不再创建 Pool/pipe；多进程 SMILES 使用只含 `atomname + atomtype` 的最小 initializer。默认 task 只返回 species name；仅 cheap batched `fork` 且所有压缩结构不超过 64 KiB 时返回 worker 已解码的 atoms/bonds。
1. 已实施：Detect 的第二次 molecule-frame compression 不再把 array 发送到 process Pool。总 payload 小于 16 MiB 或平均小于 128 KiB/molecule 时在父进程串行；足以摊薄固定成本时使用 2 个共享内存线程、最多 4 个在途 future。128 MiB 大块反馈环 wall 中位降低约 40.9%，额外 unique-memory peak 比旧 8-process `fork` 降低约 96.2%；短轨迹不承担线程开销。
1. 已实施：已知任务总数小于申请核数时，实际 worker 自动限制到任务数，避免少量 frame/transition 仍启动几十个空闲进程。
1. 已实施：SMILES worker 在一个 molecule 阶段内不再每 1,000 项回收，避免反复加载 RDKit/OpenBabel；112,300 项压力下单 worker RSS 未随任务数增长。
1. 已实施：HMM 在现有复制循环中累计压缩结构工作量；SMILES worker 数同时按平均结构复杂度和足以摊薄进程启动的总字节自适应。廉价结构可直接走父进程，重结构仍保留多进程；缺失指标的旧中间调用保持原调度。
1. 已实施：自适应串行 SMILES 对每条压缩结构只解码一次，atoms/bonds 直接复用于 fallback、miso、`.moname` 和 HDF5；真实 5,615 条结构的转换加下游准备中位约缩短 9.9%，纯解码/重建约缩短 58.9%。多进程默认保持 name-only；后续新增的受限 cheap-fork 例外见第 15 节。
1. 已实施：必须稳定排序的 SMILES/route worker 乱序计算、磁盘暂存被慢序号挡住的结果，再严格恢复输入顺序；后台 collector 原型经真实 A/B 后删除，因为它提高 worker 活跃度却增加 Step 3 wall time。
1. 已实施：`mname` 改为最小无符号 species ID + 唯一名称表，route/reaction worker 分别 mmap 两张紧凑表，避免一个长 SMILES 把每个 molecule 的定宽 Unicode 槽放大。
1. 已实施：route 主进程在接受 molecule pair 后立即聚合为 species-pair `Counter`，不再为 Matrix 保留第二份逐 event 数组；跨 atom 去重状态使用紧凑整数 key，保持 no-HMM/HMM 原有计数语义。
1. 已实施：长 atom route 在 worker 内按 65,536 行有界分块预聚合重复 molecule pair，再返回唯一 pair 和次数；全唯一路径抽样后直接返回原数组，唯一 pair 过多时也回退，避免用额外内存换取无收益的排序。
1. 已实施：route 日志记录 worker compaction 前后的 event/pair 行数和命中 atom 数，供全轨迹判断父进程串行聚合是否实际缩小。
1. 已实施：`.route` worker 返回 UTF-8 bytes，父进程使用目标 8 MiB 的字节有界二进制 `WriteBuffer`；单个超大 route 行独立立即写出，生产日志记录实际最大 batch。
1. 已实施：route worker 数按 `atom_count × frame_count` 和 atom task 数共同自适应；每个 worker 目标至少承担 65,536 个扫描值，并以每 worker 约 64 个 atom task 为上限约束，结果仅需 1 个 worker 时直接在父进程运行，避免短时间轴或窄轨迹仍创建过多 Pool worker、pipe 和 ordered spool。
1. 已实施：route 顺带累计全部原子的 molecule 变化事件，reaction worker 数按启动方式自适应：Linux `fork` 每 5 万个变化事件、`spawn`/`forkserver` 每 100 万个变化事件增加一个 worker，并保留每 5 亿个扫描值的兜底；短任务在父进程直跑，密集长任务继续并行，且 `--selectatoms` 不会漏计 reaction 工作量。10,001 帧真实甲烷输入上，`fork` 的 reaction 由 1 worker 自适应为 7 workers，完整 Step 3 三轮中位 wall 约缩短 24.8%。
1. 已实施：reaction 变化 pair 在进入 Python 图前按抽样自适应压缩；1024 行有界块先合并连续 pair，再合并高度重复的交错 pair，全唯一路径回退。百万相同/100 种交错 pair 的隔离 wall 分别缩短约 98.3%/82.5%。
1. 已实施：reaction 连续 pair 检测的两个布尔 workspace 在每个 transition 内只创建一次；1,000 万重复 pair 的独立进程中位 wall 约再缩短 12.6%、RSS 约减少 0.70 MiB。数值 gather workspace、512 行块和 encoded pair-key workspace 均经 A/B 证明回退并已删除，负结果保留在本地修改日志。
1. 已实施：默认 `.species` timeline 不再常驻“每帧一个 Counter”；根据实际占用自适应选择稠密 count/priority memmap 或稀疏 frame/species event spool。稀疏单帧使用 NumPy 聚合，spool 以稳定分组和向量化 scatter 回填，并保持旧物种计数与逐帧插入顺序。
1. 已实施：HDF5 writer 使用严格行数/字节上限的列式批次、压缩、flush、文件锁和同目录原子发布。
1. 已实施：worker 异常或调用方提前退出时，可取消 producer 会解除 semaphore 等待并清理进程、线程和临时暂存。
1. 仍需验证：在 1、8、32、96 进程下测峰值 RSS、临时空间和吞吐；进程越多不一定越快。
1. 后续候选：若 `memmap` 的跨列访问或临时空间成为瓶颈，再改为带一帧重叠的 frame-block 流水线。

可直接学习的原有做法包括：LZ4 压缩临时块、生成器式读取、`WriteBuffer`、HMM 存在信号压缩、`.route` 只记录变化点、`.reactionabcd` 用 Counter 聚合，以及 `--stepinterval`/筛选参数减少分析规模。需要检查的是这些机制是否在调用处又被 `list(...)`、有序 multiprocessing 缓冲或大矩阵复制抵消。

## 10. 当前分支的真实状态

当前工作区已经把原 SQLite 原型替换为 HDF5，并完成多项 Step 3 内存与调度修复：

- 两个旧 CSV 被合并为默认 `<input>.timeline.h5`。
- Detect 不再用默认 `chunksize=100` 把多个昂贵 frame 捆给单个 worker；现在按单帧、
  有限在途窗口、输入顺序消费。Linux-like `fork` 保留请求核数，`spawn`/`forkserver`
  按 `ceil(total_input_bytes / 8 MiB)` 摊销 worker。第二次 molecule-frame compression
  不再使用 process IPC：小 payload 留在父进程串行，总量至少 16 MiB 且平均至少
  128 KiB/molecule 时才启用 2 个有界 parent threads、最多 4 个在途结果。128 MiB
  大块反馈环相对父进程串行 wall 中位降低 40.9%，额外 unique-memory peak 比旧
  8-process `fork` 低 96.2%；真实 5 帧配对 `fork` A/B 的 Detect wall 中位由
  15.541 秒降至 4.952 秒，
  平均核数由 0.990 升至 4.544，molecule 临时文件哈希不变；生产仍需校准 wall 与
  core-hours 的权衡。
- Detect 的 timestep 不再用连续 step 作为 Python dict key，而是保存为 signed 64-bit
  array；frame provenance 按 source 内的等差 frame 段压缩，同时保留 Mapping 访问契约。
  100,000-frame 调用点反馈环的 retained allocation 由约 21.44 MiB 降至 0.78 MiB、
  `tracemalloc` peak 由约 24.13 MiB 降至 3.47 MiB。timed-output writer 会直接按 segment
  填充 provenance 块，因此 500,000 帧的构建加扫描中位还由约 0.106 秒降至 0.073 秒；
  Step 1 日志报告 compact metadata MiB 和 segment 数。
- Detect 的每个 molecule-frame occurrence 不再固定用 8-byte `array("Q")`；新 molecule
  从 native unsigned-short 开始，只在真实 frame index 溢出时扩宽到下一个更宽的无符号
  array，并以实际 dtype 压缩。100 万 occurrence 的生产调用点反馈环中，peak 由约
  8.40 B/occurrence 降至 2.20 B/occurrence（总量约 8.01 MiB 降至 2.10 MiB，-73.8%），
  wall 中位仅约 +1.6%；真实 5 帧样例保存
  22,103 个 occurrence 只用 0.042 MiB payload。Step 1 日志会报告 occurrence 数、payload
  MiB 及各 typecode 的 molecule 数；全轨迹必须结合该分布和 MaxRSS 校准实际收益。
- SMILES worker 只接收结构块，默认返回 species name；atoms、bonds 和可选 frame block 由主进程从原临时文件选择性顺序读取。唯一例外是 cheap batched `fork` 且 HMM/filter 记录的最大压缩结构不超过 64 KiB，此时 worker 返回已解码 atoms/bonds，父进程只补读 frame block。每个多进程 worker 只初始化一次最小转换器；单进程不创建 Pool。
- 主进程把 frame 序列按 NumPy block 合并为闭区间；HDF5 以有界列式 batch 保存 molecule definition、atoms、bonds、offset 和 ranges。
- molecule range 使用独立的 65,536 行应用批次，同时保留 64 MiB 字节上限和 4,096 行 HDF5 chunk；100 万 range 的生产形态反馈环把 flush 从 245 次降至 16 次、write wall 中位缩短约 61%，合法 `uint64` range 也不再经历 `int64` 往返复制。
- 反应 worker 只接收 `transition_index`，返回 `(transition_index, Counter)`；允许乱序完成，HDF5 用按 transition 索引的 `block_start/block_length` 指向紧凑事件块。
- `atomeach` 使用最小无符号整数临时 `memmap`，`conflict` 使用 1 bit/cell 的 packed 临时 `memmap`；route worker 只接收 atom index，reaction worker 只接收 transition index。reaction 的 conflict 列在 65,536-row block 内懒解码，非 compact 分支只收集 changed atom；route 使用 `chunksize=1`，reaction 使用 `1--32` 的自适应 chunksize，并把在途输入限制为每 worker 两个 chunk。
- matrix 的 dense signal block 直接按连续 frame slice 写 memmap，sparse block 继续使用有界索引；empty block 会在继续前释放 view，避免相邻完整 signal 重叠。no-HMM 还会在 index payload 不大于完整 signal 时复用已有 frame index，并只读取 atom/frame 字段，HMM 只读取 atom 字段。最终真实 5 帧样例的 5,615 个 molecule、22,103 个 frame value 全部命中直写；日志同时报告 direct 数与 dense/sparse/empty fallback 数，生产长轨迹需据此验证命中率、matrix wall、page fault 与临时盘吞吐。
- no-HMM HMM-filter 不再为 matrix 已能直写的记录重建完整 origin signal；origin 文件只含 dense fallback，且全 direct 时在父进程完成。由于 no-HMM 不会过滤 molecule，Step 3 直接复用 Step 1 临时记录，避免第二份磁盘副本。真实 5 帧样例 Step 2 从本轮仍强制 Pool 的 5.098 秒降至 0.063 秒；生产日志报告 direct 比例、fallback origin、避免复制字节和并行 fallback。
- HMM 与 no-HMM dense fallback 的 Pool 粒度按轨迹帧数收紧：每个 chunk 目标约 100 万个 signal value，在途窗口目标约每 worker 两个 chunk，并优先受旧上限约束。百万帧、64 核从默认 `100/9600` 降到 `1/128`；生产日志直接报告实际 `chunksize/max_inflight`。no-HMM fallback 还显式保序，避免乱序 signal 与原 molecule 文件错配；最终 dense 反馈环 wall/RSS 成对中位约缩短 21.7%/73.2%，真实 Viterbi wall 约缩短 15.5%，短轨迹无 wall 回退。
- reaction worker 的变化原子检测按最多 65,536 行扫描，不再为每个 transition 同时创建完整原子长度的布尔比较数组和 `intp` 索引；100 万原子全变化时单 worker 的函数增量 MaxRSS 中位约减少 5.2 MiB，最终反应计数与旧全向量路径一致。
- reaction worker 在 DFS 前去除重复分子邻接边；低出度使用 list，高出度升级为保持首次出现顺序的 dict。100 万个相同原子映射的邻接 RSS 增量中位由 64.3 MiB 降至 3.1 MiB，wall 由 0.413 秒降至 0.359 秒；20 万条全唯一边的内存基本不变、wall 约增加 3.8%。
- reaction 并行度复用 route 已观测的变化事件数，不增加矩阵扫描；真实 5 帧样例从 8 核请求自适应为父进程直跑，reaction wall 由 0.646 秒降至 0.030 秒，并避免约 100.3 MiB 的单个 child MaxRSS。调度现在区分进程启动成本：10,001 帧、450 原子的真实输入在 `spawn` 下仍串行，在 Linux-like `fork` 下为 7 workers。廉价 transition 会自适应分批，同时保证每 worker 至少约 4 个 chunk、每 chunk 至多约 100 万 atom-scan values 和 4,096 个平均修改事件；该真实输入选择 `chunksize=32/max_inflight=448`，reaction 中位由 0.469 秒降至 0.277 秒，聚合 RSS 未出现可测增长。no-event `.reactionabcd` 与 timed-output summary 统一按次数及名称确定性排序，乱序 worker 不再改变同频项的字节顺序。
- reaction 重复 pair 现在先在 1024 行有界 NumPy 块内压缩，再进入 Python 邻接图；100 万相同 pair 的 wall 中位由 0.3282 秒降至 0.00546 秒，100 种交错 pair 由 0.3180 秒降至 0.0557 秒。全唯一输入抽样后回退，1,000 万重复事件的额外临时 RSS 仍为常数级。
- reaction 的连续 pair run 检测复用 transition 级布尔 workspace，不再随 1,024 行块数反复分配；100,000 pair 回归把显式布尔分配从 198 次锁定到 2 次。三个看似节省内存但实际拖慢或增大 RSS 的 workspace/块大小候选已回退，并记录在 `docs/step3-performance-optimization-log.md`。
- SMILES worker 在单阶段内保持存活；本地 5,615 项直接 A/B 将 molecule-only wall 从 21.318 秒降到 4.093 秒，112,300 项时 child RSS 仍约 262 MiB。
- 最新真实 5,615 项反馈环中，单进程零 IPC 路径约 0.61 秒；8 进程最小 initializer 相对绑定 collector 基线中位约缩短 23%，压缩 worker 返回 payload 减少 62.4%。该便宜小样本在 2–16 进程反而更慢，生产核数必须按 wall 与 core-hours 实测选择。
- SMILES 调度现在使用 HMM 顺带累计的压缩结构字节并区分进程启动方式：`spawn`/`forkserver` 对廉价真实记录继续选择父进程，Linux-like `fork` 则在总结构工作超过 24 MiB 后启用最多 4 workers，并把平均不超过 512 bytes 的记录按 64 条分批。112,300 条真实结构记录的完整 molecule-name pipeline 三轮中位由 6.193 秒降至 4.609 秒，平均用核由 0.978 升至 2.479；复杂记录仍逐条调度。真实 5 帧小样本保持串行，避免 Pool 固定成本，生产日志会报告 start method、实际 worker、chunksize 和在途上限。
- 压缩 molecule record reader 只查询一次初始文件位置，之后随 64-byte header、payload read 和 skip 维护逻辑位置，不再对每个字段调用 `BufferedReader.tell()`。112,300 条真实记录的 worker/parent 双读会消除 898,400 次位置查询；隔离双扫描 wall/CPU 中位分别降低约 34.5%/33.8%，完整 molecule-name pipeline 的父进程 CPU 占比由 49.6% 降至 38.4%、wall 中位降低约 10.5%。字段顺序、非零起始 offset、截断检查和 `.moname` 哈希均由回归锁定。
- molecule name 使用紧凑 species ID 和唯一名称表；112,300 molecule/923,320 range 压力下 parent RSS 从约 850 MiB 降到 272 MiB，文本输出与 HDF5 语义保持一致。
- route 聚合直接生成唯一 species-pair 计数；100,001 个合成 event 只保留 2 个 Counter entry。真实 5 帧轨迹的 HDF5 语义指纹及 `.route`、`.reactionabcd`、`.reaction` 哈希在 1/8 进程间保持一致。
- route 并行度按扫描工作量和 atom task 数共同自适应；12,326 原子 × 5 帧反馈环由强制 8 workers 的 1.612 秒降至父进程直跑的 0.168 秒，且不再产生约 105–106 MB 的 child RSS。真实 450 原子 × 10,001 帧输入在请求 64 核时由 64 限制为 8 workers；配对监控中 route 为 0.261→0.116 秒、峰值进程数 65→9、聚合 RSS 1.85→0.56 GB。聚合 RSS 会重复计算共享页，只用于同机相对比较；生产全轨迹仍需实测 Slurm MaxRSS。
- 默认 species timeline 使用 64 MiB aggregation block 和最多 256 MiB 的可回退 observation spool；125 万行高占用压力中 Python peak 从约 5.94 MiB 降至 0.42 MiB、wall 中位约从 1.35 秒降至 0.54 秒。稀疏路径消除了逐帧 Python Counter 和逐 partition 重扫，真实 5 帧及 40 组固定种子差分均与旧 Counter 参考逐字节一致。
- SMILES/route 结果在调用线程中乱序接收并严格恢复输入顺序；仅实际队头 backlog 压缩到可自动清理的临时文件。后台持续收集实验未保留在最终代码中。
- ordered spool 对不超过 4 KiB 的 `str`/`bytes`/`None` 使用 tagged raw 编码，超长或结构化结果继续 LZ4；10 万短名称反馈环临时字节减少约 85%、put+pop 中位缩短约 19%。
- ordered spool 的全任务 memmap 索引不再在创建时整表写零；1,000 万任务独立进程基准中，初始化 MaxRSS 增量由约 159.5 MB 降至约 0.25 MB，只有实际 pending 的索引页会被触碰。
- HDF5 使用同目录临时文件、文件锁、状态属性和原子发布；失败构建不会覆盖正式文件。
- HDF5 属性和 molecule-stage 日志分别报告 molecule/reaction write 秒数、批次数以及 definition/range/byte 高水位，可在生产作业中直接区分 worker 等待与单 writer 瓶颈。
- 顺序 HDF5 writer 的默认 raw chunk cache 从 128 MiB 降至 1 MiB；4,904 万 range 反馈环中 MaxRSS 增量中位约降低 58%，wall 未回退。显式 cache 覆盖继续有效，实际值写入 HDF5 根属性。
- 新增 `reacnetgenerator-check-timed-output`：分块验证 schema、offset、range、event block 和计数一致性，并生成不受 molecule/reaction 内部 ID 顺序影响的语义指纹。来源路径另存为 provenance 指纹，默认不把目录迁移误报为结果变化。

涉及的主要文件为 `reacnetgenerator/_timedoutput.py`、`_timedoutputvalidate.py`、`timedoutputcheck.py`、`_packedbool.py`、`_path.py`、`_reaction.py`、`_detect.py`、`reacnetgen.py`、`commandline.py`、`tools.py` 和 `tests/test_timedoutput.py`。这些修改消除了已定位的 IPC 放大和完整矩阵在每个进程中常驻的问题，但仍需用新的大轨迹验证峰值 RSS、临时磁盘和运行时间。

## 11. 测试与选型计划

`/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 可以用于功能回归、输出一致性和初步性能测试，但小轨迹通过不能证明大轨迹不会 OOM。建议测试矩阵如下：

- 四种模式：两个功能都关闭、仅分子时间、仅反应事件、两者都开启。
- 多种并行度：1、8、32、96。
- 记录每个 Step 3 子阶段的 wall time、峰值 RSS、`atomeach/conflict` 临时文件峰值、page fault、临时空间和最终文件大小。
- 比较旧 CSV、HDF5 实现和原 SQLite 原型；必要时加入 Parquet 作为列式基线。
- 验证 `.moname`、`.route`、`.reactionabcd` 与原实现完全一致，并验证从详细事件聚合出的 `.reactionabcd` 一致。
- 对 ReacNet-Scope 的典型操作计时：按帧查分子、按原子查轨迹、按反应查发生时间、全量导入 SQL。

由于原始 OOM 数据已删除，还需要一份新的大轨迹，或由 `rp3.lammpstrj` 构造可控放大的压力数据，验证内存是否真正受块大小约束。

### 可直接在集群执行的结果验收

仓库现提供可直接提交的 `scripts/slurm-production-validation.sh`；参数、隔离目录、证据文件
和较低 CPU/48 CPU 对照方法见 `docs/slurm-production-validation.md`。脚本不会写死集群的 partition、
account、memory 或 time limit，也不会复用已有结果目录。

先为可信结果建立一次基线：

```bash
reacnetgenerator-check-timed-output trusted.timeline.h5 \
    --output trusted.timeline.manifest.json
```

候选作业完成后做完整性校验和语义比较；退出码 `0` 才表示通过，`1` 表示结果不一致，`2` 表示文件不完整、损坏或命令参数错误：

```bash
reacnetgenerator-check-timed-output candidate.timeline.h5 \
    --baseline trusted.timeline.manifest.json \
    --output candidate.timeline.manifest.json
```

若还要求输入文件的保存路径逐字一致，增加 `--require-same-source-paths`。未安装 entry point 时，可等价使用 `python -m reacnetgenerator.timedoutputcheck`。验证过程不展开逐帧 molecule rows 或重复 reaction counts；默认按 4096 行和 16 MiB payload 分块。

运行中和完成后分别保存资源证据：

```bash
sstat --jobs=1089747.batch \
    --format=JobID,AveCPU,AveRSS,MaxRSS,MaxVMSize
sacct -j 1089747 --units=G --parsable2 \
    --format=JobIDRaw,State,ElapsedRaw,AllocCPUS,TotalCPU,CPUTimeRAW,MaxRSS,AveRSS,MaxVMSize,ExitCode
```

同时保存 HDF5 manifest、标准输出/错误、提交命令、代码 commit、节点和临时磁盘峰值。用
`TotalCPU / (ElapsedRaw × AllocCPUS)` 比较 CPU 效率，并同时比较 Step 3 wall time 和总
core-hours；若较低 CPU 与 48 CPU wall 接近，应选择较低 allocation，而不是继续为低
利用率付费。n3 当前关闭 accounting，实际验收已改用 GNU time、进程树采样和 `scontrol`
allocation；不要使用该节点返回的无效 `sstat` AveCPU。

### 旧作业处置建议

若作业 `1089747` 仍停留在 Step 3、`.route` 仍未生成、CPU 仍约为 `2.1/64` 核，而且剩余 wall-time 不足以留出 Step 4–6 和结果校验时间，建议先保存上述证据，再取消并用修复分支重算。当前临时 HDF5 不是可恢复 checkpoint，不能直接接续到新实现；保留它只适合诊断。若作业已经进入 route/reaction 尾声或即将完成，可让它作为可信基线完成，但不要把其 48 小时以上的 Step 3 当作新实现的性能预期。

5% 轨迹没有暴露问题并不矛盾：该测试把 frame 数和 `N×T` 矩阵至少缩小约 20 倍，HDF5 cache 能掩盖小写入成本，ordered-result backlog 还未进入稳态；同时 unique molecule、长寿命 range、reaction transition 和结构匹配工作量不保证按抽帧比例线性缩小。短测试主要证明功能可运行，不能证明长时间背压、峰值 RSS 或 64 核扩展性。

## 12. 完成标准与下一步

完成标准：

- 峰值内存主要由块大小和进程数决定，不再随总帧数或事件总数无界增长。
- 同时开启两个功能时复用同一遍计算，内存和耗时不应近似两项相加。
- 原有文本文件的语义和格式保持兼容。
- 中间文件可校验、可版本化、失败时不覆盖已有完整结果。
- ReacNet-Scope 能流式导入 SQL，而无需重新读取原始轨迹。

建议后续顺序：先完成小轨迹功能回归；再增加 Step 3 分阶段 RSS 和临时磁盘监测；随后用同一份放大轨迹比较旧实现、HDF5 实现和原 SQLite 原型；最后根据 ReacNet-Scope 的真实查询负载确认长期格式。

## 13. 50k 帧扩展性复核与未解决边界

同一份 10,001 帧、450 原子的甲烷轨迹按输入列表重复到 20,002/50,005 帧后，当前实现的
no-HMM Path wall 为 0.757/1.102/2.775 秒；5× 帧数只放大 3.66× wall。molecule、matrix、
route 以及构建期间临时目录都未出现超线性增长。HMM 1×/5× 的 Path wall 为
0.399/1.162 秒；过滤后工作很小，因此约 1 核的短阶段利用率属于正常调度结果。

5× 运行中记录的 reaction writer wall 一度从 0.130 秒增至 1.352 秒，但 writer-only
三轮回放稳定为 0.200--0.207 秒；完成顺序与 transition 顺序无差异。重建完整
`450 × 50,005` reaction 矩阵后，纯计算在本机 10 个 CPU 上于 10--16 processes 饱和，
24/36 processes 不再加速；接入 writer 时更多进程只增加父进程被抢占的 wall。全部
并行度输出哈希一致。详细表格、HDF5 计数及 SHA-256 见
`docs/step3-performance-optimization-log.md`。

因此没有保留“修改 HDF5 压缩”或“全平台限制为 10/16 workers”的实验性改动。当前仍缺
真正 64 核 Slurm 节点的证据：必须记录 allocation/affinity、实际 worker 数、阶段 wall、
writer 属性、`TotalCPU/Elapsed`、MaxRSS、临时盘峰值和语义 manifest。生产上若 36 个
reaction workers 在 64 核 allocation 内仍长期只有约 2.1 核，则更像 affinity/cgroup、
worker 启动或未命中该计算阶段的问题，而不是这次本地已排除的 writer 超线性。

为让这项验证无需额外注入脚本，启动日志现已自动记录 `nproc`、可见 CPU、逻辑 CPU、
affinity 范围和 `SLURM_CPUS_PER_TASK`，并在过度订阅或 Slurm/affinity 不一致时告警。
该诊断不自动限制显式 `nproc`；生产人员应将其与阶段 worker 日志及 `sacct` 一起保存。

## 14. 稀疏 reaction 活动 transition 索引

route 现在把所有原子的相邻帧 molecule-ID 变化合并到一个 `uint8[T-1]` 临时 mmap；多个
worker 只做幂等 byte 写 1，不通过 IPC 返回事件数组。reaction 只扫描这些活动时间点，
并用活动任务数计算 worker/chunksize/in-flight；索引不可用时自动保留旧全扫描路径。
额外临时空间仅 `T-1` bytes，HDF5 根属性保存总/活动 transition 数和索引是否可用。

50,000 × 2,048 的 195.312 MiB 隔离矩阵只有一个活动 transition 时，reaction 中位由
0.3531 降到 0.00183 秒；显式列出全部 transition 与旧 range 路径为 0.998×，稠密输入
没有可测回退。真实 10,001 帧 HMM 轨迹只活动 1,655/10,000 个 transition，reaction
0.1758→0.0650 秒（约 -63.0%）；语义 manifest 和 reaction 文本保持一致。no-HMM 的
10,000/10,000 全活动时 route 与旧基线持平，因此不会为了稀疏优化惩罚稠密常见路径。

## 15. cheap-fork SMILES 复用 worker 解码结果

廉价 SMILES 记录在 Linux-like `fork` 下按 64 条分批时，worker 原先先解压 atoms/bonds
生成名称，父进程随后又从 molecule 文件解压相同结构以写 `.moname`、timeline 和执行
fallback/miso。现在只有 HMM/filter 记录的最大压缩结构不超过 64 KiB 时，该路径才返回
name、atoms、bonds；父进程若需要 timeline，只读取第 4 个压缩字段。串行、
`spawn`/`forkserver`、`chunksize=1`、最大值未知或超过上限时仍返回紧凑的 name-only
结果，避免全局平均值掩盖少量巨大分子并放大 IPC payload。

在由真实 5 帧、12,326 原子输入产生的 5,615 个 molecule records 重复到 112,300 条后，
同一完整 molecule-name pipeline 的交替 A/B 中位数如下：name-only 为 3.2168 秒、父进程
CPU 3.2480 CPU-s；复用结构为 2.8879 秒、父进程 CPU 2.7517 CPU-s，分别降低约 10.2%
和 15.3%。总 CPU 约降低 2.9%，平均用核由 2.303 提高到 2.502。固定可写
`MPLCONFIGDIR` 后的受控 RSS 为父进程 274.4→276.9 MiB、最大 child 38.5→39.8 MiB；
ordered spool 的典型峰值约 0.072 MiB。首次测得的约 45 MiB child 增量已确认来自
Matplotlib/font-cache 子进程污染，不作为实现依据。

重新生成 5,615 条真实记录后，压缩结构平均/最大值为 339.3/4,108 bytes，低于 64 KiB
上限，因此上述 A/B 的真实路径仍会命中；混合分布回归则在总平均仍廉价、单条最大值为
65,537 bytes 时强制回到 name-only。

同一 112,300 条记录的端到端 HDF5/matrix/route/reaction 复跑中，1/8 进程的语义指纹均为
`7d1aea66c484870b90867b690084783c84ae9887dfc99ebc4812f8b75a2bb147`，`.moname`、
`.route`、`.reactionabcd` 逐字节一致。压缩结构回传、对所有 decoded 结果再次紧凑编码、
更小 chunksize 及 reaction bitmap 候选均因 wall 回退或收益不足而未保留。该数据仍只是
本地 fork 反馈环；生产 64 核 Slurm 的阶段 wall、CPU affinity、实际 worker、MaxRSS 和
语义 manifest 仍是完成性能结论所必需的证据。

## 16. `.moname` 专用编码与压缩 record 字段分发

复用 worker 结构后，112,300 条真实记录的父进程 profile 显示通用递归
`listtostirng()` 成为新的明确热点。`.moname` 的格式固定为三层分隔符，因此改为专用的
atoms `;` join 与 bonds `,`/`;` join，不再对每个标量递归判断类型。交替 A/B 中，完整
molecule-name pipeline 的 wall 中位由 2.4671 降到 2.2563 秒（约 -8.5%），父进程 CPU
由 1.7836 降到 1.3868 CPU-s（约 -22.2%），输出 SHA-256 不变。

压缩 record reader 也把每条记录的 `dict + set` 字段分发改为一次预计算字段位置和定长
list；112,300 条、61 MiB 文件的隔离中位约 0.1655→0.1416 秒（约 -14%）。乱序字段、
非零文件起点和截断错误契约均由既有测试覆盖。最终源码三轮 wall/parent CPU 中位为
2.3212 秒/1.3848 CPU-s。2/3/4/6 worker 扫描仍以 2 workers 最快；小 generic spool
免压缩和跳过无 byte-limit `WriteBuffer` 计数均没有稳定收益，未进入源码。

## 17. SMILES 不变量与小乱序结果缓存

SMILES converter 现在每个 parent/worker 实例只编译一次 atom-name radical pattern，并只
解析一次 `Chem.MolFromSmiles("")` 空模板；每个 molecule 仍从该不可变模板创建独立
`RWMol`。112,300 条真实记录的交替 A/B 中，两项分别降低约 2.8%/3.2% wall 和
3.8%/3.4% child CPU，`.moname` SHA-256 不变。直接 `Chem.RWMol()` 虽看似更简单，却既
更慢又改变输出，已被否决。

保序结果恢复增加 1 MiB 的**已编码**内存前置预算，小 backlog 不再必经临时文件，超过
预算立即沿用 disk spool；data 文件和稀疏 mmap index 也只在第一次 spill 时创建。
112,300-result 隔离压力下 wall 中位约 1.518→0.755 秒、磁盘写入 35.15 MiB→0，估算
内存峰值仅 0.135 MiB；完整 SMILES 路径 wall 持平但 parent CPU 稳定降低约 7%--12%。
日志新增 encoded/disk/peak-memory 分项，生产复跑应同时保存这些值，不能只根据
`disk written=0` 推断没有乱序 backlog。

## 18. 重复 SMILES 名称与短 molecule-range 热路径

每个 SMILES parent/worker converter 现有一个按估算字节限界的 1 MiB LRU。键只包含
atom-type 顺序和映射到局部 atom position 后、保持输入顺序的 bond triples；因此能复用
全局 atom ID 不同但有标签结构完全一致的名称，又不会把仅同构但局部顺序不同的输入误
合并。放不进预算的单记录会在分配局部 bond key 之前直接绕过。112,300 条真实 decoded
record 中命中 111,627 条，独立转换中位 2.389→0.593 秒；完整 Step 3 的收益受父进程
writer 限制，不能用该微内核比例外推。

无 molecule frame/timestep filter 且 frame 列表不超过 64 项时，range 生成改为固定上界
的线性扫描；长列表和过滤路径仍走原 65,536-row NumPy block。112,300-record 三组 A/B
中，完整 Step 3 wall 中位 13.999→12.129 秒，molecule 9.300→7.433 秒，writer
5.977→4.260 秒，parent CPU 14.786→12.598 CPU-s，语义 manifest 不变。

内部 range dtype 判断改用 `dtype.kind`，但负值、逆序和越界校验不变；SMILES 首次解码
atom IDs 时直接生成 HDF5 schema 所需的 `uint64`，下游不再逐 molecule 复制。`int`/
`uint64` 三组完整 A/B 的 wall 中位为 11.771→11.377 秒，writer
3.854→3.109 秒，parent MaxRSS 无上升。bond ndarray 原型增加 child CPU 且 wall 未改善，
未保留。

最终 5-frame 真实输入的 Step 3 为 1.257 秒，HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`，四个文本输出逐字节
不变。上述压力只重复 molecule records 且仍为 5 帧，matrix conflict 分布不代表生产；
真实 64 核 Slurm 仍需保存 allocation/affinity、实际 worker、四阶段 wall、
`TotalCPU/Elapsed`、MaxRSS、writer 属性、临时盘峰值和 manifest。

## 19. molecule-range ID run staging 与单区间 writer 快路径

`TimedOutputStore` 不再为每个小 range block 建一个 molecule-ID ndarray，而是暂存
`(molecule_id, range_count)`，在 batch flush 时一次 `np.repeat()`；逻辑字节记账和
65,536-row/64 MiB 上限不变。单 range 使用标量完成负值、逆序和越界检查，多 range
继续向量校验；bondless definition 不再暂存两个空 bond ndarray。

112,300 个单 range/bondless molecule 的受控反馈环中，完整 wall 中位
0.935→0.341 秒，writer 记账 0.894→0.303 秒，HDF5 大小和 batch 指标不变。ID staging
隔离微基准的 Python allocation peak 约 656→300 KiB。该负载专门放大小对象路径，不代表
49,047,393-range 生产吞吐。

真实 5-frame 输入的语义指纹和四个文本 SHA-256 继续逐项一致；非 GUI 回归为 241
passed，Black/Isort/flake8/compileall/diff check 通过。真实 64 核 Slurm 验收仍是最终门槛。

## 20. 周期 Open Babel 成键的 cKDTree 候选加速

Step 3 已降至秒级后，当前本地总耗时转由 Step 1 主导。Open Babel 的周期
`ConnectTheDots()` 对 12,326 原子逐帧扫描所有 atom pair；最终实现只在默认 Open Babel、
axis-aligned orthorhombic PBC、至少 2,048 原子的安全条件下，以单线程周期 cKDTree 生成
O(N+E) 候选。共价 cutoff、0.4 Å 下界、phosphorus 规则、过价/小角清理和
`PerceiveBondOrders()` 均保持 Open Babel 语义；triclinic/small cell、数值 cutoff 边界、
无效坐标或半径整帧回退原实现，非周期和显式 ASE 模式不变。`scipy` 已列为直接依赖。

独立单进程 A/B 的 Step 1 wall 中位为 8.564→1.729 秒（约 4.95×），CPU 为
8.555→1.728 CPU-s，MaxRSS 为 299.95→303.39 MiB。5-process 顺序交替 A/B 的 wall 为
2.689→0.512 秒，child CPU 为 12.165→2.115 CPU-s，reported max child RSS 仅增加约
3.13 MiB。ASE padded-bin 原型虽能降到 4.111 秒，却使用 25.422 CPU-s/507.7 MiB，已
否决。真实 5 帧 molecule temp 的大小和 SHA-256 与旧路径逐轮一致；equal-z group 每帧
20 种随机次序也只产生一种 signature。

完整 no-HMM/timeline/reaction-event 流程的 Step 1 为 1.496 秒，总计 2.643 秒；HDF5
语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`，四个文本 SHA-256
逐项不变。非 GUI 回归为 254 passed，静态检查通过，约 294 MiB 可重建临时产物已精确
清理，原始轨迹哈希不变。真实 64 核 Slurm 仍需保存 accelerator 回退、实际 worker、
阶段 wall、`TotalCPU/Elapsed`、MaxRSS/aggregate memory、临时盘峰值、I/O 和 manifest；
本地 4.95× 不能直接外推为全轨迹比例。

生产日志现会在首帧立即报告 bond-perception mode，并在 Step 1 结束时按固定大小 Counter
汇总 cKDTree 命中及各类 Open Babel 回退；每帧 IPC 只多一个 small int（约 2 encoded
bytes）。真实样例报告 `periodic-cKDTree=5`。LAMMPS dump 解析也不再为每个 atom 创建 ASE
`Atom` 后排序，而是按 atom ID 直接填充 atomic-number/position 数组和 seen bitmap；重复、
缺失、越界 ID/type 会显式失败。最终真实单次 Step 1 为 1.409 秒，molecule temp 哈希与
旧路径一致；parser 的独立 A/B 因工具额度限制未完成，不单独宣称比例。生产验收需保存
首帧模式和最终模式计数。

真实输入强制 `fork`/5 workers 的 smoke test 已确认 mode 能经 worker IPC 回到父进程并
汇总为 `periodic-cKDTree=5`，molecule temp 哈希保持不变。

本补充验证的 4 个 molecule temp 和 font cache（约 13 MiB）因工具额度限制未能删除，
精确路径已记录在性能日志；它们不属于工作区，可在额度恢复后清理。

## 21. n3 的 48 CPU Slurm 验收

已通过 DPDispatcher 在 `node3` 完成真实 Slurm 验证。节点只有 48 个物理 CPU，因此满节点
配置为一个 task、`CPUs/Task=48`，不是 64。任务 3615 和 3619 的 10,001-frame 结果、任务
3616 的 8-CPU 对照具有完全相同的 HDF5 语义指纹和三个关键文本哈希，排除了进程数导致
的非确定性。48 CPU 相对 8 CPU 把 Step 1 从 20.785 秒降至 5.957 秒；Step 3 均约 1.45--
1.47 秒，因为短工作量按设计只选择 1/8/7 个 SMILES/route/reaction workers。

任务 3621 在 48 CPU 上完成重复 5 次的 50,005-frame 工作负载：GNU wall 38.44 秒，平均
24.41 cores，Step 1/2/3 为 28.595/0.701/3.581 秒；Step 3 四阶段为
0.456/0.419/0.534/2.068 秒。770,489 molecule ranges 只用 12 batches，217,953 reaction
rows 用 54 batches，未复现逐 molecule resize/write 的长尾。单进程 MaxRSS 为 320,456
KiB，包含 fork 共享页重复计数的进程树 RSS 采样峰值约 9.10 GiB，临时目录峰值约
17.8 MiB，无 OOM、swap 或错误日志。

n3 OpenBabel 为 3.1.0，本地历史基线为 3.2.1；相同输入在 n3 多出 2 个 molecules，并引起
少量 range/reaction 计数差异。n3 内部 8/48 CPU 和两次 48 CPU 输出完全一致，因此这是
需要固定依赖版本的跨平台基线问题，不是当前并行/批量 writer 的正确性回归。完整任务号、
指纹、SHA-256、资源数值、提交失败 3608--3611 及修正后的环境任务 3612 均记录在
`docs/step3-performance-optimization-log.md`。

## 22. n3 的同 range 量级与实际 48-worker 验收

任务 3631 用真实 10,001-frame 输入的 320 次顺序重复构造 3,200,320 frames，得到
49,310,414 canonical ranges，和旧 5% 作业的 49,047,393 ranges 同量级。它不是化学分布
意义上的真实全轨迹，但能直接压力覆盖旧瓶颈的 range/HDF5 维度。Slurm 为一个 task、48
CPUs/Task；Step 1 和 reaction 都实际启动 48 workers。

Step 1/2/3/4 为 2102.599/119.126/156.024/163.036 秒；Step 3 的
molecule/matrix/route/reaction 为 7.846/18.692/27.076/101.086 秒。49,310,414 ranges
只用 753 batches，13,949,118 reaction rows 用 3,406 batches；旧逐条 resize/write 的
数十小时长尾未复现。完整 RNG wall 为 2541.378 秒，GNU time 平均约 24.37 cores，单进程
MaxRSS 约 3.82 GiB，swap 0。进程树 RSS 因 fork 共享页重复计数不可作为物理内存；一次
49 进程 PSS 抽样约 4.75 GiB，节点同时可用内存约 235 GiB。

wrapper、科学程序和 HDF5 validator 均退出 0，`.route`、`.reactionabcd` 和报告完整；
manifest 为 3,200,320 frames、508,101,760 logical molecule rows、3,156 molecules、
500 reaction types，语义指纹
`90e9618925ec7eab870235643a35f08bc704a5e14669408e5cbdc25f8dee012b`。约 2.9 GiB 的
结果已由 DPDispatcher 回传至
`/private/tmp/rng-dpdispatcher-n3-20260808/validate-48-3m/results/run-3631`，并通过
`output.sha256` 全量校验。当前工作树 validator 再次读取该 HDF5 后退出 0，产生的 manifest
与远端文件逐字节相同；最终核心非 GUI/非 benchmark 回归为 261 passed、48 deselected。
