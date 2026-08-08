# Step 3 性能优化修改日志

## 2026-08-06：批量化 timed-output 写入

### 基线

- 分支：`perf/optimize-timed-molecule-output`
- 修改前提交：`b332ce99`
- 现象：分子识别结果由主进程逐个消费，每个普通分子约触发 8 次 HDF5 `resize + write`；主进程消费速度低于 worker 生产速度后，`max_inflight=2*nproc` 形成背压，多数 worker 等待。
- 原始最小复现：100 个简单分子触发 801 次 HDF5 `resize`。
- 无筛选 range profile：扫描 1,000,000 个 frame 约耗时 0.468 秒，产生约 4,001,103 次 Python 调用。

### 修改内容

#### `reacnetgenerator/_timedoutput.py`

- 分子定义、原子、键、offset 和存在区间改为列式有界缓冲。
- 缓冲在以下任一条件满足时写出：
  - 4,096 个分子；
  - 4,096 个分子存在区间；
  - 约 64 MiB 待写数值/字符串数据。
- atom/bond offset 改为内存计数器，不再逐分子读取 HDF5 dataset 的末尾值。
- 缓存热路径 dataset handle，避免逐分子重复执行 HDF5 group lookup。
- reaction type、reaction event 和 transition block index 改为批量写入。
- reaction block index 仅保存当前批次；没有为全部 transition 新增两个常驻 `uint64` 数组。
- 保留一个 `bool` duplicate-detection 数组，开销约为每个 transition 1 字节。
- 完成文件新增以下运行统计属性：
  - `molecule_count`
  - `molecule_range_count`
  - `molecule_write_batches`
  - `reaction_type_count`
  - `reaction_event_row_count`
  - `reaction_write_batches`

#### `reacnetgenerator/_path.py`

- 分子名称阶段完成后立即 flush 剩余 molecule batch，避免缓冲跨越后续 matrix/route 阶段。
- 无筛选的分子存在区间使用 NumPy 差分生成，不再逐 frame 查询 timestep 字典。
- frame/timestep 筛选改为向量掩码；仅启用 timestep 筛选时才读取 timestep 映射。
- 相邻重复 frame 在区间压缩前去重，保持原有语义。

#### 测试

- `tests/test_timedoutput.py`
  - 增加分子批次统计与完成文件语义测试。
  - 增加 reaction 批次统计与完成文件语义测试。
- `tests/test_reacnetgen.py`
  - 增加无筛选 range 不访问 timestep 映射的回归测试。

### 验证结果

#### HDF5 调用信号

- 100 个简单分子：`resize` 从 801 次降为 9 次。
- 100 个单事件 transition：旧调用路径约 303 次 `resize`，优化后为 6 次。
- 5,000 个分子：2 个 molecule write batch，5,000 行均可由 `iter_molecule_timeline()` 正确读取。
- 5,000 个逆序提交的 transition：2 个 reaction write batch，block index 恢复出的 transition 顺序为 0 至 4,999。

#### Range profile

- 输入：100 次压缩，每次 10,000 个连续 frame，共 1,000,000 frame。
- 修改前：约 0.468 秒，约 4,001,103 次 Python 调用。
- 修改后：约 0.0055 秒，约 4,103 次 Python 调用。
- 本地 profile 仅用于比较相同调用路径，不能直接外推生产作业加速倍数。

#### 自动测试与检查

- `pytest tests/test_timedoutput.py -q`：18 passed。
- timed-output、tools 和 detect 相关测试组合：通过。
- molecule/reaction/filter/parm2cmd 相关 `test_reacnetgen.py` 子集：10 passed，39 deselected。
- Ruff lint：通过。
- Ruff format check：通过。
- `git diff --check`：通过。

### 已知限制与后续验证

- 尚未在作业 `1089747` 的完整生产轨迹上复跑；实际 wall time、CPU 利用率、文件系统吞吐和 page fault 仍需 Slurm 实测。
- 本地完整 `tests/test_reacnetgen.py` 会进入 Tk GUI 用例；当前无显示环境发生 Tk abort，因此本次使用非 GUI 相关子集和真实 bond 轨迹的 timed-output 端到端测试。
- 当前修改没有让多个进程直接写同一个普通 HDF5 文件；继续保持 single writer，以避免 h5py/HDF5 并发写入一致性风险。
- 若 molecule batching 后仍有明显低利用率，应分别 profile `_getatomeach()` memmap 构建、ordered SMILES result consumption 和 `.route` 聚合，避免把后续串行阶段误归因于 HDF5。

## 2026-08-06：Code review 后续修复

### Review 发现

- `_itermoleculeranges()` 虽然通过 NumPy 计算边界，但随后对一个分子的全部 range 执行 `starts.tolist()` 和 `ends.tolist()`；`TimedOutputStore.add_molecule()` 又构造 Python tuple 批次并通过 `np.fromiter` 重建数组。
- 原有 profile 使用连续 frame，每个分子只产生一个 range，没有覆盖生产轨迹中高度碎片化的路径。
- molecule buffer 在已有数据后先加入下一批 range、再检查阈值，可能让一个批次超过 4,096 行或 64 MiB。
- 单个 transition 包含超过 4,096 个 reaction event 时，也会一次性超过 reaction batch 行数阈值。
- 自动测试只使用 100 行并断言 store 自报的 batch 数，没有监控真实 HDF5 `_append` 入口。
- `_APPEND_ROWS` 同时控制 HDF5 chunk、provenance 和写入批次，调整一个用途会隐式改变其他行为。
- molecule/reaction buffer 由多组平行列表和计数器维护，flush/reset 的一致性依赖手工同步。

### 红灯反馈环

首次执行：

```text
pytest -q \
  tests/test_reacnetgen.py::TestReacNetGen::test_fragmented_molecule_ranges_are_emitted_as_numpy_blocks \
  tests/test_timedoutput.py::test_molecule_output_reports_bounded_write_batches
```

修改前结果为 `2 failed`：

- fragmented range 返回 Python `int` tuple，不能满足 NumPy block 断言；
- store 不能直接消费 NumPy start/end block，在 `int(array)` 处失败。

另外建立三个边界红灯：

- 80 字节测试阈值下，molecule batch 实际写出 89 字节，而预期在加入下一条可拆分 range 前 flush 为 65 和 24 字节两个批次；
- 同一阈值下，两个可独立写出的 molecule definition 被合并为 81 字节，而预期在第二个 definition 前 flush 为 41 和 40 字节两个批次；
- 单 transition 的 4,097 个 reaction event 只形成 1 个 event write batch，而预期为 2 个。

### 修改内容

#### NumPy range 分块直传

- `_itermoleculeranges()` 按最多 65,536 个输入 frame 扫描，输出最多 4,096 行的 NumPy start/end block。
- scan block 之间保留一个未闭合 range，并正确合并跨边界的重复或连续 frame。
- frame/timestep filter 在每个 scan block 上执行，避免再为单个分子的全部 range 构造 Python list/tuple。
- `TimedOutputStore.add_molecule()` 直接消费 NumPy block，并按当前批次的剩余行数和剩余字节数继续切片。

#### 严格批次边界

- molecule definition 和 range 均在加入前检查剩余字节；当前批次不能容纳时先 flush，不再先追加后越界。单个不可拆分 molecule 自身超过阈值时仍允许独立写出。
- 单个 transition 的 reaction event 按剩余行数/字节数切片；跨 event batch 后仍只保留一个完整的 `block_start`/`block_length` 语义。
- 4,096/4,097 行、缩小后的 byte 阈值和 scan-block 连续性均成为自动回归测试。
- 测试通过包装 `_append` 统计实际 HDF5 dataset 写入入口调用次数，不再只依赖文件中的自报 batch 属性。

#### 可维护性

- `_APPEND_ROWS` 拆分为 `_HDF5_CHUNK_ROWS`、`_PROVENANCE_BATCH_ROWS` 和 `_WRITE_BATCH_ROWS`。
- molecule/reaction 的列式缓冲分别封装为 `_MoleculeBatch` 和 `_ReactionBatch`，集中管理字段生命周期和原子 reset。
- 删除只由测试调用的标量 `_shouldprintmoleculetimelinerow()` 和 `_getmoleculeranges()`；测试直接验证生产使用的向量化 range 路径。

### 碎片化微基准

输入为 100,000 个彼此不连续的 frame：

- 修改前：约 0.006153 秒，生成 100,000 个 `(int, int)` Python tuple；
- 修改后：约 0.000466 秒，生成 26 个 `(ndarray, ndarray)` block；
- 本地相同调用路径约快 13.2 倍，同时消除了逐 range Python 对象和单分子全量 `.tolist()` 内存。

该微基准只衡量 range 压缩与传递，不代表完整 Step 3 或生产作业的总加速倍数。

### 最终验证

- `pytest -q tests/test_timedoutput.py`：21 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：11 passed，40 deselected。
- `pytest -q tests/test_tools.py tests/test_detect.py`：19 passed。
- Ruff lint：通过。
- Ruff format check：通过。
- `git diff --check`：通过。

### 生产验证建议

- 当前修复消除了 review 指出的碎片化 range Python 对象路径，并严格执行本地 buffer 边界；仍需用完整生产轨迹验证 Step 3 wall time、CPU 利用率、HDF5 增长速率和 `.route` 生成时间。
- 若 CPU 利用率仍持续偏低，应单独 profile ordered worker result consumption、`_getatomeach()` 和 route 聚合；这些阶段不应通过并发写同一个普通 HDF5 文件来规避。

## 2026-08-07：Route worker 与矩阵构建临时内存优化

### 剩余放大路径

在 timed-output 批量写入和碎片化 range 修复完成后，继续审计 Step 3 的 route 与 atom-frame matrix 路径，发现三处仍会随单项规模放大的临时对象：

1. `_calculate_atom_route()` 通过 `np.flatnonzero(timeline)` 为每个 worker 创建最长为 frame 数量的 `intp` 索引；并通过 `np.diff()` 创建与 molecule ID dtype 等宽的临时数组。
1. no-HMM route 聚合为每一个被接受的 event 保存 `rr.reshape(1, 2)` 小 ndarray 视图；这些视图持续引用完整 worker 结果，最后 `np.concatenate()` 的输入对象数等于 event 数而不是 atom 数。
1. `_getatomeach()` 对一个分子的完整 signal 执行 `np.flatnonzero()`，并用 `np.nonzero(selected)` 为 overlap cell 创建两条 `intp` 坐标数组。

### 红灯反馈环

新增的最小复现首先得到以下失败：

- 一个 atom 的 10,000 个重复 route event 使 `np.concatenate()` 接收 10,000 个小 ndarray，而目标上界是每个 atom 一个二维 block；
- 100,000-frame 密集 timeline 直接调用了一次 `np.flatnonzero(timeline)`；
- 将 matrix frame scan 测试阈值缩小为 16 时，旧路径仍对完整 40-frame signal 执行一次 `flatnonzero`；
- 同一 matrix 测试对二维 `selected` block 调用了 3 次 `np.nonzero()`；
- 真实 bond 轨迹的 HDF5 中缺少可用于生产定位的 Step 3 子阶段计时属性。

对应反馈命令均在修改前失败、修改后通过，覆盖 route 对象数量、worker 索引路径、matrix scan block、overlap 语义和阶段计时元数据。

### 修改内容

#### Route worker

- 密集 timeline 先用 `np.count_nonzero()` 判断是否需要过滤；全非零时直接复用 memmap slice，不再复制完整 timeline。
- 稀疏 timeline 使用布尔 mask 过滤零值，避免创建每帧 8 bytes 的 `intp` 非零索引。
- molecule change 检测改为 1 byte/cell 的布尔 mask，并通过 `np.not_equal(..., out=...)` 写入；不再创建与 molecule ID dtype 等宽的 `np.diff()` 数组。
- 随机稀疏 timeline 差分测试逐项对照旧算法的 route pair 和 `.route` 文本，保持原有 frame 编号语义。

#### no-HMM route 聚合

- 仍按原有 `have_added` 规则保留同一最小 atom 上的重复事件，并排除其他 atom 对同一 molecule pair 的重复贡献。
- 将逐 event 的 ndarray view 改为布尔筛选后每个 atom 最多追加一个二维数组。
- 只有一个 atom block 时直接返回该数组，避免无意义的整块 `concatenate` 复制。

#### Atom-frame matrix 构建

- signal 按最多 1,048,576 frame 扫描；每个 `intp` frame index block 的逻辑上界约为 8 MiB，不再随单个分子的总存在帧数增长。
- frame block 成为外层循环，避免在 molecule atom 分块时重复扫描 signal。
- overlap 检测由两条 `np.nonzero()` 坐标数组改为布尔 overlap/conflict block；单个 cell budget 仍为 1,048,576。
- 每个 frame/cell block 使用完成后显式释放临时数组，避免相邻迭代在右值计算期间同时保留两批内存。
- 两个完全重叠的测试分子验证了 `atomeach` 后写覆盖和 `conflict=True` 的原有语义。

#### 生产可观测性

- HDF5 根属性新增：
  - `step3_molecule_seconds`
  - `step3_matrix_seconds`
  - `step3_route_seconds`
  - `step3_reaction_seconds`
  - `timed_output_write_seconds`
- 日志同步输出四个 Step 3 子阶段 wall time，下一次 Slurm 复跑可直接区分 molecule、matrix、route 和 reaction 瓶颈。

### 本地微基准

这些数字来自相同 Python/NumPy 环境的单路径基准，只用于说明临时分配和局部计算变化：

- 2,000,000-frame 密集 route worker：`tracemalloc` peak 从 17.167 MiB 降至 1.911 MiB，约降低 9.0 倍。
- 同一密集扫描的中位时间：3.291 ms 降至 0.360 ms，约快 9.14 倍。
- 50% 零值 timeline：旧 3.488 ms，新 3.628 ms，约慢 4%；这是为显著降低临时内存付出的稀疏路径扫描代价。
- 100,000 个 route event 的聚合模型：逐 row view peak 16.023 MiB，按 atom block 为 0.097 MiB，约降低 165.6 倍；该数字不包含最终业务结果本身。
- 5,000,000-frame 全非零 signal index：全量 `flatnonzero` peak 38.147 MiB，1,048,576-frame 分块并及时释放后为 8.001 MiB，约降低 4.77 倍。
- 1,000,000 个全部 overlap cell：两条坐标数组 peak 15.259 MiB，布尔 overlap/conflict block 为 1.908 MiB，约降低 8.0 倍。

### 验证结果

- `pytest -q tests/test_timedoutput.py`：25 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：11 passed，40 deselected。
- `pytest -q tests/test_tools.py tests/test_detect.py`：19 passed。
- 官方下载 fixture 的端到端组合：6 passed、4 xfailed、1 teardown hash error；其中一个 `miso=1` teardown 曾出现一次未列入既有允许集合的 reaction hash，立即独立复跑同一用例为 1 passed。随机 route 差分和本地真实 bond 端到端测试均通过，因此将其记录为需继续观察的非确定性信号，而不是忽略或归因于本轮优化。

### 尚未解决

- route 和 SMILES 仍使用 ordered result consumption；它保证 `.route`/molecule ID 顺序和有界内存，但任务耗时高度不均时可能造成队头阻塞。改变该设计需要磁盘型乱序结果暂存或生产耗时分布证据，不能仅切换为 `unordered=True`。
- matrix memmap 的实际 page fault、临时磁盘吞吐以及 64/96 worker 下的总 RSS 仍需 Slurm 验证。
- 本轮局部微基准不能替代作业 `1089747` 同规模复跑；生产结论应以新增阶段属性、`sacct`、HDF5 增长和 `.route` 完成时间为准。

## 2026-08-07：有序 worker 队头阻塞与 SMILES IPC 优化

### 队头阻塞实证

最小复现使用 2 个 worker、`max_inflight=4`、`chunksize=1` 和 12 个任务；任务 0 睡眠 0.3 秒，其余任务睡眠 0.01 秒。旧的 `unordered=False` 路径中：

- 慢任务 0 完成时间为 `3945964.924743083`；
- 序号 4 及以后任务的最早开始时间为 `3945964.925591083`；
- 即任务 4 在慢任务完成约 0.8 ms 后才开始，证明 semaphore 只在 ordered iterator yield 后释放，feeder 被慢序号结果阻塞。

这不是仅由 HDF5 推断出的现象，而是 `run_mp` 生产路径的直接时间戳证据。

### 修改内容

#### 磁盘型有序结果暂存

- `multiopen()` 可在 worker 输入上附加连续序号，并使用 `imap_unordered()` 接收完成结果。
- `run_mp(disk_ordered=True)` 在调用线程中消费 unordered worker 结果，仅把被较早慢任务挡住的结果用现有 LZ4/pickle 格式写入临时数据文件。
- `uint64[total, 2]` memmap 保存 offset 和 length，索引磁盘开销为每个声明结果 16 bytes。
- 结果进入磁盘暂存后立即释放 semaphore，后续任务不再等待较早的慢任务；对调用方仍严格按输入序号 yield，molecule ID 和 `.route` 顺序不变。
- 曾实现后台持续收集原型，用于验证 slow single-writer 背压；真实 A/B 证明其不适合作为生产默认或备用 API，最终代码未保留该线程状态机。
- 该模式要求显式 `unordered=False` 和精确 `total`，默认 `run_mp` 行为不变。
- 正常完成、异常和 generator 关闭均通过上下文清理数据文件、索引和临时目录。
- 当前只在 `nproc > 1` 的 SMILES 和 route 两个已确认需要稳定顺序的 Step 3 路径启用。

#### SMILES worker IPC

- 原实现将第四个压缩 frame block 发送给 SMILES worker，worker 不处理它，却又把同一 bytes 对象随结果返回主进程。
- 新实现给 worker 只发送 atoms、pairs 和 levels 三个结构块；worker 也只返回 name、atoms 和 bonds。
- 主进程通过独立顺序句柄读取第四个 frame block。选择性 block reader 在结构句柄中 seek 跳过 frame payload，在 frame 句柄中 seek 跳过三个结构 payload，因此每个 payload 仍只从临时文件读取一次，仅 64-byte block header 被扫描两次。
- 读取顺序与磁盘暂存恢复后的结果顺序一致，分子 timeline 与 molecule ID 的对应关系保持不变。

### 调度微基准

第一版同线程暂存的环境与负载：2 个 worker，50 个任务，任务 0 为 0.3 秒，其余为 0.01 秒，`max_inflight=4`，每组 3 次并取中位数；计时包含 multiprocessing spawn/import 开销。后续后台 collector 结果见下一节。

| 每个结果的 NumPy payload | 普通有序 | 无序 | 磁盘有序 |
| -----------------------: | -------: | -------: | -------: |
| 0 bytes | 0.9385 s | 0.8311 s | 0.8297 s |
| 64 KiB | 0.9395 s | 0.8200 s | 0.8199 s |
| 1 MiB | 0.9924 s | 0.8739 s | 0.8801 s |

- 磁盘有序在三个 payload 档位均保留严格顺序，并比普通有序快约 11%–13%。
- 1 MiB 档每次约暂存 11.014 MiB，最大 pending 为 20–22 个结果；本地结果说明暂存开销没有抵消解除队头阻塞的收益，但不能外推集群文件系统吞吐。
- 新回归测试直接断言任务 4 在慢任务 0 完成前启动，并断言临时目录在结束后为空。

### IPC payload 微基准

合成 frame block 为 `np.arange(1_000_000, dtype=np.uint64)` 的现有压缩表示：

- 原 worker 结果的 multiprocessing pickle：3.818258 MiB；
- 去掉 frame block 后的 worker 结果：0.000484 MiB；
- 该特定合成输入约减少 7,897 倍 worker-to-parent IPC bytes。

该比值取决于实际 frame 稀疏度、atoms 和 bonds 数量，只证明不参与计算的 frame block 不应穿过 worker IPC，不能视为完整 Step 3 加速倍数。

### 仍需生产验证

- 该调度修复解决“较早慢任务阻止 feeder 提交后续任务”，不把普通 HDF5 改成多进程并发写入。
- 主进程的 range 整理与 HDF5 single-writer 吞吐仍由前述 NumPy 分块、严格有界 batch 和 dataset 批量 append 优化承担。
- 生产复跑需要同时记录四个 `step3_*_seconds`、ordered spool 日志、`sacct` CPU/RSS、临时目录容量和 I/O 吞吐；下一节继续验证并实现有界内存的后台结果收集线程。

## 2026-08-07：慢 single-writer 背压与后台 collector 实验（评估后回退）

### 初版暂存仍会复现的低利用率

第一版磁盘有序实现仍在调用方线程中读取 `imap_unordered()`。当较早结果很快完成、主进程正在整理 range 或写 HDF5 时，generator 停在 `yield`，因此不能继续接收已经完成的 worker 结果，也不能释放 semaphore。

确定性反馈环使用 2 个 worker、`max_inflight=4` 和 12 个任务；任务 0 立即完成，其余任务耗时 0.02 秒。调用方收到任务 0 后暂停 0.25 秒：

- 连续 3 次运行中，任务 6 均在调用方暂停结束后才启动；
- 启动滞后分别约为 0.53 ms、0.49 ms 和 0.81 ms；
- 暂存日志的最大 pending 仅为 0–1，说明内部 result-handler 并没有绕过 semaphore 背压。

该用例直接覆盖最初问题中的“worker 已完成计算，但主进程 single-writer 较慢，导致后续 worker 长时间等待”。

### 实验修改

#### 后台结果收集

- 独立 daemon collector thread 使用带 0.1 秒 timeout 的 `IMapUnorderedIterator.next()`，持续接收完成结果。
- 一个严格有界的内存直通槽保存下一序号结果；其他结果压缩后写入磁盘，主线程不为全部乱序结果建立 Python dict/list。
- 主线程只等待当前序号，恢复后仍严格按输入顺序 yield。
- 实验模式明确要求 `chunksize=1`，保证 collector 可以定时检查取消状态；SMILES 和 route 生产调用原本就使用该 chunksize。
- 进度条仍由主线程在调用方消费结果后更新，避免 worker 已完成但 HDF5 尚未写完时错误显示 100%。

#### 可取消 producer 与异常清理

新增 worker-error 测试首次运行时在记录 `run_mp failed` 后卡住，组合测试直到 76.37 秒被人工中断。根因是 Pool feeder 线程阻塞在无限期 `semaphore.acquire()`，`pool.terminate()` 无法让该线程退出，随后 `pool.join()` 永久等待。

- 初版 `produce()` 以 0.1 秒 timeout 获取 semaphore，并检查 parent-thread cancellation event；worker 异常、collector 异常和 generator 提前关闭均能终止进程池。
- 后续全量回归发现轮询会轻微扰动既有 unordered Step 1 的完成时序，并放大 `miso=1`“首个 isomer 名称作为代表”的历史非确定性。最终实现恢复正常路径的阻塞式 `acquire()`；异常路径先设置 cancel event，再主动 `release()` 一次解除 feeder 阻塞。这样既不轮询，也保留异常清理能力。
- 普通非 disk-ordered `run_mp` 也复用该取消路径，避免相同的异常挂死。
- 回归测试验证 worker 异常可传播、提前关闭可终止慢尾任务、精确 `total` 不匹配会 fail closed，且所有临时文件均被清理。

#### 临时磁盘回收

- 暂存数据文件在 pending result 清零时立即 truncate，并把 append offset 重置为 0；消费者追上 producer 后不会继续保留已消费数据的磁盘空间。
- 日志区分累计压缩写入量、峰值暂存文件大小和最大 pending 数量。
- 当 single-writer 始终慢于 worker 时，峰值磁盘仍等于最大实时 backlog；这是用磁盘换取 worker/consumer 重叠的必要容量，生产复跑必须监控节点临时盘。

### 后台 collector 微基准

环境：2 个 worker、50 个任务、`max_inflight=4`，每组 3 次取中位数，包含 spawn/import 开销。

| 场景 | 普通有序 wall | 后台有序 wall | 普通 worker span | 后台 worker span |
| --------------------- | ------------: | ------------: | ---------------: | ---------------: |
| 无瓶颈，空结果 | 0.9924 s | 1.0173 s | 0.5973 s | 0.5821 s |
| 慢任务 0，空结果 | 0.9988 s | 0.8107 s | 0.5846 s | 0.4622 s |
| 慢消费者，64 KiB/结果 | 2.0365 s | 2.0309 s | 1.5588 s | 0.5847 s |
| 慢消费者，1 MiB/结果 | 2.0167 s | 2.1049 s | 1.5833 s | 0.6446 s |

- 无瓶颈场景约有 2.5% 本地调度开销。
- 慢头场景 wall time 约降低 18.8%。
- 慢消费者场景的 worker 活跃区间缩短约 2.5–2.7 倍，说明计算工作不再按 single-writer 节奏锯齿式提交。
- 当消费者完全主导 wall time 时，总耗时不会因调度本身显著下降；1 MiB 结果场景还因约 22.5–24.0 MiB 累计暂存写入而慢约 4.4%。这进一步说明 HDF5 range/batch 优化仍是总耗时修复的主体，collector 负责计算与写入重叠。

### 有界内存检查

使用 2 个 worker、每个结果 1 MiB NumPy 数组、0.01 秒慢消费者和 `max_inflight=4`，分别在独立 Python 进程中运行：

| 结果数 | `tracemalloc` peak | 累计/峰值暂存文件 | 最大 pending |
| -----: | -----------------: | ----------------: | -----------: |
| 20 | 8.051 MiB | 9.012 MiB | 17 |
| 100 | 8.560 MiB | 49.065 MiB | 89 |

结果数和磁盘 backlog 增长 5 倍时，parent Python allocation peak 仅增加约 0.51 MiB，而不是随 100 MiB 业务结果累计；这验证了一个内存直通槽加磁盘暂存的有界内存目标。`tracemalloc` 不包含 NumPy/native allocator 和操作系统 page cache，因此生产 RSS 仍必须用 `sacct`/采样器验证。

### 实验反馈环

- 慢消费者用例在修改前连续 3 次失败，修改后连续 3 次通过。
- 慢头与慢消费者用例同时通过，证明两类背压没有相互回归。
- 实验期间覆盖了慢消费者、`chunksize` 约束和 collector 线程清理；最终保留的回归覆盖慢头、ordered/普通 worker 异常、提前关闭、结果计数不匹配、spool truncate 和多进程真实 bond 轨迹。

### 本地真实坐标轨迹对照

输入为交接文档指定的 `/Users/huangchen/Downloads/rng_test/rp3.lammpstrj`：5 帧、每帧 12,326 个原子、约 2.0 MiB。参数为 `C H O`、`--nohmm`、同时生成 molecule timeline 和 reaction event；分别使用 `nproc=1` 与显式后台 collector 的 `nproc=8` 独立运行两次。

| 指标中位数 | `nproc=1` | `nproc=8` | 比值 |
| --------------- | --------: | --------: | ----: |
| Step 3 molecule | 16.769 s | 8.503 s | 1.97× |
| Step 3 matrix | 0.105 s | 0.134 s | 0.78× |
| Step 3 route | 8.614 s | 2.273 s | 3.79× |
| Step 3 reaction | 0.330 s | 0.862 s | 0.38× |
| Step 3 总计 | 25.863 s | 11.810 s | 2.19× |
| 全流程 | 39.512 s | 30.147 s | 1.31× |

- 该样本只有 4 个 transition，reaction 阶段在 8 进程下由启动/IPC 开销主导；不能据此选择生产并行度。
- HDF5 write time 为单进程 0.351–0.363 秒、8 进程 0.240–0.344 秒，占 Step 3 很小部分；这个 5 帧样本的主要成本是 SMILES 和 route 计算。
- 8 进程 molecule collector 两次累计/峰值暂存约 1.567–1.568 MiB，最大 pending 为 2,881–3,564；route 累计写入约 4.892–4.917 MiB，但因及时 truncate，峰值文件仅 0.036–0.062 MiB。
- parent `ru_maxrss` 为约 330–391 MiB，单个 child 最大值约 283–313 MiB；该样本没有表现出随 8 个进程成倍增加 parent RSS，但 `RUSAGE_CHILDREN` 不是所有 worker RSS 之和，不能替代 Slurm 总 RSS。

#### 输出差分

两种并行度均得到：

- `molecule_count=5615`
- `molecule_range_count=6076`
- `reaction_event_row_count=469`
- `.route` SHA-256：`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`
- `.reactionabcd` SHA-256：`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`
- 排序后的 `.moname` SHA-256：`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`
- 排序后的 timeline 逻辑 SHA-256：`f2372e106ac86fd2cb0a624022707c52f4a244229d42f1a5b7abbde2e4cb3987`

原始 `.moname` 和顺序敏感 timeline hash 在 1/8 进程间不同，且两次 8 进程运行之间也不同；排序后的语义完全相同。原因位于既有 Step 1：frame detection 默认 unordered，`d[molecule]` 的首次插入顺序随 frame 完成顺序变化，随后 molecule ID 继承该顺序。后台 collector 对接收到的输入序号仍严格有序。若未来要求不同并行度下逐字节一致，需要单独定义稳定 molecule-ID 排序规则；不能把该既有顺序差异误报为本轮调度修复的结果错误。

#### Collector A/B 与最终默认策略

为隔离 collector 自身，保持所有 HDF5、range、SMILES IPC 和 route 内存修改不变，仅切换 `nproc=8` 的 ordered-result 策略：

| `nproc=8` 策略 | molecule | route | Step 3 总计 | 暂存特征 |
| --------------------------------- | -------: | ------: | ----------: | ---------------------------------------------------------------------- |
| 普通 `imap` 有序，中位数 2 次 | 6.897 s | 2.339 s | 10.710 s | multiprocessing 内部 reorder |
| 无条件后台 collector，中位数 2 次 | 8.503 s | 2.273 s | 11.810 s | molecule 峰值约 1.57 MiB、2,881–3,564 pending |
| 最终默认同线程磁盘有序，1 次 | 7.234 s | 2.270 s | 11.233 s | molecule 峰值 0.104 MiB、393 pending；route 峰值 0.012 MiB、11 pending |

- 无条件后台 collector 在这个真实小样本上使 molecule 中位时间增加约 23%，Step 3 增加约 10%；保持 worker 忙碌没有缩短消费者主导的关键路径。
- 最终默认的同线程磁盘有序只处理实际队头乱序，真实样本相对普通有序约有 5% 调度/压缩成本，但避免大结果留在 multiprocessing reorder cache，并在慢头合成基准中快约 11%–13%。
- 因此 SMILES/route 生产路径只保留同线程 `disk_ordered=True`；后台模式的代码与 API 在实验后删除，避免为没有真实 wall-time 收益的能力长期维护线程同步和额外异常面。

### 第一轮最终验证

- `pytest -q tests/test_timedoutput.py`：32 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：11 passed，40 deselected。
- `pytest -q tests/test_tools.py tests/test_detect.py`：19 passed。
- Ruff lint：通过。
- Ruff format check：通过。
- mdformat check：通过。
- `git diff --check`：通过。
- 临时 benchmark 脚本、ordered-result 暂存目录和真实轨迹输出目录均已清理；输入轨迹未修改。

## 2026-08-07：生产结果验证器与最新真实 A/B

### 修改内容

- 新增 `_timedoutputvalidate.py`，对完成状态、schema、必需属性、dataset shape/dtype、source/frame 引用、molecule ID/offset/range、reaction type/event/block/total count 做 fail-closed 校验。
- 大型 frame、molecule、range 和 event 表按 `block_rows` 读取；molecule payload 另受 `block_bytes` 限制。不会展开逐帧 molecule rows，也不会按 reaction count 复制事件。
- molecule 和 reaction 使用 SHA-256 record digest 的 multiset accumulator，比较时不依赖既有 Step 1 产生的内部 ID 顺序。物种和反应类型仅保留紧凑 digest/index；来源路径保留为 provenance，默认不属于结果语义。
- 新增 `reacnetgenerator-check-timed-output` 和 `python -m reacnetgenerator.timedoutputcheck`。可写原子 JSON manifest、与基线比较，并用退出码 `0/1/2` 区分通过、语义不一致和无效文件。
- CLI 明确拒绝 `--output` 覆盖输入 HDF5 或 baseline manifest。

### 测试反馈环

- 修改前，新增的 manifest 测试因 API 不存在而在 collection 阶段失败。
- 修改后覆盖：内部 molecule/reaction ID 反序仍相等；一个 atom ID 变化会改变 molecule 指纹；损坏 atom offset、计数属性、完成状态或 reaction block 均失败；关键大表每次读取不超过测试设定的 2 行；CLI mismatch 返回 `1`，无效/危险输出返回 `2`。
- `pytest -q tests/test_timedoutput.py`：39 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：11 passed，40 deselected。
- `pytest -q tests/test_tools.py tests/test_detect.py`：19 passed。

### 最新真实轨迹 A/B

使用只读源 `/Users/huangchen/Downloads/rng_test/rp3.lammpstrj`，复制到临时目录后分别运行最终代码的 `nproc=1` 和 `nproc=8`：

| 指标 | `nproc=1` | `nproc=8` | 变化 |
| --------------- | --------: | --------: | ------------: |
| Step 3 molecule | 16.553 s | 6.997 s | -57.7% |
| Step 3 matrix | 0.090 s | 0.106 s | +18.0% |
| Step 3 route | 8.630 s | 2.304 s | -73.3% |
| Step 3 reaction | 0.303 s | 0.839 s | +177.1% |
| Step 3 总计 | 25.617 s | 10.292 s | -59.8%，2.49× |
| 全流程 | 39.589 s | 28.469 s | -28.1%，1.39× |

8 进程 molecule spool 累计/峰值为 `0.230/0.188 MiB`、最大 687 个 pending；route 为 `1.272/0.012 MiB`、最大 11 个 pending。小样本只有 4 个 transition，reaction 的多进程启动/IPC 固定成本仍大于计算收益。

两份 HDF5 的 `frames`、`molecules`、`reactions` 指纹逐项相等，语义总指纹均为：

```text
3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb
```

共同计数为 `frame_count=5`、`molecule_count=5615`、`molecule_range_count=6076`、`logical_molecule_row_count=22103`、`reaction_type_count=383`、`reaction_event_row_count=469`、`logical_reaction_event_count=1295`。输入副本路径不同，因此只有 provenance/source fingerprint 不同；启用严格路径比较时按预期返回不一致。

验证器处理该 556 KiB HDF5 用时约 `0.069 s`。单独导入 NumPy/h5py/工具模块后的进程 RSS 约 `50.5 MB`，完成校验后的进程峰值约 `58.1 MB`，本样本增量约 `7.6 MB`；生产规模仍需以 Slurm `MaxRSS` 验收。

### 全量回归发现的既有 `miso=1` 顺序敏感性

排除无显示环境会原生 abort 的两个 Tk GUI 用例后，联网全量回归完成 113 passed、6 xfailed、3 xpassed；最初剩余一个 teardown error：同一 dump 的反应计数完全一致，但 O2 的代表名称由历史白名单中的 `[O][O]` 变为 `[O]=[O]`。根因不是 reaction event 计算，而是：

1. Step 1 本来就用 unordered frame detection，unique molecule 首次插入顺序受完成时序影响。
1. `miso=1` 当前实现并未按帮助文本所说选择“最高频代表”，而是选择该同构组首个遇到的 SMILES。
1. 初版可取消 producer 的 0.1 秒 semaphore 轮询改变了 feeder 调度，暴露了新的首到顺序组合。

固定到未修改 `HEAD b332ce99` 独立运行两次都得到历史允许的 SHA `e1f445...`。恢复阻塞式 acquire、仅在异常时主动 release 后，当前代码的定向端到端 SHA 用例再次通过，同时 worker error/提前 close 的 6 个调度测试通过。这里没有把新 SHA 加入白名单，也没有在性能修复中改写 isomer 选择语义；“最高频代表”的实现偏差应作为独立正确性任务处理。

最终 semaphore 调整后又运行一次 `nproc=8` 真实轨迹：Step 3 molecule `8.736 s`、matrix `0.128 s`、route `2.227 s`、reaction `1.006 s`、Step 3 总计 `12.148 s`。相对本轮单进程 `25.617 s` 仍缩短 `52.6%`（`2.11×`）；小样本单次 molecule 时长存在约 7–9 秒波动。最终 HDF5 继续通过基线比较，语义指纹仍为 `3381ff...98cb`。

### 最终本地测试状态

- `pytest -q tests/test_timedoutput.py`：40 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：11 passed，40 deselected。
- `pytest -q tests/test_tools.py tests/test_detect.py`：19 passed。
- 联网端到端定向用例 `test_reacnetgen[reacnetgen_param2]`：1 passed。
- 排除两个 Tk GUI 用例的联网全量：首轮 113 passed、6 xfailed、3 xpassed、1 个上述顺序敏感 teardown error；调度修正后已定向复验通过，未再次消耗约 5.5 分钟重跑全套。
- Ruff lint/format、Black、isort、mdformat、`compileall` 和 `git diff --check`：通过。

### Manifest 放大压力检查

使用 `TimedOutputStore` 生成两个只含紧凑数值行的合成文件，在独立 Python 进程中用默认 `4096 rows / 16 MiB` 分块校验：

| molecule/range rows | reaction rows | 校验 wall | import 后 RSS | 校验峰值 RSS | RSS 增量 |
| ------------------: | ------------: | --------: | ------------: | -----------: | -------: |
| 100,000 | 100,000 | 0.722 s | 51.3 MB | 64.3 MB | 12.9 MB |
| 1,000,000 | 1,000,000 | 6.975 s | 50.7 MB | 78.9 MB | 28.2 MB |

业务行数放大 10 倍时，wall 约放大 9.7 倍，RSS 增量约放大 2.2 倍而非 10 倍。RSS 仍受 Python allocator arena、HDF5 chunk/cache 高水位和紧凑 species/reaction-type index 影响，因此这里证明的是“没有全量事件列表和 range 展开”，不是严格数学常数内存。合成 HDF5 和脚本均位于 `/private/tmp`，记录结果后删除。

本轮真实 A/B、HEAD 对照、manifest、压力文件和临时 uv cache 均已清理；它们不可恢复但都可由只读输入重新生成。原始 `/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 未修改，清理后复核 SHA-256 为 `c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

## 2026-08-07：100 帧放大反馈环与分子名称内存压缩

### 可控放大输入与基线

将只读的 5 帧 `rp3.lammpstrj` 按原始 LAMMPS dump 格式重复 20 次，得到 100 帧、12,326 原子/帧、约 40 MiB 的压力输入。Step 1 只运行一次，再复用 molecule/origin 中间文件独立启动每个 Step 3 进程；因此不同 `nproc` 的比较不包含坐标检测成本。重复轨迹保持 5,615 个分子实例不变，但把 molecule range 增至 46,166、reaction event rows 增至 11,698，适合检查时间轴放大后的退化。

修改 worker 生命周期和名称表示之前的独立结果为：

| 指标 | `nproc=1` | `nproc=8` | `nproc=16` |
| ------------------- | --------: | ----------: | ----------: |
| molecule | 18.153 s | 10.458 s | 18.218 s |
| matrix | 0.185 s | 0.147 s | 0.169 s |
| route | 3.549 s | 3.885 s | 6.008 s |
| reaction | 1.859 s | 2.261 s | 3.516 s |
| Step 3 wall | 23.757 s | 16.767 s | 27.924 s |
| parent RSS peak | 425.9 MiB | 388.3 MiB | 317.4 MiB |
| 进程树 RSS 求和峰值 | 588.1 MiB | 2,084.7 MiB | 3,546.7 MiB |

本机只有 10 个逻辑 CPU，因此 16 进程结果只证明过度订阅会同时增加 wall time 和内存，不能据此限制 Linux/Slurm 节点的生产核数。三种并行度的 HDF5 语义指纹均为 `a98e80fea6cafe9f7d5ba55c68d50cac471d6e76ba1e0853f59cccb093dfb151`。

### 不在单个 SMILES 阶段内回收重型 worker

SMILES 路径原先继承 `run_mp` 的默认 `maxtasksperchild=1000`。单进程处理 5,615 个分子时，会连续创建约 6 个需要加载 RDKit/OpenBabel 的进程。molecule-only 直接对照为：

| 策略 | wall | 单个 child RSS peak |
| --------------------- | -------: | ------------------: |
| 每 1,000 项回收 | 21.318 s | 260.9 MiB |
| 本阶段保持同一 worker | 4.093 s | 260.1 MiB |

修改前的回归测试因调用未显式传入 `maxtasksperchild` 而失败；修改后 SMILES 路径与已有 route/reaction 路径一致，显式使用 `None`。未压缩名称表示时，两次完整 `nproc=1` Step 3 为 10.001 s 和 10.981 s，molecule 分别为 5.396 s 和 5.444 s；相对 23.757 s 基线，中位 Step 3 缩短约 55.8%。

为检查长期不回收是否造成 native 泄漏，将同一批结构扩为 112,300 个 SMILES 任务。单 worker RSS peak 为 261.9 MiB，与 5,615/28,075 任务时约 260–261 MiB 基本一致，没有随任务数爬升。

### 紧凑 molecule-to-species 名称表

压力反馈同时发现 `self.mname = np.array(mname)` 会生成定宽 Unicode 数组：本样本最长 SMILES 为 1,637 字符，即使绝大多数名称很短，每个分子槽仍按最长名称分配。现在改为：

1. 每个 molecule 只保存一个最小无符号 species ID；
1. 每个唯一 species 名称只保存一次；
1. route/reaction worker 分别 mmap ID 表和唯一名称表；
1. reaction matrix 直接在数值 species ID 上聚合，仅在输出唯一反应对时解码名称。

内部 builder 已保证 ID 合法，因此 worker attach mmap 时跳过重复的 `ids.max()` 全表扫描；否则每个 worker 都会读取整张 molecule-ID 表，重新引入 `nproc × molecule_count` 的无谓 I/O。对外直接构造名称表时仍保留完整范围校验。

使用 112,300 个 molecule、923,320 个 range 的同一 molecule-only 压力输入：

| 名称表示 | parent RSS peak | child RSS peak | wall | HDF5 write |
| ------------------------ | --------------: | -------------: | -------: | ---------: |
| 每 molecule 定宽 Unicode | 850.4 MiB | 261.9 MiB | 48.501 s | 8.251 s |
| species ID + 唯一名称表 | 272.2 MiB | 263.0 MiB | 52.487 s | 8.785 s |

主进程峰值减少约 578 MiB（68.0%），child 保持稳定；该极端重复压力下 wall 增加约 8.2%，是紧凑字典查找的成本。100 帧完整输出在修改前后逐字节比较：`.moname`、`.route`、`.reactionabcd` 的 SHA-256 分别保持 `44bfbb...a5b`、`df9e49...bd3`、`16ab42...a22`，HDF5 语义指纹仍为 `a98e80...b151`。

### 被否决的拓扑缓存实验

5,615 个 molecule 中安全的局部标号拓扑有 673 个，表面重复率为 88.0%。但按复杂度分组后，`atoms+bonds >= 16` 的 339 个结构没有重复；重复几乎全部来自 RDKit 本来就很便宜的小分子。加入有界 worker LRU 后，28,075 项压力 wall 从约 6 s 增至约 13 s，构造 Python 拓扑键的成本高于节省的 SMILES 计算。该实验代码和测试已回退，不进入最终实现。

### 当前验证边界

- `pytest -q tests/test_timedoutput.py`：43 passed；Step 3/Matrix 定向回归：11 passed、40 deselected；tools/detect：19 passed。
- 100 帧的 1/8 进程输出计数和语义完全一致。
- 排除两个 Tk GUI 名称的全量回归：117 passed、6 xfailed、3 xpassed、1 个 teardown error。error 仍是前文记录的 `miso=1` 首到顺序问题；当前运行选择 `[O]=[O]`，反应计数一致但文本 SHA `9156aa...` 不在历史白名单。单独重跑同一用例再次得到同一既有 error。
- 连续高负载下 macOS 单次 wall 波动明显，因此本节只把隔离 A/B 和数量级稳定的 RSS 变化作为结论；不把后续热状态单次结果用于宣称额外加速。
- 100 帧输入、约 1.0 GiB route/HDF5/压力输出、检测中间文件、基准脚本和 `__pycache__` 已清理；原始 `rp3.lammpstrj` 未修改，SHA-256 仍为 `c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。
- 生产 Slurm 全轨迹仍需验证 `Elapsed`、`AveCPU/AllocCPUS`、`MaxRSS`、四个 `step3_*_seconds`、HDF5 增长速度和 `.route` 完成时间。

## 2026-08-07：Route 聚合常驻内存压缩

### 剩余问题

route worker 已按 atom 返回二维 molecule-pair 数组，但主进程仍把每个被接受的
event 保留到 `allmoleculeroute`，随后 Matrix 阶段再次映射物种名称并执行
`np.unique()`。因此同一物种反应重复出现时，常驻结果仍为 `O(route event)`，并与
no-HMM/HMM 去重状态同时存在。

no-HMM 去重原先还使用
`dict[(left_molecule, right_molecule)] = earliest_atom_index`。atom index 只用于区分
“之前的 atom”和“当前 atom”，长期保存 tuple key 和 value 都不是必要的数据表示。

### 修改与语义约束

- `_printatomroute()` 在接受 molecule pair 后立即映射为紧凑 species ID pair，并累加
  `Counter`；Matrix 只对唯一 species pair 解码名称，不再保留第二份逐 event 数组。
- no-HMM 仍保留原有规则：同一个 atom 内重复出现的 molecule pair 全部计数，之后的
  atom 不再为同一 molecule pair 重复贡献。
- HMM 仍对 molecule pair 做全局一次性去重。
- 两种模式的去重 key 都编码为一个无溢出的 Python 整数
  `left * (molecule_count + 1) + right`；去掉 tuple key，no-HMM 也不再长期保存冗余的
  atom index。
- route 阶段日志输出唯一 molecule pair 数和最终 species-pair Counter 数，供全轨迹
  复跑判断剩余去重状态的实际规模。
- 相同物种之间的 route 与原实现一样不进入 reaction matrix；`.route` 文本仍逐 atom
  完整输出，没有被 Counter 聚合改变。

### 反馈环与局部基准

- no-HMM 回归输入包含同一 atom 的 100,000 个重复 `1 -> 2` event，以及下一 atom
  重复的 `1 -> 2` 和新的 `3 -> 4`。旧实现返回 100,001 行 ndarray；新实现只保留
  `Counter({(species_0, species_1): 100000, (species_2, species_3): 1})`。
- HMM 回归同时覆盖同一 atom 内重复和跨 atom 重复，确认每个 molecule pair 仍只计
  一次。
- 500,000 个唯一 molecule pair 的隔离 `tracemalloc`：旧
  `dict[tuple, atom_index]` 峰值为 77.204 MiB，紧凑整数 `set` 的长期状态为
  34.792 MiB，约降低 54.9%。该数字不包含 worker 返回数组和当前 atom 的短期集合。
- 200,000 个唯一 pair、分成 200 个 atom block 的聚合中位时间：旧 0.102576 s，
  新 0.098012 s，约快 4.4%。重复 1,000 个 pair 共 200 个 block 时：旧
  0.072243 s，新 0.063809 s，约快 11.7%。两种实现接受的 event 数和唯一 pair 数
  完全一致。

### 真实轨迹结果一致性

使用只读 `rp3.lammpstrj` 的 5 帧真实轨迹分别运行 `nproc=1` 和 `nproc=8`；最终只
保留 424 个 species pair。两种并行度以及打包去重前后的结果均为：

- HDF5 语义指纹：
  `3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`
- `.route` SHA-256：
  `f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`
- `.reactionabcd` SHA-256：
  `2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`
- `.reaction` SHA-256：
  `ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8`

原始 `.moname` 仍会继承既有 Step 1 unordered 首到顺序，不能作为跨并行度逐字节
比较依据；语义指纹已消除内部 molecule/species ID 顺序影响。

### 验证状态与边界

- `pytest -q tests/test_timedoutput.py`：44 passed。
- molecule/reaction/filter/parm2cmd 相关 `test_reacnetgen.py` 子集：12 passed，39
  deselected。
- Ruff lint、修改文件格式检查、`compileall` 和 `git diff --check`：通过。
- molecule-pair 去重状态仍为 `O(unique molecule pair)`；这是保持跨 atom 去重语义
  所必需的状态。本轮消除的是与 event 总数成正比的第二份 route 聚合结果及高开销
  tuple/dict 表示。
- 完整生产轨迹上的 Route Counter 大小、RSS、wall time 和 CPU 利用率仍需 Slurm
  复跑确认，不能用 5 帧样本外推加速倍数。

## 2026-08-07：默认 species timeline 的有界自适应聚合

### 新发现的默认路径瓶颈

`needprintspecies=True` 是默认配置。原 `_printspecies()` 在 Step 4 创建
`[Counter() for frame in timestep]`，再把每个 molecule 的 frame 数组执行
`.tolist()`，逐项更新 Python Counter。常驻对象随总帧数和活跃
`(frame, species)` 数增长；即使 Step 3 已完成，长轨迹仍可能在 `.species` 输出阶段
形成新的串行内存峰值。

确定性红灯直接调用真实 `_printspecies()`：4,097 帧、两个全程存在物种。旧实现创建
4,097 个 Counter，而目标是不保留逐帧 Python accumulator；命令在 2.7 秒内稳定失败，
报告 `4097 > 2`。后续真实轨迹差分还发现，仅按全局 species ID 输出会改变第 2–5 帧
的文本顺序，虽然计数相同；因此回归进一步锁定旧 Counter 的“该帧最小活跃 molecule
ID 决定物种首次插入顺序”语义。

### 方案比较

500 万 molecule-frame 更新的隔离聚合模型：

| 表示 | wall | Python peak | 临时数据 |
| ---------------------- | -----: | ----------: | -------: |
| 每帧 Counter | 5.44 s | 22.2 MiB | 0 |
| 稀疏 frame-block event | 0.65 s | 7.9 MiB | 15.0 MiB |
| 稠密数值 memmap | 0.30 s | 0.14 MiB | 0.5 MiB |

这说明固定选择稠密或稀疏都不合适：长寿命、高占用数据中稠密表更小更快；大量短寿命
且物种很多时，`species × frame` 表会浪费临时空间。

### 最终实现

- 每个数值 cell 保存紧凑 `count` 和 `priority`；`priority = molecule_count - molecule_index`，对同一物种/帧执行 `maximum.at` 后即可恢复旧实现的最小活跃
  molecule ID 顺序。
- 若完整稠密表不超过 64 MiB，第一遍读取时直接写 species-major memmap，不再为布局
  决策重复解压 molecule frame block。
- 大表第一遍只读取四段 molecule record 的第四段 frame payload，并记录实际
  molecule-frame 行数。结构、键和 bond level payload 使用通用选择性 reader 直接
  seek 跳过。
- 第一遍同时写紧凑 observation spool；上限为 256 MiB。布局确定后直接复用它填充
  稠密或稀疏表，避免大量小 frame block 第二次 LZ4/pickle 解压。超过上限时立即删除
  spool，并安全回退到第二遍顺序读取。
- 稠密临时空间为 `species × frame × (count + priority dtype)`；稀疏临时空间为
  `(frame, species_id)` event。大表按两者准确字节数选择更小布局。
- 稀疏事件先按 frame partition 写入；partition 内对 frame 做 stable sort，保持同一
  frame 的原 molecule 顺序。单帧计数改用 NumPy `unique/index/count` 并按首次出现位置
  恢复顺序，不再构造会按活跃物种数放大内存的 Python Counter；单 event 帧走常数开销
  快路径。
- observation spool 回填稀疏表时，先按 partition 稳定分组，再以向量化目标位置一次
  scatter；不再对每个 partition 重扫整批 observation，最坏复杂度由
  `O(partition × chunk rows)` 降为一次排序/线性 scatter。
- 稀疏 partition 会按实际 event 数自动缩小；stable sort、字段副本、NumPy 去重数组和
  工作区采用保守估算，正常 block 不超过 64 MiB。单帧自身超过预算时，该帧是不可再拆
  的业务下界。
- 日志输出 molecule-frame 行数、species 数、dense/sparse 模式、observation spool
  是否复用、最终/峰值临时空间及最大 aggregation block；所有 memmap/spool 在正常、
  异常和容量回退路径均删除。

继续审查时补了两条结构性红灯：强制 sparse 模式后把模块级 Counter 替换为失败函数，
旧代码两组用例均在单帧聚合处失败；4,096 个 frame partition 的 observation 回填则
记录到 4,096 次 `count_nonzero`/整批布尔扫描。修复后前者完全不调用 Python Counter，
后者至多一次稳定分组且测试记录的整批重扫调用数为 0。

### 函数级 A/B

高占用输入：250 molecules、5,000 frames、25 species，共 1,250,000
molecule-frame 行。

- 旧：1.345 s，`tracemalloc` peak 5.944 MiB。
- 新：连续三次为 0.531/0.538/0.543 s，中位 0.538 s；peak 约 0.42 MiB。
- wall 约缩短 60%，Python peak 约降低 93%。
- 两份 898,890-byte `.species` SHA-256 均为
  `fd24d6eb1a54bb5c18ae7c2819e073699f14dd418b15d82f905d6a49149bf08e`。

极稀疏输入：10,000 frames、10,000 molecules/unique species，每个 molecule 仅存在
一帧。

- 旧约 0.22 s、peak 3.27 MiB。
- 初版稀疏实现为 6.05 s、peak 128.18 MiB；原因是第二遍解压 10,000 个小 block，且
  `np.fromfile(count=1M)` 预分配过大。
- observation spool、65,536-row 读取上限、单 frame 快路径、NumPy 聚合和向量化
  partition scatter 后约 0.61 s、peak 1.09 MiB；最终/峰值临时文件约
  0.038/0.095 MiB。包含排序/去重临时数组后的保守最大 aggregation block 估算约
  0.155 MiB。
- 该极端稀疏场景相对旧实现（本轮约 0.24 s）仍有约 0.37 s 固定管理开销，但 Python
  peak 约降低 67%，且不再随
  总帧数保留 Counter。输出 SHA-256 均为
  `1ab9401e44cdcd417d94b7ab2d685a61af59d7339c3a0778f56125b76b4c1330`。

### 真实轨迹差分与验证

同一次 8 进程 `rp3.lammpstrj` PATH 结果上，旧 Counter 参考与新实现分别处理 5 帧、
5,615 molecules、413 species、22,103 molecule-frame 行：

- 两份 `.species` 均为 58,008 bytes；
- SHA-256 均为
  `1f7f07e2770310c573c7475821530a1dc161084459394d95c4ae706853ee04cc`；
- 每帧计数和旧插入顺序逐字节一致；
- 小样本旧/新分别约 0.070/0.232 s，固定 memmap/排序成本占主导，不能用它外推长轨迹
  加速。

另以固定随机种子运行 40 组差分，覆盖空 timeline、重复/无序 frame、同物种多
molecule、dense、sparse、observation spool 复用和容量回退；每组都与旧 Counter
参考文本逐字节一致。

最终检查：`tests/test_timedoutput.py` 49 passed；Step 3/4 交叉子集 12 passed、39
deselected；tools/detect 19 passed。Ruff、`compileall`、mdformat 和
`git diff --check` 通过。生产全轨迹仍需记录 Step 4 wall、species 日志中的布局/临时
空间、节点临时盘峰值和 MaxRSS。

## 2026-08-07：Step 3 worker IPC 与单进程零 IPC 路径

### 反馈环与定位

从真实 `rp3.lammpstrj` 中固定提取前 1,024 个 molecule record，直接运行
`_CollectSMILESPaths._printmoleculename()`。性能门限设为 1.8 秒；修改前三次稳定为
2.278/2.288/2.259 秒并全部红灯。相同 record 直接调用实际 worker 函数仅约 0.493 秒，
parent profiler 的主要时间位于 multiprocessing `send/recv`、pipe `read/write` 和
`Pool.join`，而 LZ4/pickle 解压合计仅约 0.02 秒。

依次验证的假设如下：

- 仅移除 worker 返回值中的 atoms/bonds 会减少 IPC 字节，但在 `nproc=1` 下仍需创建
  Pool，故不会单独消除 wall 瓶颈；实测仍约 2.34 秒。
- `nproc=1` 直接在调用进程执行应接近纯计算时间；临时注入实测约 0.62 秒，确认命中。
- 把 Pool `chunksize` 从 1 提到 4/16/64，在单进程样本只由 2.29 秒降到约
  2.06–2.13 秒；8 进程下过大 chunk 还会造成 straggler，不能作为主要修复。
- 把完整 collector 作为 worker initializer 参数会在 spawn 时重复序列化大对象，实测
  约 15.1 秒；只传 `atomname + atomtype` 并在 worker 内构造最小转换器后才有收益。

### 最终实现

- `run_mp(nproc=1)` 在没有 worker initializer 时使用同步 Pool-compatible iterator，
  不创建进程、队列、pipe 或 semaphore。仍保留 `nlines`、`interval`、`extra`、进度条
  以及 disk-ordered 精确总数校验语义；带 initializer 的调用继续使用真实 Pool，避免
  把 worker 专用 memmap/global 生命周期泄漏到主进程。
- `run_mp` 在调用方提供整数 `total` 时把实际 worker 数限制为
  `min(requested_nproc, max(1, total))`；例如 4 个 transition 不再启动 8/64 个进程。
  全轨迹任务数大于申请核数时不改变并行度。
- 单进程 SMILES stage 进一步直接消费主进程 record，不进入 `run_mp`，也不再为
  atoms/bonds 打开第二遍读取路径；这对大于 page cache 的临时 molecule 文件避免一次
  完整结构重读。
- `nproc>1` 的 SMILES stage 改用模块级入口；每个 worker 只在初始化时接收紧凑
  `atomname` 和 `atomtype`，构造最小转换器，不再在每个 task 中 pickle 绑定的完整
  collector。
- SMILES worker 只返回 species name。主进程从第二个顺序文件句柄读取 atoms/bonds，
  需要 timeline 时同时读取 frame block；fallback VF2、miso、`.moname` 和 HDF5 仍使用
  同一 molecule record。1,024 个真实结果的压缩 worker payload 从 308,473 bytes 降到
  115,836 bytes，减少 62.4%；最大单结果从 5,270 降到 987 bytes。
- worker 不回收策略保持不变，避免重新加载 RDKit/OpenBabel；磁盘有序恢复仍保证父进程
  按 molecule 输入顺序处理。

### 真实性能与语义验证

- 1,024-record 原红灯在正式串行路径下连续为 0.704/0.714/0.705 秒，均低于 1.8 秒，
  相对红灯中位数约缩短 69%。
- 完整 5,615-record、413-species molecule-only：`nproc=1` 无 profiler 约
  0.61–0.66 秒；
  8 进程的绑定 collector 基线约 7.05 秒，最小 initializer 正式实现三次为
  5.287/5.413/5.560 秒，中位约缩短 23%。
- 该便宜小样本的 plain scaling 为 1/2/4/8/16 进程约
  0.612/1.911/2.617/4.057/8.038 秒，明确显示“申请更多核”并不等于更快。它不能替代
  全轨迹选型；复杂 molecule 或更多任务可能重新进入多进程收益区间。
- 同一份 Step 1/HMM 中间数据两次完整运行 molecule、matrix、route、reaction 和 HDF5：
  `nproc=1` wall 2.24–2.62 秒（各阶段约 0.79–0.86/0.08–0.10/0.87–1.03/
  0.49–0.63 秒），`nproc=8` wall 6.12–7.99 秒（3.24–3.77/0.08–0.11/
  1.60–2.35/1.07–1.75 秒）。该结果用于证明小样本过度并行和功能一致，不与不同
  缓存/环境下的历史 wall 直接作百分比 A/B。
- 1/8 进程的 HDF5 语义指纹均为
  `3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；
  `.route`、`.reactionabcd` 和排序 `.moname` SHA-256 分别保持
  `f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
  `2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
  `2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。
- 新增结构回归锁定：单进程不得创建 Pool；serial `extra` 输入整形和 disk-ordered 错误
  校验保持；worker 数不得超过声明任务数；单进程 SMILES 只读一遍 record；多进程必须
  使用最小 initializer；worker 返回值必须只有 name。

当前 `tests/test_timedoutput.py` 为 52 passed。生产全轨迹仍需比较 8/16/32/64 核的
Step 3 分阶段 wall、总 core-hours、MaxRSS、spool/临时盘和输出 manifest；不能从 5 帧
样本直接把推荐核数固定为 1。

## 2026-08-07：Code review 最终边界闭环

### 复审残留问题

- reaction type 原先先把反应物/产物字符串加入缓冲，再检查字节阈值。两个分别小于
  64 MiB 的长名称仍可能让合并批次超过阈值；数值 `total_count` 列也未计入
  `type_bytes`。
- 5,000 个逆序 transition 跨 write batch 的 block-index 恢复只保留在手工验证日志
  中，没有自动回归保护。

### 红灯与修复

- 将 `_WRITE_BATCH_BYTES` 缩小为 100 字节，两组 reaction type 各包含 52 字节 UTF-8
  名称和 8 字节 `total_count`。修复前测试只观察到一个 104 字节 type batch，稳定越过
  上限。
- 新 reaction type 现在先计算字符串和固定数值列的实际 payload；若与待写 event/type
  合并后会越界，则先 flush。单个不可拆分的超大 reaction type 仍允许独立写出，和
  molecule definition 的边界语义一致。
- 新增 5,000 个 transition 的逆序提交测试。数据跨两个 event write batch 后，
  `block_start` 为严格逆序的 `4999..0`、`block_length` 全为 1；公开 reader 仍按
  transition `0..4999` 恢复事件。

### 最终检查

- `pytest -q tests/test_timedoutput.py`：54 passed。
- molecule/Step 3/timestep/parm2cmd 子集：11 passed、40 deselected。
- tools/detect：19 passed。
- Ruff lint/format、mdformat、`compileall` 和 `git diff --check`：通过。
- 两轴复审未发现 serial/no-IPC、最小 SMILES initializer 或 worker-count cap 的新
  P0/P1/P2 问题。剩余建议仅为把测试文件按子系统拆分、进一步把 batch 记账方法内聚到
  dataclass；不影响本轮功能和性能正确性，未扩大重构范围。
- 完整生产 Slurm 轨迹仍是最终验收项。

## 2026-08-07：Route worker 重复 pair 预聚合

### 剩余串行热点与红灯

route worker 原先把一个 atom 的每次 molecule 变化都作为二维 pair 行返回。主进程虽然
最终只保留唯一 molecule-pair 去重状态和 species-pair Counter，但仍逐行执行 Python
循环。现有 no-HMM 回归中的 100,000 个相同 `1 -> 2` 事件最终只产生一个 key，却仍在
父进程遍历 100,000 次；单项 pytest call 约 0.04 秒。

新增结构红灯直接初始化真实 route worker：4,097 帧在 molecule 1/2 间交替，原 worker
返回 4,096 行，无法满足“2 个唯一 pair + `[2048, 2048]` 次数”的结果契约。

### 假设验证

- 100 万个相同 pair 的隔离父进程循环中，原始逐行路径中位约 0.164 秒；消费预聚合的
  单 pair 约 1 微秒。该数字只衡量父进程聚合，不代表完整 route wall。
- 同一短文本结果经现有 LZ4 暂存格式序列化后，原始 100 万行 pair 为 68,129 bytes，
  唯一 pair + count 为 285 bytes。实际 `.route` 文本仍必须完整输出，因此文本很长时
  不会获得相同比例的总 IPC 缩减。
- 8 worker、32 个 atom task、每项 100,000 个重复事件的真实
  `run_mp + disk_ordered` 交替 A/B 中，三组 old/compact wall 分别为
  1.732/1.389、2.007/1.672、2.002/1.555 秒，缩短约 17%–22%。
- 100,000 个全唯一 pair 的抽样 fast path 只检查最多 1,024 行，直接调用中位约
  0.10 ms/atom；多进程 wall 波动在约 3% 范围，不能据此宣称全唯一数据加速。

### 有界实现与语义

- 仅当 route 至少有 256 行且均匀抽样发现重复 pair 时启用预聚合；全唯一路径返回原
  ndarray，不复制或重排完整结果。
- 预聚合按最多 65,536 行执行局部 `np.unique`，再合并局部计数；不会对完整 atom
  timeline 一次性排序。
- 唯一 pair 超过 65,536 时放弃预聚合并返回原数组，避免为近似全唯一输入建立无界
  Python 字典；70,000 行、抽样重复但整体近似唯一的回归锁定该回退路径。
- 100 万、500 万和 1,000 万重复事件的独立进程测量中，compaction 相对已分配原数组的
  额外 MaxRSS 分别约 1.1、4.5 和 4.4 MiB；500 万到 1,000 万未继续线性增长。
- no-HMM 对“首次出现该 molecule pair 的 atom”仍累计该 atom 内的全部重复次数；后续
  atom 的相同 pair 仍跳过。HMM 仍让每个 molecule pair 在全部 atom 中只贡献一次。
  `.route` 文本未压缩或重排。
- route 阶段新增日志：原始 event rows、worker 返回的 pair rows 和触发预聚合的 atom
  数；生产复跑可据此判断该优化是否命中，而不从 CPU 利用率反推。

### 真实轨迹与检查

只读 `rp3.lammpstrj` 以最终代码全流程 `nproc=8` 运行完成：Step 3
molecule/matrix/route/reaction 分别为 4.022/0.172/1.994/0.593 秒，timed-output write
为 0.242 秒。该短轨迹报告 `12,751 event rows -> 12,751 pair rows, 0 compacted atoms`，符合 256 行启用阈值。输出计数仍为 5,615 molecules、6,076 ranges、383
reaction types、469 event rows；语义指纹保持：

```text
3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb
```

`.route`、`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持：

```text
f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3
2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56
ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8
2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6
```

最终回归为 `tests/test_timedoutput.py` 58 passed；molecule/Step 3/timestep/parm2cmd
子集 11 passed、40 deselected；tools/detect 19 passed。Ruff lint/format、mdformat、
`compileall`、`git diff --check` 通过。生产全轨迹仍需确认实际每 atom 的
`route rows / unique pairs` 分布、route-stage CPU 利用率和整体 wall；短轨迹每个 atom
少于 256 个变化，不触发预聚合，只用于结果一致性验证。

## 2026-08-07：Route 文本写缓冲按字节限界

### 问题与红灯

`.route` 原先使用默认 `WriteBuffer(linenumber=1200)`。缓冲只限制行数，而单个 atom
的 route 行会随轨迹长度增长；因此父进程可同时保留约 1,200 条长字符串，峰值内存由
`line count × route line bytes` 决定，并没有固定字节边界。

最小红灯把行数阈值设为 1,000、连续追加两个各 5 bytes 的 payload，并要求 8-byte
预算下在第二项加入前 flush；原构造器不支持 `byte_limit`，稳定失败。另一个真实 worker
红灯要求 route 文本跨进程时已经编码为 bytes，避免父进程文本编码和 text-I/O 缓冲。

### 实现

- `WriteBuffer` 新增可选 `byte_limit`，按 UTF-8 payload 加 separator 的实际编码大小
  记账；下一项会越界时先 flush。
- 单个不可拆分项本身超过预算时允许独立写出，但加入后立即 flush，不与其他项目共同
  驻留。`maximum_buffer_bytes` 保存运行期最大编码 batch，可用于生产日志。
- `.route` worker 返回 UTF-8 bytes；父进程用二进制 `WriteBuffer` 写出，行格式和换行
  不变。
- route 缓冲目标为 8 MiB，同时仍保留 1,200 行上限。运行日志报告实际最大 batch，
  并明确上界为“8 MiB 加一个不可拆分 route 行”。

### A/B 与被否定的方案

8 worker、32 个 route task、每项 200,000 个事件，使用相同生成、IPC、disk-ordered
恢复和 `/dev/null` sink，三组交替 old/bounded 结果为：

| 指标 | old 中位 | bounded 中位 | 变化 |
| ------------------ | -------: | -----------: | ---: |
| parent MaxRSS 增量 | 275 MiB | 131 MiB | -52% |
| wall | 1.105 s | 1.041 s | -6% |

该测试隔离的是父进程缓冲和 IPC，不包含真实文件系统写盘；RSS 数字也包含 Pool/result queue
的固定开销。

两个候选没有进入正式代码：

- worker 侧 LZ4 可把 1.09 MiB 合成 route 缩到约 0.42 MiB，但 32-task A/B 中 wall
  增加约 3%，解压峰值也没有降低 parent MaxRSS。
- 把 route `max_inflight` 从 16 降到 4 可把该压力下 parent RSS 再降低约 25%，但 wall
  增加约 10%；当前保留 `2*nproc`，等待生产 I/O 与任务时长分布再决定。

### 结果一致性

最终代码重新运行只读 `rp3.lammpstrj` 全流程；`.route` 为 12,326 行、5,494,972
bytes，SHA-256 仍为：

```text
f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3
```

HDF5 语义指纹以及 `.reactionabcd`、`.reaction`、排序 `.moname` 哈希也保持前节基线。
短轨迹 Step 3 molecule/matrix/route/reaction 为 3.568/0.122/1.862/0.542 秒；只作功能
回归，不与不同热状态的历史单次 wall 计算加速比。

最终回归：`tests/test_timedoutput.py` 59 passed；molecule/Step
3/timestep/parm2cmd 子集 11 passed、40 deselected；tools/detect 19 passed。

## 2026-08-07：Reaction 变更原子索引分块扫描

### 问题与红灯

每个 reaction worker 原先先执行
`np.flatnonzero(atomeach_before != atomeach_after)`。这会同时创建与完整原子数等长的
布尔比较数组，以及最坏情况下每个元素 8 bytes 的 `intp` 索引；64 个 transition
worker 并行时，该临时分配会按 worker 数叠加。它不影响反应正确性，但会放大 Step 3
峰值内存并增加内存带宽压力。

新增红灯构造 100,000 个全部变化的原子，追踪 `np.flatnonzero` 的输入长度，并要求单次
扫描不超过 65,536 行。旧实现只调用一次且输入 100,000 行，稳定失败。

### 实现与语义

- `_calculate_transition_reactions()` 现在按最多 65,536 个原子切片，逐块比较并提取局部
  change index；局部索引立即转换回全局 atom index。
- 块按原始数组顺序扫描，块内 `flatnonzero` 也保持升序，因此 `reactdict` 的插入顺序与
  旧全向量实现完全一致。
- 这只限制布尔比较和 `intp` change-index 临时数组；`dps_reaction()` 所需的
  `reactdict` 仍与实际变化原子数相关，本轮没有把它误称为完整反应阶段的常数内存。
- 固定随机输入覆盖跨块边界、稀疏变化和前后 conflict marker；把块大小分别设为全长与
  31 行时，最终 reaction `Counter` 完全相同。

### 独立进程 A/B

每组用全新进程运行三次；输入为 100 万个原子，对照组把扫描块设为 `N + 1`，等价于
旧全向量实现。表中为中位数，MaxRSS 增量从输入数组构造完成后开始计算：

| 变化密度 | 指标 | 旧全向量 | 65,536 行分块 | 变化 |
| --------------- | --------------- | -------: | -------------: | -----------------: |
| 约 1,000/百万 | MaxRSS 增量 | 1.06 MiB | 0.11 MiB | -0.95 MiB（-89.7%） |
| 100% | MaxRSS 增量 | 69.59 MiB | 64.39 MiB | -5.20 MiB（-7.5%） |
| 100% | reaction wall | 0.367 s | 0.374 s | +2.1% |

稀疏用例本身只有亚毫秒量级，wall 易受调度噪声影响，因此只用于内存上界验证，不报告
加速比。稠密最坏用例显示分块带来约 2.1% 的小循环开销；按 64 个同时处于该阶段的
worker 粗略外推，100 万原子全变化时仅 change-index 一项可少约 333 MiB 聚合峰值，
但生产实际节省必须以 Slurm MaxRSS 为准。

### 真实轨迹与回归

最终代码用只读 `rp3.lammpstrj` 完整运行 `nproc=8`：Step 3
molecule/matrix/route/reaction 为 3.294/0.101/1.716/0.519 秒，Step 3 核心计算
5.481 秒。HDF5 语义指纹仍为：

```text
3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb
```

`.route`、`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍逐项等于既有基线。
完整 `tests/test_timedoutput.py` 为 61 passed；molecule/timestep/parm2cmd 子集 11
passed、40 deselected；tools/detect 19 passed。真实运行和基准的临时输出已删除，原始
轨迹未修改，SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

Ruff lint/format、mdformat、`compileall` 和 `git diff --check` 均通过。

该修复降低 reaction worker 的瞬时内存，但不会单独消除生产全轨迹中主进程写 HDF5
或 route 聚合造成的低 CPU 利用率；仍需在 Slurm 上记录各阶段 wall、MaxRSS、CPU
利用率和临时盘，验证所有已实施优化的综合效果。

## 2026-08-07：Molecule range HDF5 大批次与零拷贝校验

### 生产规模信号与红灯

既有 5% Slurm 压力作业包含 49,047,393 条 `molecule_ranges`。此前 molecule
definition、reaction event 和 molecule range 共用 4,096 行应用批次；仅按 range
行数计算就会触发约 11,975 轮 flush，每轮分别 resize/write 三个 range dataset。
这与“worker 已完成但主进程长时间整理 range 和写 HDF5”的低 CPU 症状一致。

确定性红灯通过真实 `TimedOutputStore.add_molecule()` 写入 16,385 条离散 range，要求
在 64 MiB 字节预算内合并为一批。旧实现稳定产生 5 批，测试连续三次均在约 0.9 秒内
报告 `5 != 1`。

### 单变量选型

独立进程使用 100 万条 range，并按生产 `_itermoleculeranges()` 的 4,096 行 block
流入。保持 HDF5 chunk 为 4,096 行，交替运行五组：

| 指标 | 4,096 行应用批次 | 65,536 行 range 批次 | 变化 |
| --- | ---: | ---: | ---: |
| HDF5 flush 批数 | 245 | 16 | -93.5% |
| write wall 中位 | 0.0612 s | 0.0240 s | -60.8% |
| MaxRSS 增量中位 | 21.8 MiB | 18.9 MiB | -13.6% |
| 最终文件 | 约 0.737 MiB | 约 0.737 MiB | 无实质变化 |

同一输入的 NumPy range 生成约 0.0035 秒，LZ4 解码约 0.004 秒；在这个反馈环中，旧
HDF5 小批次写入明显更慢。继续把应用批次增至 262,144 或 1,048,576 行没有稳定额外
收益；把 HDF5 chunk 增至 65,536/262,144 行反而增加 wall、基础 RSS 和文件大小，均未
采用。

这里的本地绝对时间不能外推共享集群文件系统；可信结论是调用次数、相对 wall 和有界
内存。按 5% 作业行数估算，range 驱动的理论 flush 数由约 11,975 降至约 749，另加
molecule definition 和 64 MiB 字节边界触发的批次。

### 最终实现

- 新增独立的 65,536 行 molecule-range 批次上限；molecule definition、reaction
  event 和 HDF5 chunk 继续保持 4,096 行，不扩大其他对象的常驻批次。
- 64 MiB 总字节预算仍先于下一行执行 preflight flush；range 数值列最坏仍受行数和
  字节数双重限制。
- Step 1 生成的 frame/range 本来就是 `uint64`。写出校验不再先强制复制为 `int64`、
  验证后再复制回 `uint64`；合法 `uint64` block 直接复用，signed、浮点兼容输入仍先
  规范化。
- 新回归确认合法 `uint64` block 在 batching 前共享内存，并继续拒绝负 start、负
  end、start 大于 end 和超出 frame_count 的 unsigned 值。

### 真实轨迹与回归

只读 `rp3.lammpstrj` 使用最终代码完整运行 `nproc=8`：Step 3
molecule/matrix/route/reaction 为 3.353/0.152/1.744/0.516 秒，timed-output write
为 0.160 秒。短轨迹只有 6,076 条 range，仍由 5,615 个 molecule definition 的
4,096 行上限形成 2 个批次，因此只用于功能一致性，不用于本优化的加速比。

HDF5 语义指纹保持
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` 哈希逐项保持既有基线。最终
`tests/test_timedoutput.py` 为 67 passed；molecule/timestep/parm2cmd 子集 11
passed、40 deselected；tools/detect 19 passed。基准脚本和真实输出均已删除，原始轨迹
未修改。

Ruff lint/format、`compileall` 和 `git diff --check` 通过。生产全轨迹仍需对比新旧
`molecule_write_batches`、`timed_output_write_seconds`、molecule stage wall、MaxRSS
和 HDF5 吞吐；这项修复只消除已证实的小批次写放大，不把整个 Step 3 的 48 小时都归因
于 HDF5。

## 2026-08-07：主进程剖析、SMILES spool 轻量编码与生产指标

### 真实主进程 cProfile

对只读 5 帧 `rp3.lammpstrj` 的 8 进程全流程运行主进程 cProfile。带 profiler 时
molecule stage 为 3.784 秒；父进程可归属的累计时间包括：

| 路径 | 累计时间 |
| --- | ---: |
| `TimedOutputStore.add_molecule()` | 0.292 s |
| `_itermoleculeranges()` | 0.104 s |
| `_getatomsandbonds()` | 0.117 s |
| `.moname` 格式化 | 0.092 s |
| ordered spool put + pop | 0.236 s |

小轨迹上父进程本地整理合计不到约 0.7 秒，主要时间仍在等待 SMILES worker；因此没有
重新引入此前已被真实 A/B 否定的后台 collector，也没有仅凭生产 CPU 利用率推断小样本
需要写线程。生产 4,904 万 range 与小轨迹的 6,076 range 不同，仍必须依靠新增 writer
指标区分。

### 被否定的 reaction 转置候选

对 50,000 原子 × 2,048 帧的约 293 MiB 热 memmap，逐 transition 调用真实
`_calculate_transition_reactions()`：

- 当前 C-order 列扫描约 0.358 秒，额外 MaxRSS 近似为零；
- 16–512 帧显式转置块均为约 0.596–0.654 秒；
- 最快的 128 帧块仍慢约 66%，并增加约 68 MiB MaxRSS；最省内存的 16 帧块也增加约
  20 MiB。

这不能排除超出页缓存矩阵在集群上的 page-fault 问题，但证明“增加第二布局或 worker
转置块”不是当前可无条件采用的优化。候选代码和基准文件均已删除。

### SMILES 字符串 spool 专用编码

主进程剖析显示，SMILES worker 已只返回短名称，但 `_DiskOrderedResultSpool` 仍逐项执行
pickle + LZ4。确定性红灯把通用 codec 替换为失败函数；旧实现在写入第一个字符串时
稳定失败。

10 万个约 17-byte 的 SMILES 形态字符串，独立进程交替前后各五次：

| 指标 | 通用 pickle + LZ4 | tagged UTF-8 | 变化 |
| --- | ---: | ---: | ---: |
| put + pop 中位 | 0.728 s | 0.592 s | -18.7% |
| spool bytes | 11.789 MB | 1.789 MB | -84.8% |
| MaxRSS 增量中位 | 1.77 MiB | 1.63 MiB | -7.9% |

最终编码用一个 byte 类型标记直接保存不超过 4 KiB 的 `str`、`bytes` 和 `None`；超长
字符串以及数组/tuple 继续走通用压缩，限制异常单项对临时盘的放大。回归同时覆盖 UTF-8
surrogate、bytes、None、超长重复字符串和原有 NumPy array round-trip。

真实 5 帧运行中，SMILES ordered spool 总写入从此前约 0.31–0.55 MiB 降到 0.035
MiB；route tuple 继续使用通用 codec，输出语义未改变。该收益只作用于存在乱序 backlog
的结果，不外推为整个 molecule stage 的加速比。

### 可直接用于 Slurm 的 writer 指标

`TimedOutputStore` 现在分别记录：

- `timed_output_molecule_write_seconds`；
- `timed_output_reaction_write_seconds`；
- `maximum_molecule_batch_definition_count`；
- `maximum_molecule_batch_range_count`；
- `maximum_molecule_batch_bytes`；
- 实际 `molecule_write_batches`、range 行上限和 byte 上限。

molecule stage 完成时同步写日志，不必等最终 HDF5 发布后才能定位。真实小轨迹报告：

```text
6076 ranges in 2 batches, 0.166s accounted,
max 4096 definitions / 4431 ranges / 0.504 MiB
```

HDF5 中 molecule/reaction write 分别约 0.166/0.009 秒，总和与既有
`timed_output_write_seconds` 一致。完整 `tests/test_timedoutput.py` 为 69 passed；
molecule/timestep/parm2cmd 子集 11 passed、40 deselected；tools/detect 19 passed。
HDF5 语义指纹及 `.route`、`.reactionabcd`、`.reaction`、排序 `.moname` 哈希保持既有
基线。

Ruff lint/format、mdformat、`compileall` 和 `git diff --check` 通过；profile、基准和真实
输出均已清理。
生产复跑后应直接比较 molecule stage wall 与
`timed_output_molecule_write_seconds`：若两者接近，剩余瓶颈仍在 range/HDF5；若相差
很大，则应转向 SMILES worker、ordered 等待或输入解码，而不是继续盲目扩大 HDF5
batch。

## 2026-08-07：Ordered spool 索引取消全量预清零

### 红灯与根因

`_DiskOrderedResultSpool` 会为全部输入任务建立 `(offset, length)` 的 `uint64` memmap
索引。新文件用 `mode="w+"` 建立后本来就保证未写区域读为零，但旧实现仍执行
`index[:] = 0`。这会在任何 worker 结果到达前触碰每个索引页，使父进程初始化内存和
close/flush 成本都与任务总数线性增长；1,000 万任务对应 160 MB 逻辑索引。

确定性回归用真实 memmap 的轻量代理记录整切片写入，并同时验证最末索引初始不存在、
随后 `put/has/pop` 正常。旧代码稳定捕获一次全切片清零，修复后不再发生。没有使用
`st_blocks` 作为回归信号，因为 APFS 对延迟分配和全零页的物理块统计随临时目录而异。

### 独立进程基准

对 1,000 万任务的 spool 只执行初始化和关闭，修复前后各交替运行五个独立进程：

| 指标 | 全量预清零 | 使用新文件隐式零页 | 变化 |
| --- | ---: | ---: | ---: |
| 初始化中位 | 0.0258 s | 0.00435 s | -83.2% |
| close/flush 中位 | 0.0567 s | 0.00056 s | -99.0% |
| MaxRSS 增量中位 | 约 159.5 MB | 约 0.25 MB | -99.8% |

索引的逻辑地址空间和随机访问语义不变；只有真正收到乱序结果的索引页才会被触碰。
最坏情况下若全部任务都形成 backlog，索引仍可增长到原有上界，因此生产端仍需记录
ordered spool 的 maximum pending 和临时盘峰值。

### 真实轨迹与回归

只读 `rp3.lammpstrj` 使用 `nproc=8`、molecule timeline 和 reaction event 完整运行：
Step 3 molecule/matrix/route/reaction 为 3.403/0.149/1.729/0.550 秒，core computation
为 5.664 秒。SMILES spool 写入 0.035 MiB，最大 pending 为 3,118；route spool 写入
1.080 MiB，最大 pending 为 11。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` 哈希逐项保持既有基线。完整
`tests/test_timedoutput.py` 为 70 passed。原始轨迹 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

### 生产 range 行数压力环

为避免只从 100 万行外推，另以生产记录中的 `49,047,393` 条 range 直接运行真实
`TimedOutputStore`：输入仍按生产路径的 4,096 行 block 流入，frame 值在 0–65,535
间变化，三个 range dataset 保持 shuffle + gzip level 1。结果为 749 个 molecule
batch，总 wall 约 2.05 秒、writer 计时约 1.99 秒；HDF5 约 51 MB，热身后的 MaxRSS
增量约 79.7 MB，最大应用 batch 仍为 65,536 行/约 1.50 MiB。

cProfile 中约 1.62 秒位于 h5py dataset `__setitem__`，合成 range block 生成累计约
0.08 秒。该数据证明当前应用层 range batching 在本地 APFS/SSD 上不是小时级路径，
但数据可压缩性、页缓存和共享文件系统均不同于计算集群，不能把 2 秒外推为生产 wall。
生产复跑若仍慢，应先用新增 writer 指标区分文件系统写入、SMILES/结构识别和 ordered
等待，再决定是否调整 HDF5 布局。

## 2026-08-07：HDF5 顺序写缓存降至 1 MiB

### 被否定的 SMILES chunksize 候选

先复用真实 `rp3.lammpstrj` 的 5,615 个 molecule 结构记录，固定 `nproc=8`，只改变
SMILES `chunksize` 和与之匹配的 in-flight 窗口。`chunksize=1/2/4/8/16` 的单轮 wall
分别约为 4.49/4.39/4.49/4.54/4.67 秒，`.moname` SHA-256 全部为
`4f29c4226981eeb9fbdc8ccba981027d4a2818924ce2f1cbc0e5378b179cec0b`。
扩大 `chunksize=1` 的窗口也只有噪声级差异。

因此当前真实任务粒度下 IPC batching 不是剩余热点；较大的 chunk 还可能把多个复杂
结构绑定到同一 worker，降低负载均衡。该候选未进入生产代码，临时结构记录和脚本均已
删除。

### 默认 128 MiB cache 的红灯

`TimedOutputStore` 只顺序 append，构建期间不会重新读取已经写出的 chunk；但原默认值
仍给 HDF5 raw-data chunk cache 配置 128 MiB。以真实生产记录中的 49,047,393 条变化
range、4,096 行输入 block、65,536 行应用 batch 交替比较 1 与 128 MiB，各五个独立
进程：

| 指标 | 128 MiB | 1 MiB | 变化 |
| --- | ---: | ---: | ---: |
| wall 中位 | 2.225 s | 2.090 s | -6.1% |
| MaxRSS 增量中位 | 126.7 MiB | 52.7 MiB | -58.4% |
| HDF5 大小 | 约 51.031 MB | 约 51.033 MB | +约 2 KiB |

4/16/64 MiB 的单轮 MaxRSS 增量分别约 75/118/131 MiB，未显示大 cache 的吞吐收益。
1 MiB 可同时容纳约 32 个 4,096-row `uint64` chunk，超过单次 65,536-row range 写入的
16 个 chunk；顺序 writer 不需要继续保留旧 chunk。

### 最终修改与真实输出

- Python API、CLI 和 `parm2cmd()` 的默认 `timedoutputcachemib` 从 128 改为 1；显式
  `--timed-output-cache-mib` 覆盖保持兼容。
- HDF5 根属性新增 `timed_output_cache_mib`，生产结果可直接追溯实际 cache 配置。
- 回归同时锁定 parser、`ReacNetGenerator` 和命令序列的默认值，避免三处默认再次漂移。

默认 1 MiB 下，只读 `rp3.lammpstrj` 的 8 进程完整运行 Step 3
molecule/matrix/route/reaction 为 3.378/0.105/1.764/0.532 秒，core computation 为
5.628 秒。HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` 哈希逐项保持基线。完整
`tests/test_timedoutput.py` 为 71 passed。

## 2026-08-07：短 route 时间轴自适应并行

### 反馈环与假设

真实 5 帧轨迹有 12,326 个 route 原子任务，但每个任务只扫描 5 个值，总工作量只有
61,630 atom-frame values。原实现仍固定启动请求的 8 个 worker，并为严格有序输出建立
进程池、队列和磁盘 spool。这里验证了三个假设：

1. 短时间轴的主要成本是进程池和 IPC 固定开销，而不是 route 数值扫描；
1. 把 worker 数降为 1 仍会保留一个无收益的子进程，直接在父进程执行才能消除固定
   开销；
1. 长时间轴必须保留多进程，不能无条件串行化 route。

用真实 `_AtomFrameStore` 和 `_printatomroute()` 构造 12,326 原子 × 5 帧的确定性反馈
环，强制旧 8-worker 路径与自适应父进程路径交替各运行五次：

| 指标 | 强制 8 workers | 自适应父进程 | 变化 |
| --- | ---: | ---: | ---: |
| wall 中位 | 1.6124 s | 0.1678 s | -89.6% |
| 最大 child RSS 增量 | 约 105–106 MB | 0 | 不再创建 child |

两条路径的 `.route` SHA-256 均为
`130f2eee92eaef12c9a5c55f9671422c83bf74a44f63fef89c6ff437db1f1454`。另以
10,000 原子 × 500 帧（500 万 scan values）验证，自适应规则仍保留 8 workers，wall
约 3.91–4.01 秒，没有把长扫描退化为串行路径。

### 最终实现

- route worker 数按总 `atom_count × frame_count` 调整，每个 worker 目标至少承担
  65,536 个扫描值，同时不超过用户请求的 `nproc`；日志记录实际降核及总扫描值。
- 计算出的 worker 数为 1 时，父进程直接逐原子读取现有 memmap，不创建单 worker
  Pool、pipe 或 ordered spool。
- 串行与多进程路径共用 `_get_atom_route_result()`，保持 pair compaction、UTF-8 输出
  和后续聚合表示一致。
- 回归分别锁定 worker 选择、短任务禁止创建 Pool、HMM/no-HMM 聚合及 worker
  compaction 语义。

### 真实轨迹与适用边界

默认 1 MiB HDF5 cache 下，只读 `rp3.lammpstrj` 使用 `nproc=8` 完整运行。日志确认
route 从 8 workers 降为父进程直跑；Step 3 molecule/matrix/route/reaction 为
3.210/0.101/0.208/0.538 秒，core computation 为 3.916 秒。相对紧邻的修复前基线，
route 由 1.764 秒降至 0.208 秒（约 -88.2%），core computation 由 5.628 秒降至
3.916 秒（约 -30.4%）。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。完整
`tests/test_timedoutput.py` 为 75 passed。

该收益针对短时间轴上的 route 过度并行。生产全轨迹若 `N×T` 足够大仍会保留请求的
worker 数；它不能替代对 48 小时 molecule/HDF5 阶段的 Slurm 实测。生产结论仍需比较
各 Step 3 子阶段、writer seconds、TotalCPU、MaxRSS 和临时盘峰值。

## 2026-08-07：SMILES 结构复杂度自适应并行

### 任务数不能代表可并行收益

此前已经证明真实 5 帧样例的 SMILES 很便宜，但仅按 molecule 数判断仍可能在放大数据
后误用多进程。复用同一批 5,615 个真实结构记录，重复完整记录并让每个配置重新创建
worker pool：

| 结构负载 | molecule 数 | 1 proc | 2 proc | 4 proc | 8 proc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 真实记录 | 5,615 | 0.329 s | 3.224 s | 5.481 s | 5.593 s |
| 真实记录 × 5 | 28,075 | 1.580 s | 4.731 s | 7.693 s | 9.942 s |
| 真实记录 × 20 | 112,300 | 5.473 s | 18.039 s | 15.375 s | 28.344 s |

即使扩到 112,300 项，逐任务 IPC、Pool 和有序恢复成本仍高于父进程直接执行；因此不能
用“任务数量很多”作为启用全部 CPU 的充分条件。

为防止反向把重分子也强制串行，另构造真实 RDKit 路径的线性碳链对照：

| 结构负载 | 单记录压缩结构字节 | 1 proc | 2 proc | 4 proc | 8 proc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5,000 × 128 atoms | 2,399 B | 4.895 s | 4.002 s | 4.408 s | 5.761 s |
| 2,000 × 512 atoms | 6,941 B | 12.313 s | 8.198 s | 6.889 s | 7.542 s |

重结构确实重新获得多进程收益，而且合理 worker 数随单记录复杂度增加。绝对 wall 会受
进程冷启动和系统负载影响，因此这些数据用于确定调度方向和数量级，不作为跨机器固定
性能承诺。

### 最终实现

- HMM 已经逐条接收并复制保留下来的 molecule record；现在在同一循环中只累加前三个
  压缩结构 block 的长度，不解压内容、不增加文件扫描，也不把 frame block 误计为
  SMILES 工作量。
- worker 数同时受两项约束：平均每 2,048 个压缩结构字节增加一个复杂度级别；每个
  实际启动的 worker 至少需要约 3 MiB 总结构数据来摊薄固定成本。结果再限制到用户
  请求核数和 molecule 数。
- 选择 1 个 worker 时复用现有父进程零 IPC 路径；多进程路径继续使用最小 initializer、
  有界 in-flight 和磁盘有序恢复。
- 旧中间调用若没有 `smilesworkmetrics` 指标，保持原请求并行度，避免改变外部手工流水线
  的行为；日志记录请求核数、实际核数、molecule 数和平均压缩结构字节。
- 回归覆盖真实/放大/重结构选择、HMM 指标累计、缺失指标回退和自适应串行不得创建
  Pool。

### 完整样例与结果等价性

只读 `rp3.lammpstrj` 的 5,615 个 molecule 共 1,905,017 个压缩结构字节，平均
339.3 B；请求 `nproc=8` 时自适应选择父进程直跑。Step 3
molecule/matrix/route/reaction 为 1.078/0.125/0.255/0.610 秒，core computation 为
1.901 秒，Step 3 总 wall 为 2.108 秒。相对紧邻的 route 修复后基线，molecule 由
3.210 秒降至 1.078 秒（约 -66.4%），core computation 由 3.916 秒降至 1.901 秒
（约 -51.5%）。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。完整
`tests/test_timedoutput.py` 为 82 passed；相关非 GUI 回归为 13 passed。

该调度会主动降低廉价阶段的 CPU 使用率，但同时减少 wall、worker RSS、IPC 和
core-hours；因此不能只把利用率百分比当成性能目标。生产作业仍应根据日志中的实际
worker 数和各阶段 wall 调整下一次 Slurm 申请，并验证复杂结构分布、共享文件系统和
64 核环境下的阈值。

## 2026-08-07：串行 SMILES 结构只解码一次

### 红灯与根因

自适应调度使廉价结构进入父进程后，串行生成器先调用
`_calmoleculeSMILESname(record)` 解压 atoms、bond pairs 和 bond levels；外层随后又为
`.moname`、miso fallback 和 HDF5 调用 `_getatomsandbonds(record)`。同一条压缩结构因而
执行两次 LZ4/pickle 解码、NumPy 数组构造和 Python bond-list 重建。

现有自适应串行回归增加 `_getatomsandbonds()` 调用计数；旧路径对一条 record 稳定观察
到 2 次，修复目标为 1 次。多进程路径仍需要 worker 计算 SMILES、父进程准备输出，
没有把 atoms/bonds 再塞回 IPC 结果来换取这项局部收益。

### 最终实现与隔离 A/B

- `_get_serial_smiles_record()` 一次解码后同时返回 species name、原压缩 record、atoms
  和 bonds；后续 fallback、miso、`.moname` 和 timeline writer 复用同一对象。
- `_calmoleculeSMILESname_from_decoded()` 成为串行与原 worker 入口共享的实际转换函数；
  worker 的外部返回值仍然只有短 species name。
- 生成器每次只保留当前 molecule 的解码结果，内存仍为常数级；相较旧路径还避免在
  同一时刻重新分配第二组结构对象。

从同一只读轨迹重新生成 5,615 条真实压缩结构，在同一进程内预热后交替运行前后路径
各五次，checksum 均为 129,173：

| 指标 | 双解码中位 | 单解码中位 | 变化 |
| --- | ---: | ---: | ---: |
| SMILES + 下游结构准备 | 0.2946 s | 0.2654 s | -9.9% |
| 仅解码和 bond 重建 | 0.0630 s | 0.0259 s | -58.9% |

完整样例再次通过语义校验，但本轮运行同时出现 Step 1 从约 17.5 秒波动到 23.2 秒、
HDF5 writer 从约 0.177 秒波动到 0.225 秒；molecule stage 的单次 1.384 秒不能用于
判断这项约 0.03 秒的局部收益。因此性能结论采用同批数据交替 A/B，不把系统负载噪声
包装成完整阶段加速比。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` 哈希继续保持既有基线。

## 2026-08-07：reaction 邻接边在 DFS 前去重

### 红灯与根因

reaction worker 按变化原子构造前后分子二部图。旧实现对每个变化原子都向
`reactdict` 追加一条邻接边；若一个百万原子分子整体从 molecule 1 映射到 molecule 2，
图中会保留一百万条 `1 -> 2` 和一百万条 `2 -> 1`，但后续 DFS 只需要知道这两个节点
相连。重复边不会改变连通分量或反应计数，只会放大 Python list、遍历时间和 worker
RSS。

新增回归用 100,000 个相同映射原子截获 DFS 输入。修复前邻接元素总数为 200,000，
目标和修复后均为 2。另一个回归覆盖高出度容器升级，确保去重后仍保持首次出现顺序，
不因无序集合改变 DFS 的稳定遍历次序。

### 最终实现

- 低出度邻接继续使用 list；先做有界线性查重，避免大量小 dict 的对象开销。
- 唯一邻居超过 8 个时升级为保持插入顺序的 dict，后续查重为摊销常数时间；升级过程
  保留已有邻居及首次出现顺序。
- 正向、反向和 conflict 邻接统一经过同一入口。没有使用 set，文本输出的潜在平局顺序
  不受哈希迭代顺序影响。
- `dps_reaction` 只依赖邻接对象可迭代及长度，不依赖 list 的重复元素，因此无需修改
  Cython DFS 或 worker IPC 协议。

### 隔离反馈环

每个样本在独立子进程中运行，前后实现交替各三次；表中为 wall 和函数执行前后
`ru_maxrss` 增量的中位数：

| 场景 | 旧耗时 | 新耗时 | 旧 RSS 增量 | 新 RSS 增量 |
| --- | ---: | ---: | ---: | ---: |
| 1,000,000 个原子、同一分子边 | 0.413 s | 0.359 s（-13.0%） | 64.3 MiB | 3.1 MiB（-95.2%） |
| 200,000 个原子、全部唯一边 | 0.177 s | 0.183 s（+3.8%） | 90.0 MiB | 90.2 MiB（+0.2%） |

高重复生产形态显著降低内存并稍微加速；全唯一最坏对照的内存基本不变，时间代价约
3.8%。因此该优化没有用新的无界索引换取重复边收益。

### 完整样例与结果等价性

只读 `rp3.lammpstrj` 的 5 帧完整运行中，Step 3
molecule/matrix/route/reaction 为 0.975/0.097/0.231/0.646 秒，core computation 为
1.788 秒，Step 3 总 wall 为 2.003 秒。该小样本只有 4 个 transition，本次运行只用于
正确性确认，不据单次 reaction wall 声称加速。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。完整
`tests/test_timedoutput.py` 为 85 passed。

生产全轨迹仍需用 Slurm `MaxRSS` 和 reaction 子阶段 wall 验收；该优化只移除 worker
内部的重复图边，不解决共享文件系统吞吐、SMILES 复杂度分布或其他阶段的瓶颈。

## 2026-08-07：reaction 按观测工作量自适应并行

### 红灯与根因

真实 5 帧样例只有 4 个 transition，但请求 `nproc=8` 时 reaction 仍创建 4 个 worker
进程。新增调用点回归把 `run_mp` 替换为失败函数；旧路径稳定失败于“短 reaction 不得
创建 Pool”。最小输入只包含 1 个原子和 1 个 transition，直接覆盖进程启动、memmap
initializer、结果消费及 `.reactionabcd` 写出。

隔离基准验证 Pool/IPC 固定成本是主因，而不是 DFS 或 memmap 扫描：

| workload | 串行 | 2 workers | 4 workers | 8 workers |
| --- | ---: | ---: | ---: | ---: |
| 49,304 scan / 492 change | 0.0005 s | 0.73–1.02 s | 1.05–1.45 s | — |
| 990,000 scan / 9,900 change | 0.0148 s | 0.947 s | 1.245 s | 1.606 s |
| 4,990,000 scan / 4,990 change | 0.0209 s | 0.851 s | 1.035 s | 1.515 s |
| 99,950,000 scan / 1,999 change | 0.145 s | 0.696 s | 0.934 s | 1.439 s |
| 495,000 scan / 495,000 change | 0.488 s | 0.997 s | 1.194 s | 1.670 s |
| 1,990,000 scan / change | 1.191 s | 1.225 s | 1.191 s | 1.406 s |
| 2,990,000 scan / change | 1.899 s | 1.589 s | 1.436 s | 1.584 s |

约 299 万个全变化值后并行开始有稳定收益；499 万个全变化值的两轮测试中，串行为
3.153–5.641 秒，4 workers 为 3.020–3.296 秒，8 workers 为 2.762–3.112 秒。短稀疏
样例的 4-worker 路径还产生约 100.3 MiB 的单个 child MaxRSS，父进程直跑没有子进程。

### 最终实现

- route 本来已经为每个 atom 找出 molecule 变化点；worker 现在顺带返回变化次数，
  父进程累计为 reaction 调度指标，不重新扫描 `atomeach`。
- 变化次数独立于 `--selectatoms` 的 route 显示筛选；即使不输出某类原子的 molecule
  pair，其变化仍计入 reaction 工作量。
- 若时间线含未归属的零 ID，route 文本继续保持既有过滤语义，但调度指标按最多
  65,536 行的有界块统计原始相邻变化，避免漏掉 reaction 实际会扫描的 `1 -> 0 -> 1`。
- 每约 1,000,000 个已观测变化事件增加一个 worker；即使变化稀疏，每约
  500,000,000 个 atom-transition 扫描值也增加一个 worker。结果再限制到用户请求核数
  和 transition 数。
- 指标缺失的独立旧调用保持原 worker 数语义，仅继续限制到 transition 数；指标存在且
  只需 1 个 worker 时在父进程直接执行，不创建 Pool、pipe 或 worker memmap。
- 串行和多进程路径共享 `_get_transition_reaction_result()`，Counter、HDF5 event block
  和 `.reactionabcd` 输出契约不变；日志报告请求/实际 worker、变化事件和扫描值。

### 真实轨迹与结果等价性

只读 `rp3.lammpstrj` 实际观测 12,751 个变化事件和 49,304 个扫描值，请求 8 核时选择
父进程。相对紧邻的邻接去重版本，reaction 阶段由 0.646 秒降至 0.030 秒（约
-95.4%），core computation 由 1.788 秒降至 1.043 秒（约 -41.7%），Step 3 总 wall
由 2.003 秒降至 1.218 秒（约 -39.2%）。其他阶段仍有系统负载波动，因此该轮性能结论
主要采用 reaction 的同一真实输入和前述隔离交叉点，而不外推全流程加速比。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。完整
`tests/test_timedoutput.py` 为 95 passed。

生产 Slurm 仍需记录实际 worker 选择、reaction wall、TotalCPU 和 MaxRSS。阈值来自
本机 spawn/共享页缓存反馈环，不作为不同节点和文件系统上的固定最优值；日志足以支持
后续按生产证据校准。

## 2026-08-07：reaction 重复 pair 在进入 Python 图前压缩

### 红灯与根因

上一轮已让邻接容器只保存唯一 molecule edge，但 `_calculate_transition_reactions()` 仍
逐个遍历所有变化原子并调用两次 `_add_reaction_neighbor()`。100,000 个完全相同的
`1 -> 2` 映射最终只保存 2 个邻接元素，却仍执行 200,000 次 Python helper 调用。

新增红灯在真实 transition 计算入口统计 helper 次数。旧路径稳定为 200,000；最终
1024 行有界压缩允许每个压缩块各调用一次正/反向 helper，该输入最多 196 次。测试同时
要求最终 reaction Counter 保持 `A -> B`，避免只优化结构计数而绕开真实 DFS。

### 候选取舍与隔离 A/B

无条件对完整变化数组执行 `np.unique` 会伤害全唯一数据，并在重复调用中增加 NumPy
allocator 的峰值。因此最终路径先抽样、再用小块压缩；表中是正式源码交替三次的中位
wall：

| 输入 | 原逐行路径 | 当前压缩路径 | 变化 |
| --- | ---: | ---: | ---: |
| 1,000,000 个相同 pair | 0.3282 s | 0.00546 s | -98.3% |
| 1,000,000 个交错的 100 种 pair | 0.3180 s | 0.0557 s | -82.5% |
| 200,000 个全唯一 pair | 0.0979 s | 0.0964 s | 无可测回退 |

初版 65,536 行 unique 块在 1,000 万事件压力中会让 allocator 峰值增加约 25–30 MiB，
因此没有进入正式代码。最终 1,024 行块的五次独立进程 MaxRSS 增量中位由上一版约
4.33 MiB 变为 9.44 MiB，增加约 5.11 MiB，但不随 1,000 万总事件线性增长。该常数级
临时空间换取重复形态约 60× 的局部加速；仍远低于最初“每个原子保留一条 Python
邻接”的线性内存路径。

### 最终实现与语义保护

- 每个 65,536 行 scan block 最多抽样 1,024 个变化 pair；只有唯一比例不超过 75% 才
  启用完整压缩。全唯一路径只支付小样本检查，继续使用原逐行逻辑。
- 正式压缩块固定为 1,024 行。先压缩连续相同 pair；若仍高度交错，再进行 unique，
  避免常见成组原子无谓排序。
- 常见 32-bit molecule ID 把 pair 编码为单个 `uint64` key；更宽 ID 使用二维 unique。
  两条路径都按 first index 恢复首次出现顺序，不使用无序 set。
- conflict molecule 同样只保留首次标记，并继续使整个连通 reaction 被过滤；正反向图、
  DFS 和文本/HDF5 契约不变。
- 回归包含：helper 调用上界、全唯一抽样回退、超过 32-bit 的 molecule ID、conflict
  component、旧逐行路径差分，以及 20 组固定种子的重复随机二部图。

### 真实轨迹与检查

只读 `rp3.lammpstrj` 完整运行的 reaction 阶段为 0.024 秒，上一轮为 0.030 秒。6 ms
差异已接近本机计时噪声，因此不据此宣称真实样例加速；该样例只用于端到端正确性。
Step 3 molecule/matrix/route/reaction 为 0.882/0.102/0.203/0.024 秒，总 wall 为
1.242 秒。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。完整
`tests/test_timedoutput.py` 为 100 passed。

生产端仍需从实际变化 pair 分布判断命中率；若多数 block 全唯一，抽样会安全回退但
不会带来上述重复形态收益。

## 2026-08-07：reaction 运行检测 workspace 复用与候选淘汰

### 红灯与最终实现

重复 pair 压缩会对每个 1,024 行块先判断连续相同的 run。初版每个块分别创建
`run_starts` 和 `after_changes` 两个布尔数组；100,000 个相同 pair 的入口回归观察到
198 次显式布尔 `np.empty()`，总事件继续放大时会反复进入 NumPy allocator。

新增红灯锁定“同一个 transition 的所有压缩块最多创建两个显式布尔 workspace”。最终
在确认需要压缩后惰性创建 1,024 和 1,023 项的两个布尔数组，并把切片传给 run 检测；
独立调用 helper 时仍允许不传 workspace，接口和测试用途保持兼容。修复后同一回归只
创建 2 次，空间上限不随压缩块数增长。

1,000 万个相同 pair 使用独立子进程，前后路径交替各五次。旧的逐块分配路径中位 wall
为 0.09381 秒、`ru_maxrss` 增量为 9.938 MiB；复用后为 0.08199 秒和 9.234 MiB，wall
约缩短 12.6%，RSS 中位约减少 0.70 MiB。该收益虽小，但同时改善时间和内存，因此保留。

### 未保留的候选

- 用 `np.take(..., out=...)` 复用 `before_values/after_values` 数值数组时，1,000 万重复
  pair 的中位 wall 从 0.0779 秒增至 0.1339 秒（约 +72%），RSS 从 8.656 MiB 增至
  9.141 MiB；高级索引的实际实现更快，该候选和红灯测试均已回退。
- 将压缩块从 1,024 降为 512 时，1,000 万重复 pair 的中位 wall 从 0.0830 秒增至
  0.1226 秒；1,000,000 个 100 种交错 pair 也从 0.0936 秒增至 0.1381 秒，均慢约
  48%。交错场景只节省约 1.1 MiB RSS，不足以抵消稳定时间回退，因此继续使用 1,024。
- 为交错 unique 分支复用一个 `uint64` pair-key 数组时，1,000,000 个 100 种交错 pair
  的中位 wall 从 0.08874 秒增至 0.10260 秒（约 +15.6%），RSS 中位从 4.422 MiB 增至
  4.656 MiB。固定赋值/类型转换成本没有被少一次小数组分配抵消，候选和专用测试已
  回退。

这些负结果保留在日志而不留在源码中，避免后续仅凭“减少 allocation”再次引入已经
测得的反优化。

### 最终回归与真实轨迹

- `pytest -q tests/test_timedoutput.py`：101 passed。
- `pytest -q tests/test_reacnetgen.py -k 'molecule or step3 or timestep or parm2cmd'`：
  11 passed、40 deselected；`tests/test_tools.py tests/test_detect.py`：19 passed。
- 本轮 `_reaction.py` 和 `test_timedoutput.py` 的 Ruff lint/format 通过，`compileall` 和
  `git diff --check` 通过。全目录 Ruff 另报告既存生成文件 `_version2.py` 的
  `__all__` 排序/格式问题，本轮未改该生成文件。

只读 `rp3.lammpstrj` 再次以 `nproc=8`、molecule timeline 和 reaction event 完整运行。
Step 3 molecule/matrix/route/reaction 为 0.819/0.115/0.196/0.024 秒，总 wall 为
1.186 秒；HDF5 writer 记账为 0.142 秒、molecule/reaction 分别 2/1 个批次。短样例的
单次 wall 只作正确性和数量级观察，不外推为生产加速比。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。原始输入 SHA-256
仍为 `c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

生产 Slurm 全轨迹仍需记录 pair 压缩命中率、reaction wall、TotalCPU、MaxRSS 和临时盘
峰值；本地改动不能代替该最终验收。

## 2026-08-07：matrix 密集 signal 连续切片写入

### 红灯与剩余临时分配

`_getatomeach()` 已经把一个 molecule 的 signal 按最多 1,048,576 帧分块，但每个块仍
无条件执行 `np.flatnonzero()`。当长寿命 molecule 在整个块内存在时，这会先创建最多
约 8 MiB 的 `intp` frame index，再通过 `np.ix_()` 读写本来连续的 frame 区间。

新增的真实 `_getatomeach()` 调用点把 `flatnonzero` 替换为失败函数；两个完全重叠的
40 帧 dense molecule 在修改前稳定失败，同时锁定后写覆盖和 `conflict=True` 语义。
另一个红灯用 256 帧全零块跟踪 `np.all()` 输入，防止 dense 探测为了节省 dense 索引却
先对常见 empty/sparse block 增加一次完整扫描；初版候选稳定观察到 256 行，最终上限为
64 行前缀。

### 最终实现

- 每个 signal block 先检查最多 64 项前缀；前缀全真且 block 末项也为真时，才执行一次
  完整 `np.all()` 确认。empty 和常见 sparse block 不增加全块预扫描。
- 确认 dense 后不再构造 frame index，而是按现有 1,048,576 cell 上限直接使用连续
  `slice`。atom 分块、cell budget 和 memmap 后备存储保持不变。
- sparse block 继续使用原有有界 `flatnonzero + np.ix_` 路径；没有把稀疏选择展开为
  连续矩形写入。
- dense/sparse 两条路径共享同一个覆盖写入 helper，继续先检测既有 molecule ID、合并
  conflict，再写入新的 molecule ID。
- matrix 日志新增 dense/sparse/empty block 数和未生成索引的 dense frame value 数，
  生产 Slurm 可直接判断该优化是否命中。
- 差分回归覆盖跨 scan 边界的 dense、sparse、empty 混合，多个非连续 atom、小 cell
  budget、后写覆盖和 conflict；不只验证“没有调用某个 NumPy API”。

### 独立进程 A/B

合成负载为 500 万帧、一个 atom、两个完全重叠 molecule。当前源码通过强制关闭 dense
识别复现旧索引路径，前后模式交替 9 轮；表中“成对比值”为每一轮
`current / previous` 后取中位：

| signal | 旧 wall 中位 | 新 wall 中位 | wall 成对比值 | RSS 成对比值 |
| --- | ---: | ---: | ---: | ---: |
| 100% dense | 0.12982 s | 0.04758 s | 0.374（约 -62.6%） | 0.454（约 -54.6%） |
| 50% sparse | 0.08639 s | 0.08290 s | 0.982 | 1.026 |
| 1% sparse | 0.04709 s | 0.04652 s | 0.938 | 0.860 |
| empty | 0.02732 s | 0.02705 s | 0.996 | 0.979 |

macOS 独立进程的 page cache 和 `ru_maxrss` 有明显离散波动，因此只把 dense 的稳定数量
级下降作为内存收益；其余三组用于排除系统性 wall 回退，不把噪声级 RSS 差异包装成
额外优化。

### 真实轨迹与回归

在加入下一节 no-HMM frame 复用之前，本节实现对只读 `rp3.lammpstrj` 完整运行时报告
`3679 dense / 1936 sparse / 0 empty`，
即短样例 65.5% 的 molecule signal block 命中连续路径，共 18,395 个 frame value 未生成
索引。该输入只有 5 帧，不能代表生产 1,048,576 帧 block 的命中率；新增日志用于下一次
全轨迹直接取证。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。

最终 `tests/test_timedoutput.py` 为 104 passed；molecule/Step 3/timestep/parm2cmd 子集
11 passed、40 deselected；tools/detect 19 passed。本轮 `_path.py` 和测试的 Ruff
lint/format、`compileall`、mdformat 和 `git diff --check` 通过后清理合成基准与真实输出。

生产验收仍需同时记录 matrix block 命中分布、`step3_matrix_seconds`、MaxRSS、major page
fault 和 memmap 临时盘吞吐；本地 dense 合成收益不能替代全轨迹结论。

## 2026-08-07：no-HMM matrix 复用已有 frame index

### 重复工作的根因与红灯

`_HMMFilter._getoriginandhmm()` 从 molecule 临时记录的第 4 个字段读取 frame index，再用
`idx_to_signal(..., step)` 展开为完整布尔 origin signal。no-HMM 模式不会改变这个
signal，但原 `_getatomeach()` 随后仍逐 molecule 解压完整 signal、分块扫描，并用
`flatnonzero()` 重新生成临时记录中本来已经存在的同一组 frame index。

新增红灯把 origin 文件内容替换为合法外层长度头加损坏的 LZ4 payload，同时保留两个
合法 molecule frame block。修改前 matrix 会尝试解压 origin 并报错；修改后应完全依赖
已有 frame block，并继续得到后写覆盖及交叠帧 `conflict=True`。同一测试还跟踪 molecule
记录读取器：旧路径没有选择字段，新增断言先稳定失败，最终只读取 `(atoms, frames)`，即
原记录的 `(0, 3)` 字段，跳过 matrix 不使用的 bond 等 payload。

最终审阅又发现 signal fallback 的空块分支会在 `continue` 前跳过局部释放：若最后一个
block 为空，`signal_block` view 会把整条旧 signal 保留到下一条 signal 解码，造成相邻两
条完整 signal 短暂重叠。新增弱引用红灯在第二条解码时确认第一条仍存活；在空块分支先
删除 `frames` 和 `signal_block` 后，上一条 signal 会在下一条解码前释放。

### 自适应实现

- no-HMM 先从 frame block 的 LZ4 header 读取未压缩 pickle 大小，不解压 payload。只有
  `0 < content_size <= step + 256` 时才直接解码 frame index；该上限使 index ndarray 的
  payload 大致不超过原完整 bool signal，避免长寿命 molecule 从约 `T` 字节 signal
  回退成约 `8T` 字节 `uint64` index。
- 命中时直接复用已有 frame index，按既有 atom/cell budget 有界写入 memmap，不再解压、
  扫描 origin signal，也不再调用 `flatnonzero()`。覆盖顺序和 conflict 合并继续使用同一
  写入 helper。
- 超过内存阈值的 dense no-HMM molecule，以及所有 HMM molecule，仍走完整 signal 路径；
  其中全真块继续使用上一节的连续 slice，不生成 dense frame index。
- matrix 从 molecule 临时记录按模式选择字段：no-HMM 读取 `(0, 3)`（atoms、frames），
  HMM 只读取 `(0,)`（atoms）；每个 molecule 完成后显式释放 atoms 以及 frames 或 signal，
  避免上一条记录的右值在下一条解码时继续存活。
- signal fallback 的 empty block 在 `continue` 前也显式释放空 index 和 block view，避免
  最后一块为空时把上一条完整 signal 延寿到下一条解码。
- 新日志同时报告直接复用的 no-HMM molecule/frame value 数，以及回退后的
  dense/sparse/empty signal block 数，生产作业可直接确认实际分支。

### 9 轮交替独立进程 A/B

稀疏场景为 500 万帧、50 个 molecule、每个 molecule 仅存在于 1,000 帧；旧模式强制走
signal 解码/扫描，新模式走 frame 直写。密集场景为 100 万帧、5 个全寿命 molecule，
两种模式均应走 signal + dense slice，用于排除阈值检查带来的回退。每轮交替前后运行
顺序；成对比值为每轮 `current / previous` 后取中位：

| 场景 | 旧 wall 中位 | 新 wall 中位 | wall 成对比值 | 旧/新 RSS 中位 | RSS 成对比值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 稀疏 frame 直写 | 0.15260 s | 0.04416 s | 0.286（约 -71.4%） | 56.547/8.625 MiB | 0.156（约 -84.4%） |
| 密集 signal 回退 | 0.04023 s | 0.03737 s | 0.962 | 35.781/31.703 MiB | 0.906 |

macOS 独立子进程的 page cache 和 `ru_maxrss` 仍有离散波动，因此不把密集场景的小幅
下降当成额外收益；它的用途是证明自适应选择没有把 dense 输入改成 `8T` index，也没有
观察到系统性 wall 回退。稀疏场景的时间和内存下降跨 9 轮均保持同一数量级。

### 最终真实轨迹与回归

最终源码对只读 `rp3.lammpstrj` 以 `nproc=8` 完整运行时，5,615 个 no-HMM molecule
全部直接复用，共读取 22,103 个 frame value；signal block 计数为 `0/0/0`，说明 matrix
阶段没有再解压这些完整 origin signal。单次 matrix stage 为 0.120 秒，Step 3 总计
1.262 秒；短样例只用于正确性和分支取证，不外推为生产加速比。

HDF5 语义指纹继续为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 继续分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。

最终 `tests/test_timedoutput.py` 为 108 passed；molecule/Step 3/timestep/parm2cmd 子集
11 passed、40 deselected；tools/detect 19 passed。`_path.py` 与测试的 Ruff lint/format、
`compileall`、mdformat 和 `git diff --check` 均通过。

生产 Slurm 全轨迹仍需记录 no-HMM direct molecule/frame 命中数、signal fallback 分布、
matrix wall、MaxRSS、page fault 和临时盘吞吐；若生产数据存在大量长寿命 molecule，
自适应阈值会保留 signal 路径而不是用时间换取 `uint64` index 内存膨胀。

## 2026-08-07：no-HMM filter 稀疏直通与临时文件复用

### 红灯与端到端反证

上一节让 matrix 对 no-HMM 稀疏记录直接复用 frame index，但 Step 2 仍无条件执行
`frame index -> 完整 origin signal -> LZ4`。新增的 500 万帧红灯把
`_hmmfilter.bytestolist` 替换为失败函数；修改前 `_getoriginandhmm()` 连续两次都在解码
frame block 时失败，证明 matrix 不再消费的完整 signal 仍被实际构造。

第一版只把稀疏 signal 替换为空占位块后，单条函数微基准大幅改善，但真实 `rp3` 的
Step 2 仍为 5.098 秒，落在此前 3.5--4.0 秒单次波动附近。这一端到端反证说明工作已从
NumPy/LZ4 转移到 Pool 启动、IPC、结果回传和 molecule 临时文件复制，不能用微基准替代
阶段级验证。第二个红灯让 `run_mp` 直接失败；5,615 条全 direct no-HMM 记录在修改前
稳定进入该失败函数，最终则完全在父进程完成。

### 最终实现

- frame index 是否适合直接使用的判定移到 `utils`，HMM filter 和 matrix 共用同一 LZ4
  未压缩 payload 阈值，避免生产端与消费端分支不一致。
- no-HMM direct 记录不再解码 frame block、不再构造/压缩完整 origin signal，也不写
  空占位块。origin 临时文件只顺序保存真正拒绝 direct 的 dense fallback signal。
- matrix 根据同一阈值遍历 molecule 记录：direct 记录不消费 origin block，fallback
  记录才读取下一条 signal。固定种子的 direct/fallback 混合差分以及完整 filter-to-matrix
  集成测试均与强制 signal 参考逐元素一致。
- no-HMM 先在父进程处理 direct 前缀；全 direct 时完全不创建 Pool。一旦遇到第一个
  fallback，才把该记录及剩余记录交给原有并行路径，保留密集工作负载的并行能力。
- no-HMM 不会过滤 molecule，因此 `moleculetemp2filename` 直接复用 Step 1 的
  `moleculetempfilename`，不再顺序写第二份完全相同的压缩记录。HMM 模式仍保留筛选后的
  独立文件和原有处理语义。
- 新日志报告 direct/总 molecule 数、fallback origin MiB、避免复制的 molecule 文件
  MiB，以及是否实际触发并行 fallback，供生产作业直接验收。

### 9 轮交替独立进程 A/B

第一组测单条 no-HMM origin 生成。稀疏场景为 500 万帧、20 个 molecule、每个仅 1,000
个 frame index；密集场景为 100 万帧、5 个全寿命 molecule。旧路径显式重建完整
signal，当前路径按共享阈值处理：

| 场景 | 旧 wall 中位 | 新 wall 中位 | wall 成对比值 | 旧/新 RSS 增量中位 | 旧/新 origin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 稀疏 direct | 0.050356 s | 0.0000307 s | 0.000624（约 -99.94%） | 40.797/0 MiB | 0.404625/0 MiB |
| 密集 fallback | 0.033997 s | 0.034350 s | 0.974 | 23.000/23.109 MiB | 0.020804/0.020804 MiB |

稀疏当前路径的运行时间已接近计时器和 Python 调用固定成本，数量级收益来自完全跳过
`T` 长度分配；密集路径的输出完全相同，wall/RSS 差异属于独立进程噪声，没有观察到
系统性回退。

第二组只测 `_calhmm()`，使用 5,615 条、5 帧的真实规模形态，所有任务都已经是零
signal 分配，以隔离 Pool/IPC。强制 8 进程的中位 wall 为 2.22556 秒，父进程直通为
0.030896 秒，成对比值 0.01424（约 -98.58%）。两者 origin 均为 0 字节。该基准的
`RUSAGE_SELF` 不包含 worker，因此不据其 RSS 数字宣称总内存下降；不创建 8 个子进程
会消除其常驻内存，但最终数值仍以 Slurm MaxRSS 为准。

### 最终真实轨迹与回归

最终源码对只读 `rp3.lammpstrj` 以 `nproc=8` 完整运行：5,615/5,615 条记录 direct，
fallback origin 为 0 字节，避免复制 3.139 MiB molecule 临时文件，并行 fallback 未
启用。Step 2 为 0.063 秒；与本轮第一版“signal 已短路但仍强制 Pool”的 5.098 秒相比
约缩短 98.8%。Step 3 molecule/matrix/route/reaction 分别为
1.041/0.125/0.241/0.030 秒，总计 1.476 秒。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。

最终 `tests/test_timedoutput.py` 为 113 passed；molecule/Step 3/timestep/parm2cmd 子集
11 passed、40 deselected；tools/detect 19 passed。`_hmmfilter.py`、`_path.py`、`utils.py`
与测试的 Ruff lint/format、`compileall`、mdformat 和 `git diff --check` 均通过。

生产 Slurm 需同时记录 no-HMM direct 命中数、fallback origin MiB、是否触发并行、Step 2
wall、MaxRSS 和临时盘峰值。若生产轨迹前部很早出现 dense fallback，后续记录仍会进入
Pool；新增日志可直接识别该形态，不能用全 direct 的本地比例替代生产结论。

## 2026-08-07：HMM 与 dense fallback 自适应批次

### 红灯与根因

`_HMMFilter._iter_filter_results()` 的 HMM 路径和 no-HMM dense fallback 原先都沿用
`run_mp()` 默认值：`chunksize=100`、`max_inflight=150*nproc`。在 64 核作业中，这允许
最多 9,600 条 molecule record 同时在生产队列、Pool 管道或结果等待区；每条 record 的
第 4 个字段又可能包含完整长轨迹 frame index。`chunksize=100` 还会把 100 条记录绑定为
一个 Pool task，记录数不足或单条耗时不均时，能够参与工作的 worker 数显著少于申请
核数。

新增红灯构造 `step=1_000_000`、`nproc=64` 的 HMM filter，并截获 `run_mp` 参数，要求
`chunksize=1`、`max_inflight=128`。修改前连续两次都因没有传入这两个参数而在
`captured["chunksize"]` 稳定失败。no-HMM dense fallback 另有断言，保证它使用同一组
边界而不是继续走旧默认值。

### 自适应实现

批次由每条 signal 的帧数决定，目标是每个 Pool chunk 最多约处理 100 万个 signal
value，同时不改变短轨迹原有的 100 条批次：

```python
chunksize = max(1, min(100, 1_000_000 // max(1, frame_count)))
max_inflight = min(
    150 * nproc,
    max(chunksize, 2 * nproc * chunksize),
)
```

| frame 数 | 64 核 chunksize | 64 核 max_inflight |
| ---: | ---: | ---: |
| 1 | 100 | 9,600 |
| 10,000 | 100 | 9,600 |
| 100,000 | 10 | 1,280 |
| 1,000,000 | 1 | 128 |

- HMM 全量过滤和 no-HMM 首个 dense record 后的并行 fallback 共用 helper，防止两条路径
  继续漂移。
- 目标窗口约为每个 worker 两个 chunk；原 `150*nproc` 上限优先，因此短轨迹可能低于
  两个，但全局始终至少容纳一个完整 chunk。
- `maxtasksperchild` 和结果语义没有变化；本轮只改变 Pool task 粒度和 producer 回压。
- no-HMM fallback 明确使用 `unordered=False`：其 origin signal 必须与复用的原始 molecule
  文件保持同一记录顺序。HMM 仍可乱序，因为 signal 与筛选后的 molecule record 会成对
  按同一完成顺序写入各自文件。
- 运行日志新增 frame 数、实际 `chunksize` 与 `max_inflight`，生产 Slurm 可直接确认
  长轨迹是否命中边界。

### Code review 发现并修复的顺序正确性问题

需求与工程标准两条独立复审都发现同一 P1：no-HMM 已经复用输入有序的 Step 1 molecule
文件，但首版 dense fallback 仍继承 `run_mp(unordered=True)`。若两个 dense worker 逆序
完成，`originbytes` 会按完成顺序写入，matrix 却按原 molecule 顺序消费 signal，从而把
存在时间静默分配给错误 atom，继续污染 route 和 reaction。

新增红灯让受控 `run_mp` 在未声明保序时反转两条不同 dense record；修改前稳定观察到
`requested_ordering == [True]` 并失败。最终回归使用真实 `nproc=2`、`chunksize=1`：较早
dense task 延迟 0.3 秒，较晚 task 只延迟 0.01 秒，之后逐元素验证三个不同 atom 的
matrix。设置 `unordered=False` 后通过。这里没有为 fallback 增加第二个磁盘 spool；现有
`max_inflight` 已限制等待窗口。若生产任务时长偏斜仍出现明显队头等待，应评估带索引的
磁盘保序或显式 signal 映射，不能重新打开无索引乱序。

第二轮标准复审另指出首版常量名写成 `MIN_INFLIGHT`，与“目标两个、旧上限优先”的公式
相反；短轨迹表中 64 核实际只有 1.5 个 chunk/worker。常量已改为
`TARGET_INFLIGHT_CHUNKS_PER_WORKER`，文档同步为目标值，不改变运行参数。

### 独立进程 A/B

基准使用真实 `run_mp + _HMMFilter._getoriginandhmm`，前后顺序逐轮交替。父进程 RSS 用
`RUSAGE_SELF` 记录，因此不把它误称为所有 child 的总内存。

第一组在顺序修复后直接运行最终生产路径，交替 5 轮：4 workers、40 条 × 100 万帧
dense fallback。旧默认中位 wall/RSS 增量为 `2.420 s / 452.2 MiB`，最终有界保序路径
为 `1.707 s / 127.3 MiB`；逐轮成对比值中位为 `0.783 / 0.268`，即 wall 约缩短
21.7%、父进程 RSS 约降低 73.2%，两种模式输出均为 174,520 bytes。4,000 条 × 1,000
帧短任务仍使用 `100/600`，最终保序路径与旧乱序默认的 wall 中位为
`1.920 s / 1.927 s`，未观察到 wall 回退；该场景的 RSS 增量受进程启动/page cache
噪声支配，不据其数值宣称收益。

第二组加入真实 `hmmlearn` Viterbi：4 workers、40 条 × 10 万帧，交替 3 轮。旧默认
`chunksize/max_inflight=100/600`，新值为 `10/80`；wall 中位
`1.847 s -> 1.562 s`，逐轮成对比值中位 `0.845`（约缩短 15.5%）。RSS 中位
`52.0 MiB -> 51.4 MiB`，成对比值为 `0.786`；RSS 波动较大，只把 wall 和任务分发改善
作为该场景的主要结论。

### 回归与生产边界

- 新增边界、参数化公式、fallback 参数和真实逆序完成测试共 7 项通过。
- `tests/test_timedoutput.py` 为 119 passed；detect/tools/CLI/ASE 辅助文件为 35 passed；
  本地、不下载轨迹的 HMM 与主流程子集为 19 passed、1 xpassed。
- 全仓库测试仍有两项环境限制：远端基准轨迹下载被当前网络沙箱拒绝，Tk GUI 在无显示
  环境中会由系统中止。两者均不位于本轮改动路径。
- 短 `rp3` 样例是 5,615/5,615 no-HMM direct，不会进入本轮 Pool 分支；其既有语义
  指纹与文本哈希仍作为回归基线，但不能验证 HMM 长轨迹调度。

生产 Slurm 必须记录新增 limits 日志、Step 2 wall、`TotalCPU/Elapsed`、MaxRSS、临时盘
峰值和 dense fallback 比例。局部 A/B 已证明旧默认会造成任务粒度过粗与父进程预取过量，
但 64 核全轨迹的最终 CPU 利用率和共享文件系统影响仍只能通过重新提交作业验收。

## 2026-08-07：冲突矩阵位压缩与 reaction 懒解码

### 红灯、方案比较与失败反馈

Step 3 的 `atomeach[N,T]` 已改为临时 memmap，但重叠标记 `conflict[N,T]` 仍以
`numpy.bool_` 保存，每个逻辑值占 1 byte。它只供 reaction 阶段按相邻帧读取，route
完全不消费，却仍把临时盘、page cache 和顺序扫描量扩大到 `N*T`。新增
`3 atoms x 17 frames` 红灯要求文件大小为 `3*ceil(17/8)=9 bytes`；修改前连续两次均为
51 bytes，稳定证明问题存在。

原型同时比较了逐布尔值、按位压缩和稀疏 conflict event 表。0.1% 冲突率时 event 表最小，
但 10% 冲突率下已增长到 39.215 MiB，扫描约 2.211 s；同规模按位矩阵固定为
6.294 MiB，扫描约 0.026 s。event 表的最坏空间仍随冲突数增长，且 reaction 需要逐帧
重建列索引，因此没有作为统一生产格式。

第一版按位矩阵虽然把文件降为 1/8，但非字节对齐边缘仍逐 atom 调用
`flatnonzero + bitwise_or.at`。100,000 x 512 的 dense 压力下，build 从约 0.320 s 退化到
6.901 s（约 21.5 倍），因此未把该结果当成完成。最终把所有选中 atom/frame 的边缘位
合成一个有界 `uint8` mask，并以一次广播 `bitwise_or.at` 写入；字节对齐的中段继续用
`packbits` 批量 OR。结构回归会在 indexed 写入中禁止逐行 `flatnonzero`，防止这一退化
重新出现。最终 indexed/edge 写入还按 overlap 密度自适应：active cell 不超过 1/8 时，
只对整个二维 mask 做一次 `flatnonzero` 并 scatter 实际位；其余情况使用有界 `uint8`
mask 广播 OR。100 万 cell 隔离反馈中，1e-6/0.1/1.0 密度的 active/global 路径中位约为
0.00060/0.00983 s、0.00450/0.00802 s、0.01510/0.00893 s，因此阈值同时避开极稀疏
全 mask scatter 和 dense index 展开。

### 最终实现

- 新增 `_PackedBoolMatrix`，文件布局为 row-major
  `(atom_count, ceil(frame_count/8))`；bit order 固定为 little endian，末字节 padding bit
  始终清零。
- matrix builder 继续复用每个不超过 `_MATRIX_WRITE_CELLS` 的现有 overlap mask，不创建
  完整逻辑矩阵。连续 span 的对齐中段用 `np.packbits`；前后边缘和 sparse frame index
  在稀疏 active-cell scatter 与 dense mask 广播 OR 之间自适应，两条路径都不逐 atom
  扫描。
- `conflict[:, frame]` 返回懒列视图。reaction 只在既有 atom scan block 内解码实际读取的
  行；非 compact 分支一次收集 changed atom 的 conflict flag，不再逐 atom 解码位。
- `_AtomFrameStore.close()` 和 reaction worker 分别管理 packed mapping 的 flush/close；
  临时文件生命周期及异常清理契约不变。
- 新增文件大小、跨 7/8/16 frame 边界、dense/indexed 混合写入、padding、懒列读取、
  reaction conflict 排除和向量化写入回归。

### 三轮交替独立进程 A/B

生产 helper 压力使用 100,000 atoms x 512 frames（51.2 million logical cells），每种
模式独立进程运行三次并交替前后顺序。sparse 场景含 400,000 个 conflict，dense 场景
含 51.2 million 个 conflict；scan 按 reaction 的 65,536-row block 遍历全部列。表中为
中位数，RSS 同时包含约 48.8 MiB 的 `atomeach` mapping，因此不是 conflict 文件的单独
驻留量：

| 场景 | 格式 | conflict 文件 | build | 全列 scan | MaxRSS |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse | bool byte | 48.828 MiB | 0.1000 s | 0.1356 s | 201.97 MiB |
| sparse | packed bit | 6.104 MiB | 0.1084 s | 0.0634 s | 166.75 MiB |
| dense | bool byte | 48.828 MiB | 0.2934 s | 0.1575 s | 208.83 MiB |
| dense | packed bit | 6.104 MiB | 0.3419 s | 0.0674 s | 164.09 MiB |

最终文件比值固定为 0.125；整体 MaxRSS 下降约 17.4%/21.4%，全列 scan 缩短约
53.2%/57.2%。build 的局部位打包成本约增加 8.4%/16.5%，没有再出现逐 atom 版本的
数量级退化。生产总 wall 是否改善仍取决于 conflict 写入比例、共享文件系统 page cache
以及 reaction 扫描占比，不能用该局部基准直接外推。

### 真实轨迹、语义与回归

只读 `rp3.lammpstrj` 以 `nproc=8`、no-HMM、molecule timeline 和 reaction event 完整
运行。最终 sparse-active 分支加入后，Step 3 molecule/matrix/route/reaction 分别为
1.158/0.144/0.285/0.030 s，总计 1.657 s；5-frame 短样例受启动和本机 cache 波动影响，
只用于正确性与生产分支取证。

HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。
`tests/test_timedoutput.py` 最终为 123 passed；molecule/Step 3/timestep/parm2cmd 子集为
11 passed、40 deselected；tools/detect 为 19 passed。Black、Isort、flake8、compileall
和 `git diff --check` 均通过。

本轮三份 conflict benchmark 脚本和真实轨迹临时输出目录均已清理；只读输入未修改，
SHA-256 仍为 `c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

生产 Slurm 仍需同时记录 conflict 临时文件峰值、Step 3 matrix/reaction wall、MaxRSS、
page fault、临时盘吞吐和 `TotalCPU/Elapsed`。位压缩消除了 conflict 的 1-byte/cell
结构浪费，但不能单独解决 molecule SMILES 或其他串行主进程阶段；完整 64 核利用率仍需
重新提交全轨迹验证。

## 2026-08-08：按进程启动方式校准 reaction 并行度

### 真实长时间轴反馈环与根因

新增只读真实输入
`ch4_o2_3000K.lammpstrj`（135,333,201 bytes，SHA-256
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`）的可重复
Step 3 harness。输入包含 10,001 帧、450 个原子；一次 Step 1/2 准备后固定复用同一
中间状态，Step 3 观测到 3,154 个 molecule、162,026 个时间范围、360,249 个原子
变化事件和 4,500,000 个 atom-transition scan value。

上一版 reaction 自适应阈值完全来自 macOS `spawn` 测量，并把每个 worker 的变化事件
目标设为 1,000,000。该规则直接用于 Linux Slurm 默认的 `fork` 后，上述真实工作量只
选择父进程串行执行。隔离复测证明两类启动方式的固定成本不能共用同一阈值：

| 启动方式 | worker | reaction wall | 平均核数 | child MaxRSS |
| --- | ---: | ---: | ---: | ---: |
| `spawn` | 1 | 0.9885 s | 0.995 | 0 MiB |
| `spawn` | 2 | 2.8907 s | 3.052 | 187.4 MiB |
| `fork` | 1，3 轮中位 | 1.0048 s | 0.994 | 0 MiB |
| `fork` | 7，3 轮中位 | 0.4343 s | 5.941 | 32.4 MiB |

`spawn` 的两个 worker 因解释器和依赖重新导入约慢 2.9 倍；`fork` 的 7 个 worker 则把
wall 缩短约 56.8%。route 子阶段也交叉验证了这个平台差异：同一 450 × 10,001
输入在 `spawn` 下父进程约 0.486 秒、16 workers 约 7.715 秒；在 `fork` 下 8/16
workers 约 0.167/0.208 秒，均快于父进程。route 的 Linux 路径已经能够并行，故本轮
没有为提高利用率而改写其 worker 协议或写入路径。

### 最终调度与确定性修复

- `fork` 每 50,000 个已观测变化事件增加一个 reaction worker；`spawn`、`forkserver`
  和其他启动方式继续使用保守的 1,000,000 阈值。500,000,000 scan values 的兜底阈值
  保持不变，避免在缺少实际收益证据时扩大稀疏扫描并行度。
- worker 数仍受用户请求核数和 transition 数限制；变化事件指标缺失时保持旧调用的
  回退行为。日志现在同时报告实际 worker 数和 multiprocessing start method，便于
  Slurm 现场确认是否命中 `fork` 分支。
- Code review 指出 no-event 路径原先以 `unordered=True` 的完成顺序更新 `Counter`，
  `most_common()` 会让同频 reaction 的文本顺序随 worker 完成次序变化。并行阈值降低后
  会在原本串行的 Linux 工作量上暴露该问题。最终 no-event summary 与 timed-output
  路径统一按 `(-count, reactant, product)` 排序，不牺牲 worker 的乱序执行；回归显式
  反转 transition 完成顺序并逐字节校验输出。

### 完整 Step 3 A/B、资源代价与结果等价性

固定 `fork`、请求 16 核，分别强制 reaction 为 1/7 workers，前后交替各运行三次。
表中为中位数；检测与 HMM 准备不计入 wall：

| 指标 | reaction 1 worker | reaction 7 workers | 变化 |
| --- | ---: | ---: | ---: |
| Step 3 wall | 2.0379 s | 1.5322 s | -24.8% |
| reaction stage | 1.1142 s | 0.6344 s | -43.1% |
| Step 3 平均核数 | 1.226 | 2.954 | +141% |
| CPU seconds | 2.498 s | 4.595 s | +84% |
| root MaxRSS | 215.8 MiB | 222.1 MiB | +2.9% |
| 单个 child MaxRSS | 27.7 MiB | 32.8 MiB | +18.4% |

该修复明确用更多 core-seconds 换取更短 wall；它不是无条件启动全部请求核数。短样本中
molecule、matrix、route 和单 writer 仍有串行区段，因此 Step 3 平均核数不会接近 16；
全轨迹能否继续扩展到 64 核仍需生产节点验证。

三轮串行/并行的 `.reactionabcd` SHA-256 均为
`af8303960eaa4118059d6d9211cf056e19486938cedd07998021a3bc1d7f1654`；完整运行的
`.route` SHA-256 均为
`af09a4386b071bc6308bf6f3cc6f1828878fdbd99b0de6acb86844d8534b77a4`，HDF5 语义
指纹均为
`2174c01055c46e4680ef50c536dae83d0a2438f7f67606ba5fe24180a96d2cde`。不同完成顺序
可能改变 HDF5 内部压缩布局和文件字节数，但 manifest 语义、文本产物和 reaction 计数
保持一致。

确定性排序加入后还重新运行了 5 帧 `rp3.lammpstrj` 全流程。HDF5 语义指纹仍为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`，因此没有
改变既有小轨迹的精确产物。

回归结果：`tests/test_timedoutput.py` 为 126 passed；molecule/timestep/parm2cmd 子集为
11 passed、40 deselected；tools/detect 为 19 passed。Black、Isort、flake8（按项目
忽略 E203/E501/W503）、compileall 和 `git diff --check` 均通过。Code review 的
Standards 轴无 P0-P2；Spec 轴发现的输出确定性和本地日志两个 P2 均在本节闭环。

生产复跑必须记录 start method、实际 reaction worker 数、各 Step 3 子阶段 wall、
`TotalCPU/Elapsed`、MaxRSS 和临时盘吞吐。50,000 是当前真实输入上的安全起点，不是
不同 CPU、NUMA 和共享文件系统上的永久最优值；若 64 核节点出现调度或 I/O 饱和，应
根据同一日志指标继续校准，而不是只以 CPU 百分比判断成功。

## 2026-08-08：Linux `fork` 下廉价 SMILES 分批并行

### 先排除 range/HDF5 写入 Python 路径

5% 作业记录包含 12,501 帧、110,301 个 molecule 和 49,047,393 个 molecule range，
总 wall 为 3 小时 38 分 55 秒。为避免继续凭日志增长速度猜测，本轮先建立与现有
`TimedOutputStore` 相同接口的反馈环：10,000 个 molecule、4,440,000 个碎片化 range
中，range 生成约 0.084 秒、writer 约 0.287 秒，完整受测路径在 `cProfile` 下约
0.722 秒；既有 49,047,393-range 压力环的 writer 约 2.05 秒、749 个有界 batch、
约 80 MiB RSS 增量。因此本机证据不支持“range 扩容/写入 Python 路径本身造成数小时
串行 wall”的假设。共享文件系统实际吞吐仍须由 Slurm 现场指标验证，不能由本机结果
排除。

### 剩余瓶颈与失败候选

从只读 `rp3.lammpstrj` 的 Step 1/2 中间记录提取 5,615 条真实 SMILES 输入，共
1,905,017 个压缩结构字节，平均 339.3 bytes。重复到 112,300 条后与 5% 作业的
molecule 数接近。上一版 `_smiles_worker_count()` 的阈值全部来自 macOS `spawn`
反馈，并以平均结构复杂度限制 worker；廉价记录无论总量多大都保持父进程串行。

直接把 Linux `fork` 的 worker 数提高并不能解决问题。保持 `chunksize=1` 时，
112,300 条记录的 2/4/8/10/16-worker 结果都慢于串行，2 workers 也约为 5.38 秒，
worker 越多越差。该反例确认主要固定成本是逐记录 Pool/IPC 与 ordered spool 调度，
而不是缺少可运行进程；只提高 CPU 利用率会产生负加速。

把廉价记录改为 `chunksize=64` 后，112,300 条隔离 SMILES 在 2 workers 下稳定比相邻
串行测量快约 22%；280,750 条在 4 workers 下快约 43%，8 workers 又开始回退。完整
`_printmoleculename()` 路径还包含父进程二次解码、`.moname` 格式化/写入、紧凑名称表和
严格有序恢复，112,300 条强制 A/B 由 3.922 秒降至 3.285 秒，280,750 条由 9.301 秒
降至 6.872 秒；输出哈希分别保持
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec` 和
`64552e321b0a5dcc2849974ca41caf3b3052a1aa057443cd9a53ab94bf9ec5ee`。

### 最终调度规则与交替 A/B

- `spawn`、`forkserver` 和未知启动方式保留既有复杂度/总工作量规则；真实廉价记录仍
  走零 Pool 的父进程路径。
- `fork` 下，廉价记录每累计约 24 MiB 压缩结构工作增加一个候选 worker，最多 4 个；
  仍受原 3 MiB/worker 的总工作量下限、用户请求核数和 molecule 数限制。5,615 条
  小样本继续串行，112,300/280,750 条分别自动选择 2/4 workers。
- 只有平均压缩结构不超过 512 bytes 的 `fork` 任务才使用 64-record chunk；复杂结构
  保持 `chunksize=1`。在途输入上限为 `2 × workers × chunksize`，ordered spool 仍负责
  严格恢复顺序和异常清理。

对 112,300 条完整 molecule-name pipeline 交替执行 `spawn` 串行和 `fork` 自动调度各
三轮，结果如下（中位数）：

| 指标 | 串行 | `fork` 自动调度 | 变化 |
| --- | ---: | ---: | ---: |
| wall | 6.193 s | 4.609 s | -25.6% |
| 平均核数 | 0.978 | 2.479 | +153.5% |
| CPU seconds | 5.919 s | 11.427 s | +93.1% |
| root MaxRSS | 186.4 MiB | 185.6 MiB | -0.5% |
| 单个 child MaxRSS | 0 MiB | 27.4 MiB | +27.4 MiB |

六轮 `.moname` SHA-256 均为
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec`。该修复与
reaction 调度一样，用更多 core-seconds 换取更短 wall；上限为 4 workers 是当前真实
分布下父进程解码/格式化开始占主导后的保守值，不代表完整 Slurm 节点的永久最优值。

新增回归覆盖 cheap-fork 分批调用、`spawn`/复杂结构不分批、在途上限，以及
`spawn`/`fork` 下 1/2/4-worker 自适应选择。`tests/test_timedoutput.py` 为 136 passed；
molecule/Step 3/timestep/parm2cmd 子集为 11 passed、40 deselected；tools/detect 为
19 passed。Black、Isort、flake8（忽略项目既有 E203/E501/W503）、compileall 和
`git diff --check` 均通过。

生产复跑除既有 reaction 指标外，还必须保存 `SMILES worker count`、`SMILES pool`、
ordered spool、molecule-stage wall、`TotalCPU/Elapsed` 和 MaxRSS。若 Linux 节点没有
报告 `fork`，或自动调度仍为 1 worker，本地 A/B 不能外推；若 4 workers 后 molecule
阶段仍主导，则下一轮应先剖析父进程二次解码/格式化，而不是继续无证据地增加 worker。

## 2026-08-08：压缩记录读取消除逐字段位置查询

### 红灯与 profile 归因

继续复用上一节从只读 `rp3.lammpstrj` 提取并重复的 112,300 条真实结构记录，运行完整
`_printmoleculename()`，而不是只测一个解压函数。修复前连续三轮父进程 CPU 占总 CPU
的 49.7%/49.6%/48.6%，中位 49.6%；wall 中位 4.740 秒。反馈命令以 45% 为固定红线，
同时要求 `.moname` SHA-256 为
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec`，因此能够同时捕获
父进程失衡和结果变化。

父进程 profile 依次检查结构解码、通用文本格式化、名称表、WriteBuffer 和 ordered
result 消费。名称表 append 约 0.11 秒、WriteBuffer append 约 0.28 秒，不是主因；
`_getatomsandbonds()` 和通用 formatter 分别累计约 2.74/2.15 秒。更直接的异常是
`_iter_compressed_record_fields()`：worker 输入 reader 与父进程 reader 都为每条记录
读取四个字段，每个字段在已经持有 `fstat` 文件长度的情况下仍调用一次
`BufferedReader.tell()`，总计 898,400 次。带 profiler 时这些位置查询自身累计约
1.53 秒。

新增回归用两条四字段真实格式记录和非零文件起始 offset 包装生产 reader。修改前调用
`tell()` 8 次，红灯要求只查询一次；修改后 reader 从该初始位置开始，随 header read、
payload read 和 seek skip 维护局部整数位置，仍用文件总长检查每个 payload 边界，并按
调用者指定顺序返回字段。

### 隔离扫描与完整路径 A/B

隔离反馈环对同一 65,823,620-byte 文件执行两遍 `(0, 1, 2)` 字段扫描，共读取
224,600 条记录、76,200,680 个选中字段字节；legacy/current 交替七轮：

| 指标 | 每字段 `tell()` | 局部位置追踪 | 变化 |
| --- | ---: | ---: | ---: |
| wall 中位 | 0.6666 s | 0.4367 s | -34.5% |
| CPU 中位 | 0.6501 s | 0.4303 s | -33.8% |

重新运行最初三轮完整 molecule-name 反馈命令后，45% 红线转绿：

| 指标 | 修改前中位 | 修改后中位 | 变化 |
| --- | ---: | ---: | ---: |
| wall | 4.740 s | 4.240 s | -10.5% |
| 父进程 CPU | 5.316 s | 4.085 s | -23.2% |
| 父进程 CPU 占比 | 49.6% | 38.4% | -11.2 pct |
| 平均核数 | 2.248 | 2.433 | +8.2% |

三轮输出哈希仍全部为上述基线。38.4% 已接近一个父进程与两个计算 worker 同时活跃时的
理论三分之一，不再是父进程与全部 worker CPU 相当的失衡状态。该 reader 也被 matrix
字段选择路径复用，因此不增加新缓存、线程或常驻内存。

### 被否决的后续候选

- 专用 `.moname` formatter 在 profile 下减少递归调用，但三组无 profiler 的交替完整
  A/B 分别轻微回退、回退和持平，没有稳定 wall 收益，未进入源码。
- reader 加速后重新强制 2/3/4 workers，三轮 wall 中位为 4.523/5.153/6.125 秒；更多
  worker 继续增加 IPC 和 core-seconds，112,300 条的自动 2-worker 选择保持不变。
- `chunksize=64/128/256` 在系统负载变化中出现相反的成对 wall 方向；较大 batch 还会
  放大 head-of-line 和在途窗口，因此没有用不稳定中位数覆盖已验证的 64。

最终 `tests/test_timedoutput.py` 为 138 passed；molecule/Step 3/timestep/parm2cmd 子集为
11 passed、40 deselected；tools/detect 为 19 passed。Black、Isort、flake8（忽略项目
既有 Black overload stub 和裸 `except` 规则冲突）、compileall 与
`git diff --check` 均通过。生产 Slurm 仍需用真实共享文件系统复核 molecule wall、
`TotalCPU/Elapsed`、实际 SMILES worker 数和临时盘吞吐；本地 page-cache 结果不能代替
集群 I/O 证据。

## 2026-08-08：Detect 单帧调度与跨平台启动成本控制

### 证据边界与新的反馈环

本地已经没有 `pct_05` 或全轨迹的原始 HDF5、Slurm stdout/stderr 和阶段计时；现存
`docs/oom-optimization-experiment-report.md` 只能确认 1%/2%/5% 作业总耗时分别约为
46 分钟、1 小时 32 分钟和 3 小时 39 分钟，不能把总耗时归因到某个 Step。因此本轮没有
用总作业时间继续猜测 Step 3，而是对仍可复现的只读
`/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 逐阶段计时。经过前几轮 Step 2/3
优化后，这份 5 帧、每帧 12,326 原子的真实轨迹中，Detect 已成为新的主要本地瓶颈。

修改前的第一组反馈结果如下；三种运行均生成 5,615 条 molecule 记录，SHA-256 都为
`840d60e0d5a3ac1659b6bef3f13aea99d20d15ebbfd5ca7c63d24120ac62cf0b`：

| 调度方式 | Detect wall | 平均核数 | 观察 |
| --- | ---: | ---: | --- |
| `nproc=1` | 8.491 s | 0.997 | 父进程直接执行 |
| 默认 `spawn, nproc=8` | 16.302 s | 2.855 | 启动多个重依赖 worker，反而回退 |
| 强制 `fork, nproc=8` | 8.722 s | 0.991 | 仍几乎只有一个 worker 忙碌 |

根因不是 Open Babel 单帧计算不能并行，而是 `_readinputfile()` 沿用了 `run_mp()` 的
`chunksize=100` 默认值。5 个 frame 被装进同一个 Pool task，一个 worker 串行处理全部
帧，其余 worker 等待。`spawn` trace 还显示 Detect 实际创建了两个 Pool：帧识别 Pool
约 9.560 秒、45.38 child CPU seconds；随后仅压缩分子帧数组的第二个 Pool 又花约
7.412 秒、42.61 child CPU seconds。第二个 Pool 在这个规模上几乎全是解释器和重依赖
导入成本。

把 `chunksize` 单独改为 1 后，`fork` wall 降到 2.895 秒、平均 4.27 核，但 molecule
文件哈希变为 `9eef...`。这是因为原路径使用 unordered 完成顺序更新
`d[molecule]`，不同 frame 的首次完成次序会改变 molecule ID。随后把消费改为保序，
wall 仍为 2.900 秒左右，且恢复上述精确哈希。这一红灯排除了“用不稳定 ID 顺序换速度”
的做法。

### 最终实现

- frame detection 固定为 `chunksize=1`，`max_inflight=2 * workers`，并按输入 frame
  次序消费结果；既能分发昂贵单帧，也不会无界预取整帧文本或改变 molecule ID。
- `fork` 继续保留用户请求的 worker 数；复制写时共享已导入模块，启动成本低。
- `spawn`/`forkserver` 按 `ceil(total_input_bytes / 8 MiB)` 的经验门槛计算实际 worker，
  并受用户请求数约束。当前真实输入及重复 2/4/8 次时，约 1.99/3.97/7.95/15.90 MiB
  分别选择 1/1/1/2 workers；小输入直接走 `run_mp(nproc=1)`，不创建 Pool。
- 非 `fork` 下的 molecule-frame compression 固定在父进程执行，避免为短小数组再次
  启动一组重依赖 worker；`fork` 下保留原并行路径。
- 日志明确输出 start method、实际 Detect worker、`chunksize` 和在途上限，生产作业
  可以直接确认是否命中 Linux `fork` 路径。

8 MiB 是基于当前真实帧成本的保守启动门槛，不是硬件无关常数。重复 8 次、40 帧、
约 15.90 MiB 的输入用两个 `spawn` frame workers 时 wall 为 71.133 秒；child CPU 为
136.21 秒，说明缩短 wall 仍会增加 core-seconds。重复 4 次、约 7.95 MiB 的边界样本在
不同系统负载下出现相反方向，因此最终没有把门槛降低到 4 MiB。

### Linux-like `fork` 配对 A/B

为避免不同系统负载污染结论，对相同 5 帧输入、请求 8 workers，强制 `fork` 并交替运行
旧 `chunksize=100` 与最终单帧保序调度，各取三轮中位：

| 指标 | 旧调度 | 最终调度 | 变化 |
| --- | ---: | ---: | ---: |
| Detect wall | 15.541 s | 4.952 s | -68.1% |
| 平均核数 | 0.990 | 4.544 | +359% |
| CPU seconds | 15.39 s | 22.50 s | +46% |

新路径用更多 core-seconds 换取明显更短 wall；这正是需要缩短生产总时长时的显式权衡，
并不意味着 64 workers 在所有帧规模上都最经济。旧/新六轮 molecule 文件哈希完全一致。

### 端到端与回归

最终代码对只读输入的副本请求 `nproc=8`，本机 `spawn` 按 1.987 MiB 输入自动选择父进程，
日志报告 `chunksize=1, max_inflight=2`，Step 1 为 14.010 秒。该单次 wall 受当时机器负载
影响，只用于验证自动分支，不与不同负载下的 8.491 秒串行基线作加速比。完整 no-HMM、
molecule timeline 和 reaction-event 流程成功结束，并保持既有精确结果：

- HDF5 语义指纹：
  `3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；
- `.route`、`.reactionabcd`、`.reaction`、排序 `.moname` SHA-256 分别为
  `f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
  `2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
  `ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8`、
  `2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`；
- 计数仍为 5 frames、5,615 molecules、6,076 ranges、383 reaction types 和
  469 compressed reaction-event rows。

新增 8 个定向回归覆盖单帧/有限在途/保序调用、非 `fork` compression 回退，以及
`fork`、`spawn`、`forkserver` 的 worker 选择。最终 `tests/test_detect.py` 为 25 passed；
`tests/test_timedoutput.py` 为 138 passed；molecule/Step 3/timestep/parm2cmd 子集为
11 passed、40 deselected；`tests/test_tools.py` 为 2 passed。Black、Isort、flake8
（忽略 E203/E501/W503）、compileall 和 `git diff --check` 均通过。

生产 Slurm 验证仍未完成。复跑时必须保存 Detect 日志中的 start method/实际 workers，
Step 1 与各 Step 3 子阶段 wall、`TotalCPU/Elapsed`、MaxRSS 和共享文件系统吞吐；只有
生产全轨迹能判断 64 核是否在 frame detection、SMILES、route 和 reaction 各阶段达到
合理平衡。

本轮 Detect profiler、重复输入、Matplotlib cache 和端到端输出目录均已清理；这些文件
不可恢复但可由只读输入重新生成。原始 `rp3.lammpstrj` 未修改，最终 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

## 2026-08-08：Detect 每帧元数据紧凑化

### 真实调用点的 retained-memory 红灯

上一节解决 frame worker 调度后，继续检查会随完整轨迹帧数线性增长的 parent 常驻对象。
反馈环直接调用生产 `_Detect._readinputfile()`，固定只有一个 molecule，只增加分析帧数；
用 `tracemalloc` 在 Detect 对象构造后开始计数，并在 `_readinputfile()` 返回、GC 完成后
读取 retained/peak。门限固定为 160 B/帧，能够捕获两份 per-frame Python 容器的放大，
而不是只检查“没有 OOM”。

修改前三次 100,000-frame 运行稳定为约 224.77 B/帧，retained 22,477,213--
22,477,310 bytes，peak 25,299,903--25,299,954 bytes，反馈命令退出码为 1。50,000 帧
仍为 224.69 B/帧，10,000 帧为 178.13 B/帧；差异来自 Python dict 的容量台阶，但两者
均稳定越过红线。

变量探针逐项排除了相邻工作：

- 所有 frame 都返回空 molecule 后仍为 224.68 B/帧，说明同一 molecule 的
  `array("Q")` 帧序列不是 retained 主因。
- 把 timestep 和 source-frame 都固定为小整数后降至 192.71 B/帧；独立 Python 整数约
  贡献 32 B/帧，但容器本身仍超过门限。
- `tracemalloc` 将约 85 B/帧直接归到 `framesource[step] = (source_id,
  source_frame)`；另一个 timestep dict 自身在 10,000 帧时约占 288 KiB。
- 全仓库消费者只使用 `len()`、连续整数索引和 `framesource.get()`；没有 `.items()`、
  稀疏键写入或需要保持 dict 插入顺序的调用。

因此正确假设是：连续 `step` 被同时保存为两个 dict key，`framesource` 又为每帧分配
tuple，造成了不必要的对象放大。

### 紧凑表示与 provenance 块路径

- timestep 改为 signed 64-bit `array("q")`，与 timed-output HDF5 的 `int64` 契约
  一致，只保留 8-byte value，不再重复保存连续 step key。
- `_FrameSourceMap` 继续提供 Mapping 的 `len`、整数索引、迭代、相等比较和 `.get()`；
  内部只为 arithmetic provenance segment 保存 analyzed start、source ID 和第一个
  source-frame。正常情况下 segment 数接近有实际选中 frame 的输入文件数，而不是总帧数。
- 同一 source 的 source-frame 若不符合 `stepinterval` 等差关系，会自动开始新 segment，
  因此紧凑化不依赖输入一定规则；乱序/缺号 analyzed step 会立即报错，不会静默错配。
- segment append 缓存上一个 source 和下一个预期 frame，避免每帧回读数组并重新乘法；
  50 万帧构建中位由第一版 0.104 秒降至最终约 0.072 秒。
- timed-output provenance writer 对该 Mapping 使用 `fill_arrays()` 跨 segment 直接填充
  `uint32 source_id`/`uint64 source_frame` 块，并将连续 timestep sequence 作为 NumPy
  view 分块写入；外部传入的普通 dict 继续走原兼容路径。
- Step 1 新日志报告 frame 数、紧凑 payload MiB 和 provenance segment 数，生产现场可
  直接检查异常 segment 爆炸。

### 内存与预处理 A/B

最终实现对同一 100,000-frame 反馈环连续三轮均转绿：

| 指标 | Python dict/tuple | 紧凑 sequence/segment | 变化 |
| --- | ---: | ---: | ---: |
| retained bytes | 22,477,310 | 817,512 | -96.4% |
| retained bytes/frame | 224.773 | 8.175 | -96.4% |
| tracemalloc peak bytes | 25,299,954 | 3,640,282 | -85.6% |
| provenance segments | 100,000 tuples | 1 segment | -99.999% |

按 5% 报告中的 12,501 帧线性推算，全轨迹约为 250,000 帧；仅这两份元数据的 retained
Python allocation 可由约 53.6 MiB 降到约 2.0 MiB。该数字是受控反馈环外推，不是 Slurm
RSS 实测，也不能解释原作业全部 52 GB 峰值。

为检查内存优化是否拖慢 HDF5 前处理，对 500,000 帧交替运行旧 dict 和最终紧凑路径七轮：

| 元数据阶段中位 | dict 路径 | 紧凑路径 | 变化 |
| --- | ---: | ---: | ---: |
| 构建 | 0.0305 s | 0.0717 s | +0.0412 s |
| provenance 块扫描 | 0.0758 s | 0.00093 s | -98.8% |
| 构建 + 扫描 | 0.1063 s | 0.0726 s | -31.7% |

数组 append/segment 检查本身略慢于 dict assignment，但 HDF5 前的向量化块生成收回了
成本；绝对开销相对于每帧化学识别仍很小。

### 契约与最终验证

新增/扩展回归覆盖生产 `_readinputfile()` 返回紧凑 timestep、Mapping 行为、跨 segment
块填充、乱序拒绝、多输入文件加全局 `stepinterval`，以及紧凑 provenance 写入 HDF5 后
的 `source_id/source_frame/timestep` 精确值。最终结果：

- `tests/test_detect.py`：26 passed；
- `tests/test_timedoutput.py`：138 passed；
- molecule/Step 3/timestep/parm2cmd 子集：11 passed、40 deselected；
- `tests/test_tools.py`：2 passed；
- Black、Isort、flake8（忽略 E203/E501/W503）、compileall 和 `git diff --check`
  均通过；当前环境没有 Ruff，因此没有宣称 Ruff 结果。

最终源码重新运行只读 `rp3.lammpstrj` 副本的完整 no-HMM、molecule timeline 和
reaction-event 流程。日志报告 5 frames、0.000 MiB compact metadata、1 provenance
segment；HDF5 frame/source fingerprints 与整体语义指纹继续为既有基线，其中整体为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`。`.route`、
`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。

生产复跑应同时保存新日志中的 frame metadata MiB/segment 数和 MaxRSS；若 segment 数
接近 frame 数，说明输入 provenance 不规则，需要结合 source 边界检查，而不能直接套用
本轮“一段/文件”的内存外推。

本轮 metadata memory/scan harness、Matplotlib cache 和两份端到端输出已清理；它们不可
恢复但可由只读输入重新生成。原始 `rp3.lammpstrj` 未修改，SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

## 2026-08-08：Detect molecule-frame 索引自适应位宽

### 调用点红灯与归因

上一轮已压缩每帧 timestep/provenance，但 `_Detect._readinputfile()` 仍为每个 molecule
保存固定 `array("Q")` 的 frame occurrence。完整轨迹若有数千万个 occurrence，仅数组
payload 就会占数百 MiB，并在压缩临时记录时产生同宽 NumPy/pickle 数据。

反馈环直接调用生产 `_Detect._readinputfile()`，固定 1,000 帧、每帧 1,000 个相同
molecule key，共 1,000,000 occurrence，并在调用返回后记录 `tracemalloc` peak、wall、
实际 array typecode、压缩记录哈希和解码结果。修改前固定 `Q` 的稳定 peak 约为
8,400,061 bytes（8.400 B/occurrence），超过 6 B/occurrence 红线；wall 中位约
0.0706 秒，内部记录 SHA-256 为
`ae9172637c2e7bd11d2ea9785b1b6471267a7e6ef0198bf208f4a262a6b049d9`。

控制变量显示瓶颈来自 occurrence payload，而不是外围对象：500 molecule × 1,000 帧与
1,000 molecule × 500 帧均约为 8.40--8.45 B/occurrence；跳过压缩后仍为
8.375 B/occurrence。因此没有通过调整 dict 或压缩 worker 数掩盖这条常驻内存来源。

### 最终实现与被否决方案

- 新 molecule 从 native unsigned-short `array("H")` 开始；append 真正溢出时，才按
  `B/H/I/Q` 顺序选择运行时 `itemsize` 更大的无符号 array 并复制一次。常见平台上即
  `H -> I -> Q`，65535/65536 边界由生产 append 路径回归覆盖。
- 同宽候选会被跳过，边界测试也由运行时 `array(...).itemsize` 推导，不假定所有平台的
  C `unsigned short/int` 必然分别为 16/32 位。这一项修复了 standards code review
  发现的跨平台 P2。
- `_compressvalue()` 不再强制转回 `uint64`，而是保留实际 NumPy dtype。HMM 的
  `idx_to_signal`、no-HMM direct path、Matrix 和 Path 均按通用整数 ndarray 消费；新增
  回归证明同一 compact/uint64 frame block 生成完全相同的 HMM origin 和 filtered signal。
- 初版从 `array("B")` 起步会在第 256 帧让本反馈环的 1,000 个 molecule 分别触发一次
  Python `OverflowError`；最终 payload 与 `H` 相同，但 wall 增至约 0.090--0.094 秒，
  因而否决。`H` 起步避免短/中等轨迹的集中异常，同时仍比旧 `Q` 节省 75% raw payload。
- 保留 `H` 但压缩时强制 `uint64` 的中间版本 wall 约 0.078--0.082 秒、peak 约
  2.254 B/occurrence；保留 compact dtype 后恢复到接近旧路径的 wall，并进一步降低 peak。
- storage 日志的 type/count/payload 统计合并成一次 value 扫描，避免为了可观测性对大量
  molecule array 重复遍历三次。

### 内存与 wall A/B

最终实现重复五轮；表中为中位数。wall 的约 +1.6% 小于本机短任务波动，不视为可测性能
回退；主要收益来自常驻 payload、压缩输入和 no-HMM 直写判断都保持窄 dtype。

| 指标 | 固定 `array("Q")` | 自适应 `H/I/Q` + compact 压缩 | 变化 |
| --- | ---: | ---: | ---: |
| `tracemalloc` peak | 8,400,061 B | 2,204,764 B | -73.75% |
| peak / occurrence | 8.400 B | 2.205 B | -73.75% |
| 100 万 occurrence raw payload | 7.629 MiB | 1.907 MiB | -75.0% |
| `_readinputfile()` wall 中位 | 0.0706 s | 0.0717 s | +1.6%，无可测回退 |

最终五轮内部记录哈希稳定为
`237eb308436986dbb8ea7f7d9d4094d97e4dd58e055a647648e2cdaf089790ee`。它与旧哈希不同是
pickle 中的内部 dtype 变窄所致；解码后的 frame 值、顺序和 HMM signal 由回归锁定。对约
250,000 帧的生产轨迹，跨过 65,535 帧后仍活跃的 molecule 会扩为常见平台的 4 B/index，
较旧 `Q` 减半；此前结束的 molecule 继续保持 2 B/index。该比例只描述 array payload，
不能直接当作 Slurm MaxRSS 的降幅。

### 回归与真实输出

最终本地结果：

- `tests/test_detect.py`：31 passed；覆盖生产 append 扩宽、运行时位宽、compact 序列化和
  compact/uint64 HMM signal 等价；
- `tests/test_timedoutput.py`：138 passed；
- `tests/test_tools.py`、`tests/test_ase.py` 与 reaction/timestep/molecule/parm2cmd 子集：
  23 passed、40 deselected；
- Black、Isort、flake8（忽略 E203/E501/W503）、compileall 与 `git diff --check` 通过；
  当前环境没有 Ruff，因此没有宣称 Ruff 结果。

最终源码重新处理只读 `rp3.lammpstrj` 副本的完整 no-HMM、molecule timeline 和
reaction-event 流程。Step 1 日志报告 22,103 occurrence、0.042 MiB payload、5,615 个
`H` molecule；no-HMM 的 5,615/5,615 molecule 全部 direct，避免 3.058 MiB molecule
copy。验证器给出既有计数：5 frames、5,615 molecules、6,076 ranges、383 reaction
types、469 compressed reaction-event rows。HDF5 frame/source/semantic fingerprints 分别为：

```text
fc055b4e759f0f45a51caffc2e728fb6545253d15750afb48b694a7e4a0127ae
e7101d8e763c4fd9da3244c09ca57523daaac7401a4f3ede2a5911e50b75c909
3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb
```

`.route`、`.reactionabcd`、`.reaction` 和排序 `.moname` SHA-256 仍分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`2d64c0d01f72688e53c25c57fbe73d0ca47d481caa907ba1c8381f638e5429c6`。输入副本与原文件
SHA-256 均为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

两轴 code review 在修复跨平台测试 P2、补齐本节证据与 handoff 后，未发现剩余实现层
P0--P2。生产 Slurm 验证仍未完成；全轨迹复跑应保存 typecode 分布、occurrence payload、
Step 1/Step 3 子阶段 wall、`TotalCPU/Elapsed`、MaxRSS、临时盘峰值和共享文件系统吞吐。

本轮 occurrence-memory harness、Matplotlib/uv 临时 cache 和端到端输出目录已清理；这些
生成文件不可恢复，但可由只读输入重新生成。原始轨迹未修改。

## 2026-08-08：Detect molecule-frame 压缩移除进程 IPC 并自适应双线程

### fork 红灯与根因

上一轮缩窄 occurrence dtype 后，第二次 molecule-frame compression 在 Linux-like
`fork` 下仍沿用 `detect_nproc` 个进程；`run_mp` 默认又把最多 100 个 array 组成一个
chunk，并允许约 `150 * nproc` 个输入在途。压缩本身很短，但父进程必须 pickle array、
通过 pipe 发送给 child，再接收压缩 bytes。该路径把共享内存中已经存在的数据复制成 IPC
payload，worker 数越多，累计 CPU 和在途内存反而越高。

反馈环直接调用生产 `_Detect._compressvalue()`，固定 256 个 molecule、每个 131,072 个
`uint32` frame index，总输入 128 MiB，并对父进程与全部 child 的 USS 求和。旧 8-worker
`fork` 默认路径三轮中位为：

- wall 约 0.638 秒；
- CPU 约 0.914 CPU-s；
- baseline 以上额外 unique-memory peak 约 151.56 MiB；
- 压缩输出哈希在同组串行/并行比较内一致。

同一输入只在父进程流式压缩时，额外 unique-memory peak 约 4--6 MiB，且 wall/CPU 都更低。
把 process Pool 改成 `chunksize=1` 和有限在途可以降低峰值，但 2/4/8 个 process 在
64--512 MiB 输入上仍没有稳定快过父进程，且 core-seconds 明显更高。因此最终实现完全
禁止这一阶段通过 process IPC 传输 array；这与 frame detection 的多进程 Pool 无关。

### 双线程交叉点与最终选择

pickle/LZ4 的大块压缩会释放 GIL，因此继续测试了共享同一 array 的有界线程。线程结果按
提交顺序消费，最多保留 `2 * workers` 个 future；2/4/8 threads 在大块输入上收益接近，
所以只保留 2 threads、4 in-flight。交叉点结果如下，均为交替运行中位且输出哈希一致：

| 输入形态 | 父进程串行 | 2 threads | 结论 |
| --- | ---: | ---: | --- |
| 44,920 B，总计 5,615 个小 array | 约 0.020 s | 约 0.065--0.076 s | 短任务保持串行 |
| 64 MiB，平均 64 KiB/array | 0.0389 s | 0.0402 s | 无收益，保持串行 |
| 64 MiB，平均 128 KiB/array | 0.0392 s | 0.0250 s | thread wall -36.2% |
| 64 MiB，平均 256 KiB/array | 0.0379 s | 0.0240 s | thread wall -36.7% |
| 128 MiB，平均 512 KiB/array | 0.0797 s | 0.0471 s | thread wall -40.9% |

大块记录在总量 8 MiB 时没有稳定收益，16 MiB 起才开始摊薄 executor 固定成本。最终调度
因此同时要求：请求至少 2 核、至少 2 个 molecule、总 frame-index payload 不小于
16 MiB、平均 payload 不小于 128 KiB；满足时使用 2 个 parent threads，否则使用
`run_mp(1)` 的零 Pool 串行路径。阈值判断只看已有统计，不额外扫描 array。

最终生产 helper 对 128 MiB 输入交替五轮的中位结果为：

| 指标 | 父进程串行 | 2 bounded threads | 变化 |
| --- | ---: | ---: | ---: |
| wall | 0.07971 s | 0.04712 s | -40.9% |
| CPU | 0.07958 CPU-s | 0.08662 CPU-s | +8.9% |
| 输出 SHA-256 | `db5d9d67…bb46e` | `db5d9d67…bb46e` | 相同 |

启用进程采样器的三轮中位中，串行和 2-thread 的额外 unique-memory peak 分别为
4.39 MiB 与 5.73 MiB；线程只多 1.34 MiB，却比旧默认 8-process `fork` 的 151.56 MiB
降低 96.2%。采样器存在时 thread wall 仍比串行低约 38.8%。该基准说明线程窗口有界，
不表示生产全轨迹的总 RSS 会按相同比例下降。

### 契约、回归与真实输出

- `_bounded_thread_map()` 最多提交 4 个 future、严格按输入顺序 yield，并在异常或调用方
  提前结束时取消尚未开始的 future；调用处用 `ExitStack + closing` 显式关闭原始生成器，
  下游文件写入失败也会立即进入 cancel/executor shutdown，而不依赖 traceback 释放或 GC。
  executor context 最多只需等待两个正在运行的压缩。
- molecule key 和压缩 frame block 继续通过 `zip_longest` 保序配对，数量不一致仍立即
  报错；线程只读独立 array，不修改 `_Detect` 状态。
- 日志会报告串行或 2-thread 选择、总 payload、平均 KiB/molecule、最大在途数和包含
  临时文件写入的 compression/write wall，便于生产现场确认阈值是否命中。
- 定向红灯在实现前为 9 failed/2 passed；code review 的异常清理回归修复前另有 1 failed。
  最终 `tests/test_detect.py` 为 44 passed，覆盖 fork/spawn 的串行与线程分支、阈值上下
  边界、有界提交、保序、真实 executor、下游写入失败和提前关闭时的 pending cancel。
- `tests/test_timedoutput.py` 为 138 passed；`tests/test_tools.py` 加 `tests/test_ase.py`
  为 12 passed；reaction-event/timestep/molecule/parm2cmd 子集为 13 passed、38 deselected。
- Black、Isort、Ruff、flake8（忽略 E203/E501/W503）和 `git diff --check` 均通过。

最终代码以强制 `fork`、8 核重新处理 5 帧 `rp3.lammpstrj`。该小样本只有 0.042 MiB
payload、平均约 7 B/molecule，正确选择父进程串行；资源清理修复后的最终复跑中，
compression/write 为 0.026 秒，Step 1 为 2.793 秒，Step 3 为 0.658 秒。为了把本轮改动
与历史基线中的一个超大分子
SMILES/bond-order 差异隔离，另外执行了：

1. 最终代码第二次独立强制 `fork` 运行；
2. 临时隔离包恢复本轮修改前的“父进程串行压缩”源码再运行。

三者的非 provenance HDF5 fingerprints 均为 frame
`fc055b4e759f0f45a51caffc2e728fb6545253d15750afb48b694a7e4a0127ae`、molecule
`1eae03ccca00de6f814edac45c6e59053c979b207a08751dc89c2887b23aa00a`、reaction
`49f7cb18a8ad3014ac37159274bf4eaf9758dd3ce324eb3146eefdf926fbc1ac`；验证器比较退出码为
0。最终版与修改前隔离版的 `.route`、`.reactionabcd`、`.reaction`、`.moname` 也逐字节
一致，对应 SHA-256 分别为：

```text
a2d7b41d34cb73a77effe59e12b72ea43559c6237bdeeaa48833e328c5f7a4e8
c8726fcde90f946bb3ab15351b40cbcdd9312026ad5226ae009f2c7f85225be1
8cd32420a96c906944aa0eacae954034d415a885007c5c37c1fbe7e160d923c7
03de533118958811ab55b86a52edc158363d3b14b91309e77d17abb5c62f5dcb
```

资源清理修复后的最终复跑再次通过同一 manifest 比较，四个文本输出也逐字节一致。这证明
本轮压缩调度没有改变计算结果。更早保存的历史基线在一个超大、由 dump 坐标推断
的分子上给出另一组 bond-order/SMILES；由于最终版 repeat 与恢复修改前源码均一致，不能
把该历史差异归因于本轮线程优化，已作为独立的跨运行可复现性观察保留。输入副本和原始
文件 SHA-256 均仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

生产 Slurm 验证仍未完成。全轨迹复跑需要保存 compression 调度日志、Step 1/Step 3
wall、`TotalCPU/Elapsed`、MaxRSS、临时盘峰值、typecode 分布及 HDF5 manifest；若平均
array 小于 128 KiB，正确行为仍是串行而不是强行消耗全部申请核数。

本轮 compression-memory harness、强制 `fork` runner、隔离的修改前 package、
Matplotlib cache 和全部端到端输出目录均已从 `/private/tmp` 清理；这些产物不可恢复，但可
由只读输入重新生成。已有的其他 `rng-cutoff-*` 临时文件不属于本轮，未作修改。原始
`rp3.lammpstrj` 最终 SHA-256 复核仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

## 2026-08-08：真实 10,001 帧轨迹的高 nproc 调度与 reaction 批处理

### 真实轨迹反馈环与基线

为了避免继续只用 5 帧样例外推，本轮使用只读甲烷燃烧轨迹
`ch4_o2_3000K.lammpstrj`：135,333,201 bytes、10,001 帧、450 个 C/H/O 原子；原文件与
`/private/tmp` 副本 SHA-256 均为
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`。运行方式为强制
`fork`、no-HMM、同时写 molecule timeline 和 reaction events。

首次完整 `nproc=8` 运行中，Detect wall 为 11.589 秒、CPU 为 86.926 CPU-s、平均使用
7.50 核；Path wall 为 1.042 秒、CPU 为 3.589 CPU-s、平均使用 3.45 核。因此这个本地
规模没有复现生产作业的 3.3% 总体 CPU 利用率，但证明 Detect 并行正常，并把下一轮可测
瓶颈定位到 Path。Step 1/2 结果随后复制成只读快照，使每次 Path 回放约 1 秒；快照回放
与完整运行的 `.route`、`.reactionabcd`、`.moname` 逐字节一致，HDF5 也通过项目语义
manifest 比较。

修改调度前，对同一快照做 3 轮交错顺序的 `nproc` 矩阵；表内均为中位数：

| 请求 nproc | Path wall (s) | CPU (CPU-s) | 平均核 | route (s) | reaction (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.5403 | 1.4919 | 0.970 | 0.258 | 0.980 |
| 2 | 1.0847 | 2.1883 | 2.031 | 0.176 | 0.607 |
| 4 | 0.9564 | 2.7377 | 2.863 | 0.130 | 0.524 |
| 8 | 0.9017 | 3.3049 | 3.604 | 0.114 | 0.461 |
| 16 | 0.8920 | 3.2854 | 3.672 | 0.125 | 0.456 |
| 32 | 0.9272 | 3.4524 | 3.779 | 0.150 | 0.464 |
| 64 | 0.9407 | 3.6027 | 3.801 | 0.192 | 0.454 |

1→8 核仍有明确 wall 收益，但 8 核以后 reaction 已固定为 7 个实际 worker，继续增加申请
核数不再加速；route 却仍按请求值增加到 16/32/64 个 worker，导致短 atom task 的进程
启动、调度和 ordered spool 成本上升。该证据也说明不能把“尽量占满 64 核”作为优化
目标：本输入的有效并行上限约为 8 核，提高申请资源只会增加 core-seconds。

### route 的 atom-task 下限

原 `_route_worker_count()` 只用 `atom_count * frame_count / 65,536` 估算 worker。该输入有
约 450 万 scan values，允许 64 workers，但实际只有 450 个 atom task，即每 worker
约 7 项。新增红灯锁定 `_route_worker_count(64, 450, 10_001)`：修改前返回 64，期望为
8。最终选择器继续保留原 scan-values 约束，同时要求每 worker 约有 64 个 atom task，
用 `ceil(atom_count / 64)` 作为额外上限；短时间轴和宽轨迹的既有分支不变。

同一 `uv run` 环境的 route-only 配对复测中，修复后 `nproc=64` 的两次 Path 为
0.867/0.897 秒、route 为 0.119/0.116 秒；相对修改前 3 轮中位，route 约降低 38.8%，
Path 约降低 6.2%，并与 `nproc=8` 的 wall 持平。另一次 10 ms 进程采样的旧调度等价/
最终调度配对结果为：

| 指标 | 64 route workers | 自适应 8 workers | 变化 |
| --- | ---: | ---: | ---: |
| route wall | 0.261 s | 0.116 s | -55.6% |
| Path wall | 1.038 s | 0.876 s | -15.6% |
| Path CPU | 3.750 CPU-s | 3.262 CPU-s | -13.0% |
| 峰值进程数 | 65 | 9 | -86.2% |
| 峰值聚合 RSS | 1,848,492,032 B | 559,775,744 B | -69.7% |

macOS 无权读取 USS；这里的聚合 RSS 是父子进程 RSS 求和，会重复计算共享页，不能解释为
Slurm MaxRSS 或物理内存。它只用于同机同输入配对，仍清楚显示去掉 56 个短命 fork
worker 后没有隐藏的内存代价。

### reaction 的自适应 IPC chunksize

route 修复后，reaction 仍占 Path 约一半。该阶段把 10,000 个 transition 逐项用
`Pool.imap_unordered(..., chunksize=1)` 发送；每项只有 450 个扫描值、平均约 36 个修改
事件，计算太短，10,000 次小 IPC 成为主要开销。临时注入不同 chunksize，并把在途窗口
保持为每 worker 两个 chunk，3 轮交错结果如下：

| reaction chunksize | reaction 中位 (s) | Path 中位 (s) | CPU 中位 (CPU-s) |
| ---: | ---: | ---: | ---: |
| 1 | 0.469 | 0.880 | 3.256 |
| 4 | 0.331 | 0.741 | 2.928 |
| 8 | 0.304 | 0.716 | 2.821 |
| 16 | 0.283 | 0.708 | 2.733 |
| 32 | 0.277 | 0.686 | 2.680 |
| 64 | 0.291 | 0.708 | 2.714 |

最终没有把 32 无条件写死。`_reaction_chunksize()` 在 `1--32` 之间选择，并同时满足：

- 总 transition 至少为每个 worker 留约 4 个 chunk，避免粗粒度负载失衡；
- 每个 chunk 至多约 1,000,000 个 atom-scan values；
- 根据已观测修改事件的平均密度，把每个 chunk 目标限制在约 4,096 个修改事件；
- 修改事件未知、任务很短或只用父进程时保持 `chunksize=1`；
- `max_inflight = 2 * workers * chunksize`，即每 worker 最多两个在途 chunk。

该真实输入自动选择 7 workers、`chunksize=32`、`max_inflight=448`。10 ms 采样下，
chunksize 1/32 的聚合 RSS 分别为 559,775,744/559,529,984 bytes，没有可测增长；
32 的 reaction/Path 分别为 0.270/0.681 秒。未注入任何调试参数的最终生产路径确认运行为
reaction 0.283 秒、Path 0.696 秒、2.720 CPU-s、平均 3.91 核。相对本节最初
`nproc=64` 的 3 轮中位，最终单次确认 wall 约降低 26.0%、CPU 约降低 24.5%；生产集群
仍需用多轮 Slurm 指标确认。

### 结果契约、回归与边界

最终 `nproc=64` 输出与首次完整 `nproc=8` 输出的计数均为 10,001 frames、3,154
molecules、154,108 ranges、497 reaction types、43,588 compressed reaction-event rows；
HDF5 语义指纹均为
`7dc0197f3618a2d34fce5c3c6283260f425520510905d62c6a4547c3799d0d38`，manifest mismatch
为空。原始 HDF5 内部 reaction ID 顺序和 wall-time 属性可随无序 worker 完成顺序变化，
因此不能用文件字节哈希代替逻辑 manifest。三份文本输出在基线/最终间逐字节一致，
SHA-256 分别为：

```text
route         af09a4386b071bc6308bf6f3cc6f1828878fdbd99b0de6acb86844d8534b77a4
reactionabcd  af8303960eaa4118059d6d9211cf056e19486938cedd07998021a3bc1d7f1654
moname        5c8a1117fed6fccf5b2a5e65ce945d7cc56c739295d5e82906dc1bb7f9d9fee6
```

实现前 route 红灯为 1 failed，reaction selector 红灯为 5 failed；最终定向调度测试和
串行/并行汇总测试均通过。完整 `tests/test_timedoutput.py` 为 145 passed；排除
`tests/test_reacnetgen.py` 的全部测试为 207 passed；reaction/timestep/molecule/parm2cmd
子集为 13 passed、38 deselected。直接运行全套测试在无 GUI 的 macOS 会话中被
`tkinter.Tk()` 系统中止，故没有宣称 GUI 套件通过；失败堆栈未进入本轮代码。Black、
Isort、Ruff、flake8（忽略 E203/E501/W503）及 `git diff --check` 均通过。

该反馈环仍只有 10,001 帧和 450 原子，证明的是高 `nproc` 下短 route/reaction 任务的
调度修复，不等价于生产 25 万帧/更大原子数或 HMM 模式。生产 Slurm 仍需记录实际
route workers、reaction workers/chunksize/max_inflight、各子阶段 wall、
`TotalCPU/Elapsed`、MaxRSS、临时盘峰值和 manifest；若最终仍只需约 8 个有效 worker，
应相应降低 CPU 申请，而不是为提高利用率强制启动 64 个进程。

本轮 439 MiB 的甲烷快照/回放目录和 6 个临时基准脚本已从 `/private/tmp` 删除；这些
生成物不可恢复，但可由只读原轨迹重新生成。原轨迹最终 SHA-256 仍为
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`。已有
`rng-cutoff-*` 临时文件不属于本轮，已确认保留。

## 2026-08-08：50,005 帧扩展性与 HDF5 reaction writer 归因复核

### 1×/2×/5× 真实轨迹扩展性

继续使用同一份 10,001 帧、450 原子的只读甲烷轨迹，按输入文件列表重复 1、2、5 次，
从而得到 10,001、20,002、50,005 帧且不制造不同化学内容的可比工作负载。每组均为
Detect 请求 8 核、Path 请求 64 核，同时写 molecule timeline 与 reaction events。当前
测试机为 10 个逻辑 CPU；因此 Path 64 只用于压力测试显式高 `nproc` 调度，不代表 64 核
Slurm 节点。no-HMM 结果如下：

| 重复数 | Detect wall (s) | Detect 平均核 | Path wall (s) | Path CPU (CPU-s) | Path 平均核 | molecule / matrix / route / reaction (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1× | 11.245 | 7.579 | 0.757 | 2.905 | 3.835 | 0.211 / 0.107 / 0.122 / 0.309 |
| 2× | 24.033 | 7.343 | 1.102 | 5.625 | 5.104 | 0.254 / 0.119 / 0.207 / 0.515 |
| 5× | 66.934 | 7.508 | 2.775 | 15.137 | 5.455 | 0.367 / 0.298 / 0.418 / 1.676 |

1×/2×/5× 分别选择 7/14/36 个 reaction worker，均为 `chunksize=32`；route 均限制为
8 workers。总帧数放大 5 倍时 Path wall 为 3.66 倍，molecule、matrix、route 都没有
出现超线性退化。10 ms 进程采样的 Path 峰值进程数分别为 9、15、37，输出目录最终大小
约为 7.28、14.47、35.96 MB；构建期间观察目录峰值约为 21.37、41.20、101.84 MB，
随帧数近似线性。父子进程 RSS 求和峰值约为 0.55、0.96、3.06 GB，但 macOS 无法读取
USS，且 RSS 会重复计算 fork 共享页，只能用于同机相对比较，不能当作 Slurm MaxRSS。

HMM 模式也补做 1×/5×：HMM wall 为 0.331/1.406 秒，Path wall 为 0.399/1.162 秒；
过滤后都只剩 1,274 个 molecule，modified events 为 6,585/34,873，reaction 正确选择
父进程串行。此时 Path 平均约 1.0 核是工作量很小且阶段仅持续 0.4--1.2 秒的合理结果，
不能与生产作业连续 48 小时只使用约 2.1 核混为一谈。低利用率只有在 wall 长时间不降时
才构成性能故障。

### writer 超线性假设的反证

no-HMM 的 HDF5 计数与属性最初呈现一个可疑信号：

| 重复数 | ranges | reaction rows | molecule writer (s) | reaction writer (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1× | 154,108 | 43,588 | 0.080 | 0.130 |
| 2× | 308,202 | 87,178 | 0.109 | 0.280 |
| 5× | 770,484 | 217,948 | 0.170 | 1.352 |

5× 数据只有 5 倍 reaction rows，运行内记录的 reaction writer wall 却为 10.4 倍，因此
先将“压缩 dataset 的随机块写入随规模超线性恶化”列为首要假设。随后从 5× HDF5 恢复
50,004 个 transition Counter，只回放 `TimedOutputStore` writer；按原 worker 完成顺序和
transition 顺序各跑三轮，reaction write 均稳定在 0.200--0.207 秒，总阶段为
0.216--0.222 秒，54 batches、217,948 rows。两种顺序无差异，直接否定了 writer 本体
在 50k 帧规模超线性退化的假设。

为分离计算与写入，又从 molecule/range 表重建精确的 `450 × 50,005` atomeach/conflict
矩阵。不同进程数的纯 reaction 计算三轮中位如下：

| nproc | 1 | 4 | 8 | 10 | 16 | 24 | 36 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| wall (s) | 4.641 | 1.721 | 1.216 | 0.998 | 0.954 | 0.989 | 0.986 |

10--16 个进程已经饱和本机 10 个 CPU，继续增加到 24/36 没有 wall 收益。接入真实 HDF5
event writer 后，8/10/16/24/36 进程的总 wall 中位分别为
1.180/1.102/1.153/1.194/1.238 秒，而 writer 内计 wall 为
0.556/0.550/0.706/0.866/0.855 秒。writer-only 仍只需约 0.20 秒，说明增加 worker 后的
“writer 时间”主要是父进程在系统过度订阅时被工作进程抢占；当前计时是 wall，不是
writer 独占 CPU 时间。所有组合的 `.reactionabcd` SHA-256 均为
`6953f09bb5a4e83a87b67b0ab3c58ed7c539d0c4c58c614ee448df6e6b16a4c5`，计算结果一致。

### 决策与生产验证边界

本轮不修改 HDF5 压缩、块布局，也不把 reaction worker 硬限制为 10/16：这会用本机
10 核过度订阅现象错误约束真正的 64 核 Slurm 节点。在 64 核 allocation 上，36 个
reaction worker 不应产生本机同样的 CPU 争用；若仍只使用约 2.1 核，优先检查 CPU
affinity、cgroup/Slurm 实际配额、worker 是否成功启动，以及具体停留在哪个子阶段。

生产复跑必须同时保存：`SLURM_CPUS_PER_TASK`、`os.sched_getaffinity(0)` 或等价可用 CPU
集合、程序请求和实际 route/reaction workers、reaction chunksize/max_inflight、各阶段
wall、HDF5 molecule/reaction write 属性、`sacct` 的 `TotalCPU/Elapsed/AllocCPUS/MaxRSS`、
临时目录峰值与语义 manifest。只有 64 核节点上的这些证据才能决定继续优化 IPC、降低
CPU 申请，还是排查调度/亲和性；当前本地结果不支持新增一个平台无关的 writer 修复。

为消除下一次复跑的启动诊断盲区，`ReacNetGenerator` 现在会在参数解析后记录请求
`nproc`、进程可见 CPU 数、逻辑 CPU 数、紧凑 affinity 范围和
`SLURM_CPUS_PER_TASK`。当请求进程数超过可见 CPU，或 Slurm 数值与 affinity 数量不一致
时发出 warning；显式 `nproc` 仍保持不变，避免诊断功能擅自改变已有 CLI/API 语义。
日志契约先建立 2 个红灯测试，修改前为 2 failed，修改后为 2 passed；分别覆盖
64→4 的 Slurm/非连续 affinity 不一致和无 affinity 平台的逻辑 CPU fallback。最终
`tests/test_timedoutput.py + tests/test_detect.py + 2 个 CPU 日志测试` 为 191 passed；
reaction-event/timestep/molecule/parm2cmd 相关子集为 15 passed、38 deselected。Black、
Isort、flake8（忽略项目既有 E203/E402/E501/W503）及 `git diff --check` 均通过。

本节扩展性实验完成后再次核对只读原轨迹 SHA-256 为
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`。约 66 MB 的
扩展性输出目录和 4 个本轮临时 harness 已从 `/private/tmp` 删除；它们不可恢复，但可由
只读输入重新生成。已有 `rng-cutoff-*` 仍不属于本轮，未作修改。

## 2026-08-08：route 活动 transition 索引消除稀疏 reaction 全矩阵扫描

### 红灯与根因

此前 route 已逐原子扫描完整 timeline，并准确累计 modified atom event 数；但只把总数
交给 reaction 调整 worker 数。`ReactionsFinder` 仍对全部 `T-1` 个 transition 逐项读取
`atomeach[:, t:t+2]`，即使某个 transition 没有任何原子改变也扫描全部 `N` 个原子。
HMM 或稳定轨迹中，真实变化可能只覆盖少量时间点，因此该路径把稀疏问题重新放大为
`O(N*T)` 列扫描和 `T-1` 个 Python/Pool task。

先在真实 `findreactions()` 调用 seam 建立两个红灯：要求只访问给定的活动 transition，
并差分比较稀疏扫描与旧全扫描的 `.reactionabcd`。修改前均因 API 不支持活动索引而失败
（2 failed、145 deselected）。随后为 route 建立 3 个红灯，锁定 `T-1` 字节临时索引、
跨未分配 molecule ID `0` 的原始变化和父进程/worker 共享结果；修改前均失败。

### 最终实现与内存边界

- `_AtomFrameStore` 新增 `uint8[T-1]` 临时 memmap；空间只随帧数增长，与 atom 数无关。
  选择 1 byte 而不是 packed bit，是为了让多个 route worker 只执行“把该 byte 写成 1”
  的幂等同值写入，避免跨进程 bit read-modify-write 竞态。
- route 已有的 change mask/`change_time` 直接标记活动 transition，不额外扫描 timeline，
  也不把变化索引数组经 IPC 返回。包含 `0` 的 timeline 继续按 65,536 行有界扫描；全块
  都变化时直接填 1，避免 dense boolean-index 微内核回退。split route 不重复标记。
- reaction 将索引规范化为有序、唯一、合法的原始 transition index，只把这些任务交给
  父进程或 Pool；worker 数、chunksize、in-flight 和 scan-value 估算都改用活动任务数。
  索引未知时仍完整扫描，保持兼容路径；0 个活动 transition 时不读取任何 matrix 列。
- HDF5 根属性新增 `reaction_total_transition_count`、
  `reaction_active_transition_count` 和
  `reaction_active_transition_index_available`，离线即可判断优化是否命中。

### 隔离反馈与稠密退化检查

195.312 MiB 的真实 `_AtomFrameStore` 形态为 50,000 atoms × 2,048 frames，只在一个
transition 发生 `A+B->C`。三轮交替中位如下：

| 模式 | reaction wall | reaction CPU | 相对全扫描 |
| --- | ---: | ---: | ---: |
| 全部 2,047 transitions | 0.353108 s | 约 0.353 s | 1.00× |
| 仅 1 个活动 transition | 0.001834 s | 约 0.00183 s | 192.57× faster |
| 显式索引全部 2,047 个 | 0.352314 s | 约 0.352 s | 0.998× wall |

三种输出逐字节一致；活动索引文件仅 1.999 KiB。100 万个全部变化 transition 的 route
计数微内核，原计数/同时标记中位为 0.000076/0.000088 秒，即绝对增加约 12 微秒、比值
1.157；修复前第一版 boolean assignment 的比值曾为 5.422，已通过 dense fill 快路径
消除。3 个真实并行 route worker 对同一 transition 的重叠写入测试也得到精确 union。

### 真实 10,001 帧结果与语义

同一只读甲烷轨迹使用强制 `fork`、8 核复跑：

- no-HMM 为 10,000/10,000 个 transition 全活动，route 为 0.121 秒，与修改前真实基线
  0.122 秒相当；reaction 没有跳过任务。HDF5 语义指纹仍为
  `7dc0197f3618a2d34fce5c3c6283260f425520510905d62c6a4547c3799d0d38`，`.moname`、
  `.route`、`.reactionabcd` 哈希仍分别为
  `5c8a1117fed6fccf5b2a5e65ce945d7cc56c739295d5e82906dc1bb7f9d9fee6`、
  `af09a4386b071bc6308bf6f3cc6f1828878fdbd99b0de6acb86844d8534b77a4`、
  `af8303960eaa4118059d6d9211cf056e19486938cedd07998021a3bc1d7f1654`。
- HMM 为 1,655/10,000 个 transition 活动；活动索引版 reaction 为 0.06504 秒，同一输入
  禁用索引的全扫描版为 0.17578 秒，wall 降低约 63.0%。两次独立 HMM 运行的内部 ID/
  文本顺序可不同，但验证器归一化后的 frames、molecules、reactions 指纹逐项一致，完整
  语义指纹均为
  `46099ccf61aad9cc17224781ccba819ad1a012c35d0320e1fb8ab0d878007f00`；
  `.reactionabcd` SHA-256 均为
  `0c4df8dfc1406660de331167faf10bef1ca8e1e4a453483fc2769e7ca84c1a89`。

最终 `tests/test_timedoutput.py + tests/test_detect.py + 2 个 CPU 日志测试` 为 199 passed；
排除 GUI/reacnetgen 集成文件的全部测试为 215 passed；相关集成子集为 15 passed、
38 deselected。Black、Isort、flake8（忽略项目既有 E203/E402/E501/W503）、compileall
及 `git diff --check` 均通过。该优化只消除“时间上稀疏”的 reaction 二次全扫描；
no-HMM 全 transition 活动或 molecule/route 主阶段仍需按原瓶颈处理，不能把隔离的
192× 外推为完整 Step 3 加速比。

本轮约 8.6 MB 的真实差分输出目录和 2 个临时 harness 已从 `/private/tmp` 删除；这些
产物不可恢复，但可由只读输入重新生成。原轨迹 SHA-256 最终复核仍为
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`；已有
`rng-cutoff-*` 不属于本轮，未作修改。

## 2026-08-08：cheap-fork SMILES 复用 worker 已解码结构

### 当前分支因果复核

为回答“是否由当前分支开发造成”，本轮固定比较分支点 `02a12117` 与已提交基线
`b332ce99`。`02a12117` 的 SMILES worker 已返回 name/atoms/bonds/frames，父进程再逐条
执行 fallback/miso、`.moname` 和 timeline spool，所以单父进程 fan-in 并非 HDF5 分支
从无到有引入。`b332ce99` 新增的 `TimedOutputStore.add_molecule()` 又位于同一父循环，
最初对 molecule/species/atom/bond/offset/range datasets 逐 molecule 多次执行
`resize + write`；它会放大既有串行临界路径，方向上能够解释工作进程更长时间等待。

因此证据支持“当前分支放大并暴露既有瓶颈”，不支持“48 小时低利用完全由当前分支唯一
造成”。原始全轨迹已删除，且作业 `1089747` 的精确 commit、阶段 profile、CPU affinity
和 writer 属性未保存，仍需排除 cgroup/binding、实际 worker 数和其他串行子阶段。当前
工作区的批量 HDF5 writer 修复分支新增的细粒度写放大；SMILES/route/reaction 的调度、
索引与解码复用则同时改善早于该分支的父进程 fan-in。

### 根因、红灯和适用边界

在此前已经启用的廉价 `fork` 批处理路径中，worker 会解压 molecule record 的 atoms、
bonds 并计算 SMILES，但只把 species name 返回父进程。父进程随后为 `.moname`、miso/
fallback 和可选 timeline 再次读取并解压同一结构。对于 112,300 条便宜记录，SMILES
计算本身已经足够快，这一重复父进程解码成为可测串行占比。

本轮先建立两个红灯：一项要求 decoded worker 契约返回 name/atoms/bonds；另一项要求
cheap-fork pool 复用该结构、父进程只读取 timeline 字段且不得再次调用结构解码。修改前
均因入口不存在或父进程重复解码而失败。平台/批次参数和最大记录指标共同锁定选择边界：
仅当 worker 数大于 1、start method 为 `fork`、既有调度选择 `chunksize>1`，且所有保留
记录的最大压缩结构不超过 64 KiB 时复用；串行、`spawn`、`forkserver`、复杂记录和最大值
未知的旧中间调用继续 name-only。混合分布回归还固定“平均值廉价但单条为 65,537 bytes”
必须回退。测试替身显式断言 worker 函数并返回真实三元组，避免三字符字符串被 Python
偶然解包后造成假通过。

最终实现让 worker 从已经解码的 atoms/bonds 计算名称后直接返回三者。父进程按原始输入
顺序消费结果；需要 molecule timeline 时，第二个文件游标只读取第 4 个 frame payload，
否则不执行父进程结构扫描。fallback VF2、miso、`.moname` 与 HDF5 writer 都复用同一份
decoded 结构。HMM/no-HMM 已遍历的复制循环同时累计最大压缩结构字节，不解压也不增加
文件扫描；最大值未知或超过 64 KiB 时整阶段保持 name-only。ordered-result spool、
in-flight 上限和 64 条批次保持不变，因此乱序结果仍落盘保序，待处理内存不随 molecule
总数无界增长，也不会由少量巨大分子突破已验证的单记录边界。

### 隔离 A/B 与内存复核

只读输入 `/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 含 5 帧、12,326 原子，Detect
生成 5,615 条 molecule records；将这些记录原样重复 20 次得到 112,300 条、38,100,340
字节压缩结构字段的反馈环。相同代码、输入和输出路径做交替 A/B，性能快照不含 import：

重新生成原始 5,615 条记录验证新增 guard：总量 1,905,017 bytes，平均 339.3 bytes，
最大单条 4,108 bytes，低于 65,536-byte 上限；因此重复 20 次后的 A/B 仍命中 decoded
复用路径。审查用临时目录在指标输出后为空并已删除。

| 路径 | wall 中位 (s) | parent CPU (CPU-s) | child CPU (CPU-s) | 平均用核 |
| --- | ---: | ---: | ---: | ---: |
| name-only，父进程重新解码 | 3.216778 | 3.247961 | 4.196495 | 2.303 |
| worker decoded 结构复用 | 2.887869 | 2.751710 | 4.474196 | 2.502 |

wall 降低约 10.2%，父进程 CPU 降低约 15.3%，总 CPU 由约 7.444 降至 7.226 CPU-s
（约 -2.9%），平均用核提高约 8.6%。所有运行 `.moname` SHA-256 均为
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec`。

首次用 `ru_maxrss` 观察到 child 约 94→140 MiB，但该轮恰好触发 Matplotlib 字体缓存
子进程，不能归因于 worker payload。固定可写 `MPLCONFIGDIR=/private/tmp/rng-mpl-cache`
并预热后重新测量，name-only/reuse 的 parent RSS 为 274.375/276.938 MiB，最大 child
为 38.531/39.750 MiB，增量分别只有 2.563/1.219 MiB；decoded ordered spool 的典型
峰值约 0.072 MiB。所有结果再做紧凑 ndarray 编码虽然内存相近，却增加 CPU 并削弱 wall
收益，因此未保留。

### 端到端语义与被否定候选

同一 112,300 条记录继续执行完整 HDF5、matrix、route、reaction：nproc=1 的 molecule/
总 collect 为 7.386/10.306755 秒，nproc=8 候选为 5.592/8.519057 秒。该对比同时包含既有
并行调度，只作为端到端语义和实际链路证明，不作为本功能的独立加速比。两者 manifest
语义指纹均为
`7d1aea66c484870b90867b690084783c84ae9887dfc99ebc4812f8b75a2bb147`，差异列表为空；
`.moname`、`.route`、`.reactionabcd` SHA-256 分别为
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec`、
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

以下候选未进入源码：回传压缩 record 的中位 wall 为 3.057886 秒，慢于 decoded 结构；
对 decoded 结果统一二次紧凑编码增加 CPU；chunksize 8/16/32 均慢于 64；reaction 的
row/frame bitmap 在 200,000 行、129 帧、64 个活动 transition 上只有约 1.01--1.34×
微内核收益，却需要新的排序、IPC 和 spool 契约，风险高于收益。现有活动 transition
索引继续保留，不新增 bitmap/event spool。

### Code review 闭环

Standards 首轮指出三条父进程路径用带 `None` 哨兵的 5 元位置 tuple 传递结果，属于容易
错位的 data clump；现改为具名 `_SmilesResultRecord`，compressed/decoded 两个工厂集中
校验状态，消费端只读具名字段。Spec 审查指出平均值会掩盖巨大离群 molecule、旧 handoff
仍绝对声称 name-only、分支因果结论不足；三项分别由最大记录 guard、全篇协议修订和
`02a12117..b332ce99` 源码对比闭环。Standards 复审又指出 total/max 两个统计量跨三模块
并列传递会造成 shotgun surgery；现合并为 `_SmilesWorkMetrics`，经单一
`smilesworkmetrics` 键跨 HMM、主对象与 Path 传递。两个独立审查轴最终均确认无剩余
P0--P2。

`tests/test_timedoutput.py` 当前为 163 passed；相关 Detect/reacnetgen 子集为 14 passed、
83 deselected；排除无头 GUI 集成文件的全部测试为 225 passed。Black、Isort、flake8
（忽略项目既有 E203/E402/E501/W503）、compileall 和 `git diff --check` 均通过。生产
64 核 Slurm 尚未复现；因此本节只证明本地 cheap-fork 路径消除了父进程重复解码并保持
语义，不把约 10.2% molecule-name wall 收益外推为完整 Step 3 或作业 `1089747` 的
加速比。最终仍需保存 affinity、实际 worker、各阶段 wall、`TotalCPU/Elapsed`、MaxRSS、
临时盘峰值与语义 manifest。

本轮 98 MiB 的重复 molecule/HDF5/文本输出目录和 176 KiB 的专用 Matplotlib cache 已从
`/private/tmp` 删除；这些反馈环产物不可恢复，但可由只读输入重新生成。原始
`rp3.lammpstrj` 未修改，最终 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。已有
`rng-cutoff-*` 不属于本轮，已确认保留。

## 2026-08-08：`.moname` 专用编码与压缩 record 字段位置表

### 修复后 profile 与红灯

在上一节 decoded 结构复用完成后，重新从只读 `rp3.lammpstrj` 生成 5,615 条 molecule
records，并原样重复到 112,300 条、38,100,340 个压缩结构字节。强制 `fork` 后自动选择
2 workers、`chunksize=64/max_inflight=256`。带 cProfile 的当前基线 wall 为 4.034 秒；
父进程通用 `_formatmoleculename()` 经 `listtostirng()` 产生约 2,459,420 次递归调用和
3,068,820 次 generator 取值，`listtostirng()` 累计约 2.034 CPU-s，已经超过其他可直接
修改的父进程纯 Python 热点。

`.moname` 只有固定的 `name + " " + atoms(;分隔) + " " + bonds(,和;分隔)` 协议，无需
通用任意维递归。先增加 3 个红灯，分别覆盖单原子/空 bonds、多 atoms/bonds 和空字段，
并 monkeypatch 通用 formatter 为直接失败；旧实现为 3 failed，新实现全部通过。生产代码
现在直接用专用 join，未改变文本格式，也不增加持久对象。

### 交替 A/B 与 worker 复核

相同输入、输出和 2-worker 调度按旧通用/专用 formatter 交替各三轮，不含 Detect/HMM：

| formatter | wall 中位 (s) | parent CPU 中位 (CPU-s) | `.moname` SHA-256 |
| --- | ---: | ---: | --- |
| 通用递归 | 2.467068 | 1.783569 | `1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec` |
| 专用 join | 2.256256 | 1.386763 | `1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec` |

wall 降低约 8.5%，父进程 CPU 降低约 22.2%。父进程变快后又筛查 2/3/4/6 workers，单轮
wall 分别为 2.173/2.312/2.701/3.277 秒；更多 workers 同时增加 parent IPC 和 child
CPU，因此没有修改已校准的 2-worker 选择。这里的目的仍是缩短 wall/core-hours，而不是
人为让所有申请 CPU 看起来繁忙。

### record reader 与否决候选

新的 profile 中 `_iter_compressed_record_fields()` 仍在每条记录创建 dict、做 set membership
并按请求顺序重建 tuple。把请求字段一次映射到输出位置后，每条只分配定长 list；在同一
61 MiB 文件上排除首次冷测的四轮中位约为 0.1655→0.1416 秒（约 -14.5%），112,300 条
记录和 38,100,340 checksum 完全一致。既有回归覆盖 `(2,0)` 乱序字段、非零起始 offset、
短 header、短 payload 和 record 中断，因此没有削弱错误检测。

两个候选未进入源码：对 ≤4 KiB generic spool 结果跳过 LZ4 的三轮 wall 中位只有约
2.350→2.333 秒，低于噪声且没有稳定 CPU 收益；`WriteBuffer` 在无 byte limit 时跳过
字节计数的中位反而约 2.363→2.397 秒。最终源码独立三轮 wall 为
2.253/2.321/2.366 秒，parent CPU 为 1.331/1.385/1.418 CPU-s，输出哈希继续不变。

`tests/test_timedoutput.py` 最终为 166 passed；相关 Detect/reacnetgen 子集为 14 passed、
83 deselected；排除含 Tk GUI 集成文件的其余测试为 228 passed。Black、Isort、compileall
及 `git diff --check` 通过；flake8 对本轮路径通过，`utils.py` 另显式排除该文件原有的
E302/E704 overload stub 和 E722 bare-except，未把既有风格问题误记成本轮回归。

本轮 153 MiB 的重复记录、profile 输出、临时脚本和专用 Matplotlib cache 已从
`/private/tmp/rng-smiles-profile-20260808` 精确删除；这些产物不可恢复，但可由只读输入
重新生成。原始 `rp3.lammpstrj` 未修改，最终 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`；已有
`rng-cutoff-*` 不属于本轮，继续保留。

## 2026-08-08：SMILES 不变量缓存与有界保序内存前置缓存

### 修复后反馈环与 worker profile

再次从只读 `rp3.lammpstrj` 生成 5,615 条真实 molecule records，并原样重复到
112,300 条、64,139,660 bytes record 文件和 38,100,340 个压缩结构字节。强制 `fork`
后仍由既有调度选择 2 workers、`chunksize=64/max_inflight=256`。本轮开始时，带
cProfile 的完整 molecule-name pipeline 为 3.039 秒，父/子进程分别消耗 2.498/5.459
CPU-s；无 profiler 的受控基线约 2.15 秒，`.moname` SHA-256 为
`1b5691f5e68ca404967e6bef44cdf1ae5a6984da88c01580b62c1fcae58cf3ec`。

独立串行 worker profile 在 112,300 条记录上报告 4.712 秒，其中 `_re()` 累计约
0.504 CPU-s；它为每个 molecule 重复排序相同的 atom names、拼装相同 pattern，再交给
`re` 的全局缓存。`convertSMILES()` 还为每个 molecule 重复执行
`Chem.MolFromSmiles("")`，虽然生成的空 RDKit 模板在一个 converter 生命周期内不变。

两个红灯分别在旧实现稳定失败：第一次 `_re()` 后禁止再次调用 `sorted()`；连续转换两个
相同单碳结构时统计 `Chem.MolFromSmiles()` 只能调用一次。最终实现把编译后的 radical
pattern 和解析后的空 RDKit molecule 缓存在每个 parent/worker converter 实例上；每个
实际 molecule 仍创建独立 `RWMol`，不会共享可变图。

### 不变量缓存 A/B

相同 112,300 条输入按修改前/候选交替各三轮，均运行完整 Pool、保序恢复、名称表和
`.moname` 输出：

| 候选 | 修改前 wall 中位 | 候选 wall 中位 | 修改前 child CPU | 候选 child CPU | 输出 |
| --- | ---: | ---: | ---: | ---: | --- |
| atom-name pattern 编译一次 | 2.1598 s | 2.1004 s | 4.1857 CPU-s | 4.0264 CPU-s | 哈希一致 |
| 空 RDKit template 解析一次 | 2.1089 s | 2.0404 s | 4.0878 CPU-s | 3.9470 CPU-s | 哈希一致 |

pattern 候选的 wall/child CPU 分别降低约 2.8%/3.8%，空模板候选分别降低约
3.2%/3.4%。缓存 pattern 后的串行 worker profile 为 4.314 秒，`_re()` 降至约
0.142 CPU-s，`sorted()` 从 112,300 次降至 1 次。这里报告的是同机同输入的局部 A/B，
不把两个比例直接相加，也不外推成生产全 Step 3 加速比。

直接使用 `Chem.RWMol()` 的候选三轮均慢约 4%，且 `.moname` 哈希从 `1b5691...` 变为
`3dfde2...`；这证明解析得到的空模板包含影响输出的 RDKit 状态。该候选未进入源码，最终
只缓存修改前已经使用的 `Chem.MolFromSmiles("")` 结果。

### 1 MiB 有界保序内存前置缓存

profile 中的小乱序结果仍经过 pickle/LZ4、memmap index 和临时文件 seek/read/write。
现在 `_DiskOrderedResultSpool` 先把**已编码**结果放入固定预算的 dict；每项按 payload 加
128 bytes 保守记账，默认估算总量最多 1 MiB，超过预算的单项或 backlog 立即保留原磁盘
路径。`results.bin` 和稀疏 index mmap 也延迟到第一次真实 spill 才创建，纯内存 backlog
不会预先占用 `16 × total` 的临时文件逻辑空间。无预算时类本身仍可完全按旧磁盘语义
运行。日志分别报告累计 encoded bytes、实际 disk written、峰值文件、峰值估算内存和
最大 pending，避免把“未写盘”误报成“没有乱序”。

完整 SMILES 路径的三组磁盘/内存交替 A/B 中，逐对 wall 比值中位约为 0.998，属于持平；
但 parent CPU 每组都降低约 7%--12%，本地磁盘写入从约 4.6--5.1 MiB 降为 0，峰值估算
内存仅 0.126--0.251 MiB。为了隔离 worker 抖动，又用 112,300 个代表性 generic 结果、
每组 320 条逆序 backlog 直接运行真实 spool：

| spool | wall 中位 | 累计磁盘写入 | 峰值文件 | 峰值估算内存 |
| --- | ---: | ---: | ---: | ---: |
| 全磁盘 | 1.5184 s | 35,149,900 bytes | 100,160 bytes | 0 |
| 1 MiB 前置缓存 | 0.7550 s | 0 | 0 | 141,120 bytes |

隔离 wall 降低约 50.3%。新增回归同时覆盖小 backlog 不触盘、单个结果超过预算时必须触盘、
pop 后记账归零、无 spill 时不创建 data/index 文件及原有 truncate/duplicate/truncated
codec 契约；因此优化的是常见小乱序，不是把原先的磁盘上界替换成无界 RAM。

### 否决候选与最终验证

- 真实 atom payload 解压后是 `list[int]`，不是可直接复用的 ndarray；`np.array()` 是必要
  紧凑化，worker profile 也只有约 0.040 CPU-s，因此没有伪装成“去掉重复 copy”。
- bond row 改为 tuple 的完整 A/B 只改善约 1.1% wall，child CPU 反而增加约 0.9%；真实
  平均只有 3.43 bonds/record，在途内存仅节省几十 KiB，未进入源码。
- 跳过空 RDKit 模板解析会改变结果，已明确否决；只有语义相同的模板复用被保留。

延迟磁盘创建后的最终源码独立三轮 wall 为 2.033/1.972/1.964 秒，中位 1.972 秒；
parent/child CPU 中位为 1.064/3.827 CPU-s。每轮日志显示约 4.89--4.94 MiB encoded、
0 disk written、0 peak file、0.126 MiB peak memory，`.moname` 哈希继续为
`1b5691...f3ec`。

`tests/test_timedoutput.py` 为 171 passed；排除 Tk GUI 集成文件的其余测试为 233 passed；
Detect/reacnetgen 相关子集为 14 passed、83 deselected。Black、Isort、flake8、compileall
和 `git diff --check` 通过。该数据仍是本地 fork 反馈环；真实 64 核 Slurm 的 affinity、
实际 worker、四段 Step 3 wall、`TotalCPU/Elapsed`、MaxRSS、临时盘峰值和语义 manifest
仍是生产验收所需证据。

本轮 151 MiB 的重复 records、A/B 输出、profile、脚本和 Matplotlib cache 已从
`/private/tmp/rng-next-profile-20260808` 精确删除；这些临时产物不可恢复，但都可由只读
输入重新生成。原始 `rp3.lammpstrj` 未修改，最终 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`；已有
`rng-cutoff-*` 不属于本轮，继续保留。

## 2026-08-08：重复 SMILES 名称缓存与短 molecule-range 快路径

### 完整 Step 3 反馈环

本轮不再只剖析 molecule-name 微内核，而是把只读
`/Users/huangchen/Downloads/rng_test/rp3.lammpstrj` 的 Detect/no-HMM 状态固定后，独立
运行完整 `_CollectPaths.collect()`。每轮记录父/子进程 CPU、molecule/matrix/route/
reaction 与 writer wall、文本 SHA-256、HDF5 manifest 和 cProfile；Detect/HMM 不计入
Step 3 A/B。基线 5 帧、5,615 molecules、6,076 ranges 的四阶段为
0.988/0.163/0.309/0.033 秒，完整 Step 3 为 1.502 秒，平均约 0.99 核。profile 中
SMILES 为 0.578 秒，逐 molecule timeline 整理/HDF5 为 0.229 秒，route 为 0.309 秒。

为了让父进程 writer 开销可测，又把同一 5,615 条真实 molecule record 原样重复 20 次，
得到 112,300 molecules、121,520 ranges。该压力仍只有 5 帧，所以适合验证名称、短 range
和 definition writer，不代表生产长时间轴；重复定义还会人为制造 matrix conflict，故
本轮没有根据其约 4.3 秒 matrix 阶段修改矩阵算法。

### 1 MiB/进程的 SMILES 名称 LRU

真实 5,615 molecules 只有 413 个最终 species。新增红灯要求两个仅全局 atom ID 不同、
但 atom-type 顺序和局部 bond 序列完全相同的结构只调用一次 `convertSMILES()`；另一个
红灯把预算缩至 600 bytes，锁定 bond level 不同不能误命中且旧项必须被逐出。修改前分别
稳定表现为重复转换和缺少预算常量。

最终键由 atom-type 序列和按输入顺序映射到局部 atom position 的 bond triples 组成，
不含全局 atom ID；它只复用**完全相同的有标签图**，不会把仅仅同构但局部顺序不同的图
错误合并。每个 parent/worker converter 使用独立 `OrderedDict`，按 key payload、name 和
每项 512 bytes 保守开销累计，估算上限固定为 1 MiB；超预算项不缓存，满额时按 LRU
逐出。单记录还会先按 atom/bond 数估算 key 大小，放不进预算时在构造局部 bond ndarray
之前直接绕过缓存，避免“最终不缓存但先产生大瞬时副本”。缓存值包括 RDKit 失败的
`None`，但 fallback VF2 仍在父进程按原路径执行。

112,300 条 decoded 真实结构的独立新进程 A/B 中，当前逐条转换中位 2.3890 秒，有界缓存
中位 0.5930 秒，名称序列哈希一致；命中 111,627/112,300，缓存 673 项且没有逐出。完整
重复-record Step 3 中 child CPU 从 4.788 降至 2.528 CPU-s，wall 只从 14.238 降至
13.976 秒，因为父进程 writer 与 worker 重叠并已成为主瓶颈。短轨迹三组完整 A/B 的
Step 3 wall 中位为 1.4118→1.3811 秒；这里不把微内核的 75% 收益外推为全流程比例。

### 短 frame 序列的线性 range 扫描

112,300-record profile 显示 `_itermoleculeranges()` 对通常只有数帧的每个 molecule 都
启动 `np.diff/flatnonzero/concatenate`，累计约 1.85 CPU-s。红灯在 6 项
`[0, 1, 1, 3, 4, 7]` 输入上禁止调用 `np.diff`，并要求仍输出闭区间
`[0,1]、[3,4]、[7,7]` 的 `uint64` 数组；修改前稳定失败。

现在无 frame/timestep filter 且不超过 64 项时使用单次 Python 线性扫描，固定最多保留
64 项；更长序列和所有过滤路径继续使用原有 65,536-row bounded NumPy scan。三组
112,300-record 交替 A/B 中：

| 指标 | NumPy 短序列路径中位 | 线性短序列路径中位 | 变化 |
| --- | ---: | ---: | ---: |
| 完整 Step 3 wall | 13.9990 s | 12.1287 s | -13.4% |
| molecule stage | 9.2997 s | 7.4333 s | -20.1% |
| timed-output writer | 5.9767 s | 4.2603 s | -28.7% |
| parent CPU | 14.7858 CPU-s | 12.5979 CPU-s | -14.8% |

两组 parent MaxRSS 中位约 297.3/297.7 MiB，没有随 molecule 数建立新的常驻列表；压力
manifest 均为 `8a68098f244934c7bf59a0d46a69714e4737a7b534afcf2c724b6490263d651a`。

### Writer dtype 快路径与 atom array 复用

range 生成已输出 `uint64` 后，`TimedOutputStore.add_molecule()` 仍对每个 starts/ends
执行四次通用 `np.issubdtype()`。新增红灯在已规范化数组路径禁止调用该层级检查，同时
保留负值、start>end 和越界回归；实现改为常数时间的 `dtype.kind` 分支，所有原有值域
校验和 zero-copy 语义不变。相邻放大复跑的 writer 为 4.216→3.713 秒；该项没有独立三组
端到端 A/B，因此只作为 profile 闭环，不单独宣称稳定比例。

SMILES 解码后的 atom IDs 原先固定为平台 `int`，父进程写 HDF5 时又复制为 `uint64`。
现在第一次紧凑化就使用 schema 所需的 `uint64`；位宽仍是 8 bytes，worker IPC 和峰值
元素数不变，下游 writer 可直接共享数组。三组强制旧 `int`/最终 `uint64` 交替 A/B：

| 指标 | `int` 中位 | `uint64` 中位 | 变化 |
| --- | ---: | ---: | ---: |
| 完整 Step 3 wall | 11.7708 s | 11.3772 s | -3.3% |
| molecule stage | 6.9936 s | 6.6875 s | -4.4% |
| timed-output writer | 3.8536 s | 3.1089 s | -19.3% |
| parent CPU | 12.2419 CPU-s | 11.9180 CPU-s | -2.6% |

parent MaxRSS 中位约 299.0/297.8 MiB，文本哈希和压力 manifest 均一致。把平均 3.43 条
bond 也改成 ndarray 的临时原型 wall 为 11.425 秒，未优于最终中位，child CPU 反而约
2.58→3.10 CPU-s；该候选未进入源码。

### 最终语义、性能与回归

最终源码重新运行原始 5 帧轨迹，完整 Step 3 为 1.257 秒，molecule/matrix/route/
reaction 为 0.740/0.157/0.316/0.034 秒，writer 为 0.133 秒。相对本轮最初同一反馈环的
1.502/0.988/0.235 秒，分别约降低 16.3%、25.1% 和 43.6%；这是本地短轨迹结果，不外推
为 48 小时生产作业比例。HDF5 语义指纹保持
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和 `.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`173299dfc8c574e8bc060a73c6343d32cae7353702a09fee53b7d64ec505ae15`。

最终 `tests/test_timedoutput.py` 为 176 passed；排除 Tk GUI 集成文件为 238 passed；
Detect/reacnetgen 相关子集为 15 passed、82 deselected。Black、Isort、flake8、
`compileall` 和 `git diff --check` 通过。真实 64 核 Slurm 仍未验证；生产复跑必须继续
保存 affinity/实际 worker、四阶段 wall、`TotalCPU/Elapsed`、MaxRSS、writer 属性、
临时盘峰值和语义 manifest，不能把本地 5-frame/重复-record 比例直接套到全轨迹。

本轮约 444 MiB 的完整 Step 3 A/B 输出、profile、重复 record、脚本与 cache 已从
`/private/tmp/rng-whole-step3-profile-20260808` 精确删除；Detect 生成在系统临时目录的
3,206,983-byte molecule 文件和空 origin 文件也按精确路径删除。这些产物不可恢复，但可
由只读输入重新生成。`rng-cutoff-*` 不属于本轮并继续保留；`uv.lock` 仍不存在。原始
`rp3.lammpstrj` 的大小/mtime 未变，最终 SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

### 双轴 code review

最终修改按 Standards/Spec 两个维度独立复核，均未发现可复现的 P0--P2 问题。Standards
轴核对了 LRU 命中、淘汰、`None` 缓存、字节记账、超预算前置拒绝、range 快路径、
`dtype.kind` 校验和 `uint64` 下游兼容性，7 项定向测试通过。Spec 轴进一步对每个长度
1--64 的 1,000 组随机重复、间断及乱序 `uint64` 输入执行新旧 range 路径差分，并核对
filter/长列表回退、VF2 fallback、HDF5/文本语义和本地 A/B 结论边界，13 项定向测试通过。
审查没有产生需要继续修复的发现；审查工具临时生成的未跟踪 `uv.lock` 已精确删除。

## 2026-08-08：molecule-range ID run staging 与单区间标量校验

### 父进程小对象热点

上一轮完成短 range 生成后，`TimedOutputStore.add_molecule()` 仍为每个 range block 调用
`np.full(take, molecule_id)`，在 flush 时再把大量小 ID ndarray 拼成连续列。对
112,300 个 molecule、每个一个 range 的真实规模形态，旧路径每个 4,096-definition
batch 会同时保留 4,096 个 ID ndarray；无 bond molecule 还会额外暂存两份空 ndarray。
紧邻 cProfile 又显示，单 range 已经是规范化 `uint64` 时仍对 `start > end` 和越界分别
调用 `np.any()`；三轮共 336,900 次 `add_molecule()` 对应 673,800 次 reduction，累计
约 0.857 秒，是该反馈环中最大的可消除 Python 调度项。

先建立三个红灯：小 range block 禁止逐 molecule 调用 `np.full()` 并要求多 molecule 的
ID/run length 与 start-frame 顺序精确对齐；单 range 在禁止 `np.any()` 时仍须完成合法
写入；无 bond molecule 不得把空 bond atom/order ndarray 放进 batch。修改前三项分别
稳定触发旧分配路径。

### 最终实现与边界

- range batch 只保存 `(molecule_id, range_count)` run 描述，flush 时用一次
  `np.repeat()` 生成连续 ID 列；start/end 仍保留原有 `uint64` view，未增加复制。
- `byte_count` 继续按最终三个 `uint64` range 列的逻辑字节记账，65,536-row 和 64 MiB
  两个既有上限不变；跨 flush、同 molecule 多 block 和定义/range 分离语义不变。
- 单 range 直接把两个标量转为 Python `int`，一次检查负值、逆序和 frame 越界；空 block
  不执行 reduction，多 range block 继续走原有 NumPy 向量校验。既有 signed/unsigned
  负值、`start>end` 和越界参数化回归全部通过。
- 无 bond molecule 仍写入每个定义的累计 offset，但不再把两个空数组保留到 flush；
  一个满 4,096-definition 的 bondless batch 因此少保留 8,192 个空 ndarray 对象。非空
  bond 的 dtype、顺序和 HDF5 schema 没有变化。

### 受控性能与内存反馈

同一个 112,300-molecule、单 range、无 bond 的真实 `TimedOutputStore` 反馈环，三次连续
运行取中位；HDF5 都是 28 个 batch、最大 4,096 ranges/262,144 logical bytes，文件均为
203,835 bytes：

| 实现状态 | 完整 wall 中位 | writer 记账中位 |
| --- | ---: | ---: |
| 修改前 | 0.9346 s | 0.8938 s |
| ID run staging 后 | 0.8849 s | 0.8432 s |
| 再加单 range 标量校验 | 0.4920 s | 0.4516 s |
| 最终（跳过空 bond staging） | 0.3411 s | 0.3030 s |

最终相对修改前约降低 63.5% wall 和 66.1% writer 记账。该压力刻意放大“一 molecule/
一 range/无 bond”的父进程小对象路径，不能外推到 49,047,393 个碎片 range 的生产
吞吐；range-heavy block 仍使用 NumPy 和既有 HDF5 批量写入。

另把 ID staging 单独做 5 轮交替独立进程微基准：旧小 ndarray/新 run 的 wall 中位为
0.33425/0.18272 秒，checksum 都为 `6305701150`；`tracemalloc` peak 为
655,964/300,060 bytes，分别约降低 45.3% 和 54.3%。该 peak 只覆盖 Python/NumPy staging
微内核，不当作进程 RSS 或 Slurm MaxRSS。

### 真实轨迹、语义、回归与清理

最终源码重新处理只读 5-frame `rp3.lammpstrj`。独立进程首次 Step 3 四阶段为
0.578/0.087/0.183/0.021 秒，writer 为 0.042 秒；同进程 RDKit 热身后的四阶段为
0.185/0.086/0.185/0.020 秒，writer 为 0.043 秒。这里只报告当前值，不把 RDKit 初始化、
文件页缓存不同的两轮混成 A/B。HDF5 语义指纹继续为
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`；`.route`、
`.reactionabcd`、`.reaction` 和 `.moname` SHA-256 继续分别为
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8` 和
`173299dfc8c574e8bc060a73c6343d32cae7353702a09fee53b7d64ec505ae15`。

新增 3 项回归后，`tests/test_timedoutput.py` 收集 179 项；排除 Tk GUI 集成文件的完整
回归为 241 passed。Black、Isort、flake8、compileall 和 `git diff --check` 通过。
约 9.6 MiB 的真实输出/profile、两份精确 Detect 临时文件和 6 个本轮脚本/profile 已
删除；`rng-cutoff-*` 继续保留，`uv.lock` 不存在。只读原轨迹大小/mtime 未变，SHA-256
仍为 `c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

本轮只消除已证实的 parent-side 小对象和 reduction 开销。真实 64 核 Slurm 仍需记录
各阶段 wall、writer 属性、实际 worker、`TotalCPU/Elapsed`、MaxRSS、临时盘峰值与语义
manifest；当前工作区没有 `squeue/sacct`，不能用本地结果替代最终生产验收。

## 2026-08-08：周期 Open Babel 成键候选的有界 cKDTree 加速

### Step 1 热点复核与被否决的 ASE 原型

在前述 Step 3 已降至秒级后，重新对只读 5-frame `rp3.lammpstrj` 做逐阶段反馈。单进程
Step 1 的三次强制旧路径中位为 8.5640 秒/8.5546 CPU-s，Python cProfile 只能解释约
0.38 秒；继续拆分 Open Babel 原生调用后，首帧 `ConnectTheDots()` 约 1.41 秒，约占原生
成键阶段的 90%，`PerceiveBondOrders()` 约 0.15 秒。审计 Open Babel 3.1.1 源码确认：
周期分支仍按 z 排序后遍历所有 atom pair，再用 minimum image 和
`Rcov(i)+Rcov(j)+0.45 Å` 判断，因此是明确的 O(N²) 热点；非周期分支已有 z 方向提前终止，
本轮不改。

第一版用 ASE `neighbor_list()` 产生候选，5 帧 bond signature 与旧路径一致，Step 1 wall
降到 4.111 秒；但 CPU 达 25.422 CPU-s（约 6.18 核），MaxRSS 达 507.7 MiB。ASE 的 padded
bin 中间数组会按 bin 数和最大 bin occupancy 展开，且 NumPy/BLAS 线程可能与 frame
process 叠加；这会在 64 进程下放大 RSS 和线程超卖，因此该实现未保留。

### 最终算法与保守回退

最终只对 `pbc=True`、至少 2,048 原子、正向 axis-aligned orthorhombic cell 的默认
Open Babel 路径启用周期 `scipy.spatial.cKDTree`：

- tree 只物化全局最大共价 cutoff 内的真实候选 pair，空间复杂度为 O(N+E)，不建立 ASE
  的全格点 padded pair 数组；`query_pairs()` 为单线程，不在每个 frame worker 内再启动
  BLAS worker；
- 候选仍用 Open Babel 元素共价半径、`+0.45 Å` 上界和 `0.4 Å` 下界二次过滤；按原
  `ConnectTheDots()` 的 z/atom 顺序插入，并复现 phosphorus 第六键的 F/Cl 例外；
- 过价、45° 小键角、H-H 优先删除和最长键删除仍由同一个 Open Babel molecule 执行，
  随后继续调用原生 `PerceiveBondOrders()`；只有 O(N²) pair discovery 被替换；
- triclinic/旋转 cell、小于阈值的体系、cell 小于两倍最大 cutoff、非有限坐标、未知共价
  半径，以及距 0.4 Å 或 pair cutoff 不超过 `1e-7 Å` 的数值边界，均整帧回退历史
  `ConnectTheDots()`；非周期和显式 `use_ase` 路径不变；
- Open Babel 对完全相同 z 使用平台相关的 C++ unstable sort；最终实现以 atom ID 稳定
  打破平局。真实 5 帧每帧把全部 equal-z group 随机重排 20 次，得到的 bond signature
  各自都只有 1 种，并与旧路径一致；构造的 equal-z 过配位和 phosphorus 第六键也做了
  差分回归。

`scipy` 原本已由 HMM 依赖间接提供，本轮在 `pyproject.toml` 中把直接使用关系显式声明。
常密度子体系的 connection-only 交叉测试中，256 原子时 tree 尚慢于旧路径，512 原子
开始更快；保守阈值仍取 2,048，此时为 0.00650/0.04068 秒（约 6.25 倍）。12,326 原子
的 pair discovery + insertion + cleanup 为 0.04372/1.43687 秒（约 32.9 倍）；该比例不
包含双方共有的 bond-order perception、文本解析和 molecule DFS，不能当作完整 Step 1
加速比。

### 独立进程 A/B、内存和多进程反馈

每轮都从同一只读副本启动新进程；旧路径用运行时阈值强制回退，最终路径使用默认设置：

| 单进程指标 | 旧 `ConnectTheDots` | 最终 cKDTree | 变化 |
| --- | ---: | ---: | ---: |
| Step 1 wall 中位 | 8.5640 s（3 轮） | 1.7287 s（5 轮） | -79.8%，4.95× |
| process CPU 中位 | 8.5546 CPU-s | 1.7278 CPU-s | -79.8% |
| MaxRSS 中位 | 299.95 MiB | 303.39 MiB | +3.44 MiB，+1.15% |

最终五轮 `CPU/wall` 都约为 1.00，没有 ASE 原型的线程放大。每轮均为 5 frames、5,615
molecules、3,206,983-byte molecule temp，SHA-256 都是
`ab8145a664d5d198aa9a675cdf515c25c5d8c6d37b96902bc1941e55be72add4`。

再强制 `fork`、5 frame workers 做三组顺序交替 A/B，避免两组 benchmark 同时争用 CPU：

| 5-process 指标 | 旧路径中位 | 最终路径中位 | 变化 |
| --- | ---: | ---: | ---: |
| Step 1 wall | 2.6894 s | 0.5119 s | -81.0%，5.25× |
| child CPU | 12.1648 CPU-s | 2.1151 CPU-s | -82.6% |
| reported max child RSS | 155.23 MiB | 158.36 MiB | +3.13 MiB，+2.0% |

该 5-frame 测试最多只能同时使用 5 个 worker，不能替代 64 核长轨迹的 affinity、吞吐和
aggregate RSS 验收；但它确认 cKDTree 不再像 ASE 原型那样在每个 process 内额外吃多核。

### 完整语义、回归与清理

最终代码重新跑完整 no-HMM、molecule timeline、reaction event 流程：Step 1/2/3/4
分别为 1.496/0.038/1.000/0.108 秒，总计 2.643 秒。HDF5 计数仍为 5 frames、5,615
molecules、6,076 ranges、383 reaction types；语义指纹保持
`3381ffcc6da52aa9a008bae3316f06fcb7dbbfb7d8c571024e49e19f75bc98cb`。`.route`、
`.reactionabcd`、`.reaction`、`.moname` SHA-256 分别保持
`f7a7f8e223ffb1e5a7908a23919b60af8210a5c472851e89b93b7fc639f7ecb3`、
`2f4a345b48d8544c08ca9713199634fa0dd6cf273ab0ef9219de64c87d532d56`、
`ea0c350219f9ce7e08d76c8d7350e9d5b49c93771804f784d9e576c7737a9cb8`、
`173299dfc8c574e8bc060a73c6343d32cae7353702a09fee53b7d64ec505ae15`。

新增 13 项周期成键回归覆盖 orthorhombic boundary、equal-z、过配位清理、phosphorus、
随机 C/H/N/O/F/P/Cl 差分、数值 cutoff、small/triclinic cell、阈值调度和“不调用 ASE
bin”。排除 Tk GUI 集成文件的最终回归为 254 passed；Black、Isort、按项目忽略
E203/E501/W503 的 flake8、`compileall` 和 `git diff --check` 通过。

约 294 MiB 的 Open Babel 源码审计副本、轨迹副本、完整输出、profile、A/B 脚本和 27
个精确列出的 Detect 临时文件已删除，均不可恢复但可由只读输入重新生成；既有
`rng-cutoff-*` 全部保留，`uv.lock` 不存在。原始 `rp3.lammpstrj` 的大小/mtime 仍为
2,083,407/1,778,844,745，SHA-256 仍为
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。

真实 64 核 Slurm 仍是最终门槛：必须记录 accelerator 是否因 cell/cutoff 回退、实际
frame worker、Step 1/Step 3 wall、`TotalCPU/Elapsed`、MaxRSS/aggregate memory、临时盘
峰值、共享文件系统吞吐和语义 manifest。本地结果支持重新提交验证，不支持直接宣称
全轨迹会按 4.95× 缩短。

### 生产命中观测与 LAMMPS 数组解析

完成算法修复后又补齐生产可验证性：每个 frame result 只增加一个 small-int mode（pickle
记录由 23 增至 25 bytes），父进程用固定大小 `Counter` 聚合，不保存逐帧 mode 列表。
首帧完成后立即报告所用模式，Step 1 结束时汇总 `periodic-cKDTree`、ASE、非周期、低于
阈值、unsupported/narrow cell、无效坐标/半径和 cutoff-boundary 数量。真实 5 帧日志为
`Coordinate bond perception: 5 frame(s); periodic-cKDTree=5`，因此生产复跑可直接确认是否
实际命中，而不再只能从总耗时反推。

修复后 profile 的 Python 可见部分仅约 0.398 秒；剩余约 1.31 秒主要在 Open Babel 原生
bond-order perception，继续替换会扩大化学语义范围，本轮停止该方向。可见热点中，旧
LAMMPS parser 每 5 帧创建 61,630 个 ASE `Atom` 对象并再按 ID 排序。现在预分配紧凑
`int16 atomic_numbers`、`float64 positions` 和一个 bool seen bitmap，按 atom ID 直接
填充后一次构造 `Atoms`；同时明确拒绝重复、缺失、越界 ID 和越界 atom type，避免损坏
帧留下未初始化数组。乱序 `water.dump` 回归禁止调用 `Atom()`，并核对 ID 顺序、元素、
坐标和 cell。

工具额度限制阻止了为 parser 单独创建交替 A/B 临时脚本，因此不宣称独立加速比例；最终
真实单次 Step 1 为 1.409 秒，仍输出 5 frames、5,615 molecules、3,206,983 bytes，SHA-256
保持 `ab8145a664d5d198aa9a675cdf515c25c5d8c6d37b96902bc1941e55be72add4`。新增观测和数组解析后
非 GUI 回归为 255 passed，`compileall` 与 `git diff --check` 通过；格式化后继续执行最终
静态复核。真实 Slurm 应保存首帧模式和最终模式计数，若不是全部 cKDTree，应先按原因
解释回退，再比较 CPU/RSS。

最终又以真实 5-frame 输入强制 `fork`/5 workers 做 IPC smoke test；父进程先报告首帧
`periodic-cKDTree`，结束时汇总 `periodic-cKDTree=5`，molecule temp 大小和 SHA-256 仍为
3,206,983 bytes / `ab8145a664d5d198aa9a675cdf515c25c5d8c6d37b96902bc1941e55be72add4`。

本补充验证新增的 4 个 3,206,983-byte molecule temp 和 176 KiB font cache 原计划精确
删除，但删除操作被当前工具额度限制拒绝，未绕过限制；暂留路径为
`tmpkihy3rk1`、`tmpe1_tmgw6`、`tmpuidbbfl9`、`tmp4tkfniky`（系统临时目录）及
`/private/tmp/rng-step1-postfix-mpl`，合计约 13 MiB。它们均为可重建临时产物，不属于
工作区；此前已清理的约 294 MiB 产物和保留的 `rng-cutoff-*` 状态不变。

## 2026-08-08：64 核生产验收脚本与证据闭环

本地环境没有 `squeue`/`sacct`，无法把 5-frame 和 50k-frame 反馈环冒充真实全轨迹验收。
为消除下一次提交时命令、输入、commit、affinity、临时盘和 manifest 证据再次缺失的问题，
新增 `scripts/slurm-production-validation.sh` 和 `docs/slurm-production-validation.md`。

脚本只固定与本次验证有关的安全默认值：一个 task、64 CPU、同时生成 molecule timeline
和 reaction event；partition、account、memory 和 time limit 仍由提交命令明确提供。每次
提交创建独立的 `run-<job-id>`，把轨迹以只读符号链接接入，拒绝覆盖已有目录。它保存输入
SHA-256、git commit/脏状态、实际 module 路径、Slurm/affinity、精确命令、阶段日志、GNU
time、临时目录 60 秒采样、HDF5 semantic manifest、全体顶层文件 SHA-256 和作业结束后
应执行的 `sacct` 命令。若提供 baseline manifest，HDF5 内容不一致会使作业非零退出。

为避免 64 个 Python process 各自再启动 BLAS/OpenMP 线程，三个常见底层线程变量只在
用户未显式设置时取 1；`TMPDIR` 优先使用节点的 `SLURM_TMPDIR`。包装脚本禁止在透传参数
中重新指定 input、atom、nproc 和 timed-output，防止科学输出逃出验收目录或元数据与实际
命令不一致。它不取消旧作业、不提交新作业，也不声称本地已经完成生产验收。

本节新增的是可重复验证能力，不是新的算法加速数据。最终门槛仍是用相同全轨迹和计算
参数完成至少一轮候选，并保存最终 `sacct`；若 16/64 核 wall 接近，应按 core-hours 选择
16 核，而不是为表面利用率强制更多 worker。

### 包装脚本 smoke test

使用仓库内乱序 atom-ID 的 1-frame `tests/inputs/water.dump`（338 bytes，SHA-256
`50be7f3c00549529168eed671c5787044a7aca083f06ced2ab6a5233e23a319d`）以相对
`RNG_PYTHON=./.venv/bin/python`、`nproc=1`、`H,O`、no-HMM 完整执行两次。首次试跑在计算
启动前发现 metadata probe 多转义了一层引号；修正为不导入包的 `importlib` 元数据查询，
并把相对 Python executable 在切换输出目录前解析为绝对路径。

修正后第一次无 baseline、第二次带第一次 manifest 作为 baseline，均完成 Step 1--6、
wrapper/HDF5 validator 退出码均为 0。第二次 run directory 共 22 个哈希文件，HDF5 状态为
`complete`，计数为 1 frame、3 atoms、1 molecule、1 range、0 reactions，semantic
fingerprint 为 `7430d18b5ff137a9caa19812b3a2e186da4011b03f58419fea2e99a643c8bd02`。
结果目录、命令、输入 SHA、git 状态、metadata、临时盘采样、manifest、validation log、
final status 和 output hashes 均按设计生成；第二次复用显式字体缓存后包装 wall 为 3.05
秒，其中 RNG 自报总计算 0.741 秒，该值只用于验证包装路径，不作为算法基准。

另验证透传 `--timed-output=escape.h5` 会在建立 output root 前以退出码 2 拒绝；无 Slurm
环境且未给 `--nproc` 时改为安全回退 1。`bash -n` 和 `git diff --check` 通过，本机未安装
ShellCheck。smoke test 的两份成功结果和一份启动前失败目录合计约 4.3 MiB，已按精确的
`/private/tmp/rng-slurm-wrapper-smoke-20260808` 测试根目录清理；拒绝测试的一份文本也已
删除。清理未触碰既有 `rng-cutoff-*` 或此前因工具限制保留的临时文件。

包装修改后的最终非 GUI 回归使用已有 Anaconda 测试环境执行，结果为 255 passed；此前
第一次尝试的项目 `.venv` 没有安装 pytest，命令在 collection 前退出，不属于测试失败。
`compileall`、`bash -n`、`git diff --check` 继续通过，`uv.lock` 不存在。原始
`rp3.lammpstrj` 的大小/mtime/SHA-256 继续为
2,083,407 / 1,778,844,745 /
`c7d7dce618f52b8e96451b4194c67ce63c976d22380e3b6d8c936bcb17503e85`。
按既有约束未再次尝试删除的五个旧临时路径仍存在，大小分别为 176 KiB 和四份约 3.1 MiB；
本轮新 smoke 目录已不存在。

### 10,001 帧包装验证与 BSD time 退出码修复

完成性审计确认本机没有 `sbatch`、`squeue`、`sacct` 或 `sstat`，但仍保留一份真实
10,001-frame、450-atom 甲烷燃烧轨迹。因此用包装脚本、8 processes、no-HMM 首次执行
生产形态 smoke。计算本身完整跑完 Step 1--6：Step 1 为 18.487 秒、Step 2 为 3.305 秒、
Step 3 为 5.083 秒，其中 molecule/matrix/route/reaction 分别为
0.595/0.112/3.109/1.225 秒，总计算为 27.564 秒。输出日志显示 3,154 molecules、
154,108 ranges、360,249 modified atom events 和全部 10,000 active transitions；这与此前
同输入反馈环的数量一致。

该次包装最终退出 1，不是 RNG 计算错误：macOS `/usr/bin/time -l` 在沙箱内无法读取
`kern.clockrate`，即使被测程序退出 0 仍返回 1，导致 wrapper 在 HDF5 manifest 前停止。
为避免资源采集器覆盖科学程序退出码，BSD 分支现在由一个最小 Bash wrapper 单独写出
`run-status.raw`；外层优先使用该值，只有子进程未能留下状态（例如被强杀）时才回退到
time 的退出码。GNU time/Slurm 路径不变。修复后还需重跑一次同轨迹，确认 manifest 和
最终 wrapper status 闭环后才记录为成功。

修复后的第二次 10,001-frame 运行提供第一次完成 HDF5 生成的 manifest 作为 baseline；
该 baseline 自身先与历史记录核对。两次 HDF5 semantic fingerprint 都为
`7dc0197f3618a2d34fce5c3c6283260f425520510905d62c6a4547c3799d0d38`，molecule/range/
reaction type/compressed reaction row 计数均为 3,154 / 154,108 / 497 / 43,588；`.moname`、
`.route` 和 `.reactionabcd` SHA-256 也分别精确保持
`5c8a1117fed6fccf5b2a5e65ce945d7cc56c739295d5e82906dc1bb7f9d9fee6`、
`af09a4386b071bc6308bf6f3cc6f1828878fdbd99b0de6acb86844d8534b77a4`、
`af8303960eaa4118059d6d9211cf056e19486938cedd07998021a3bc1d7f1654`。候选 validator、
baseline comparison 和最终 wrapper 均退出 0，`validation.log` 为空，说明没有 mismatch。

第二次阶段 wall 为 Step 1/2/3 = 18.042/3.889/5.954 秒，Step 3 的
molecule/matrix/route/reaction = 0.631/0.120/3.813/1.344 秒；完整 RNG 自报 28.719 秒。
BSD time 留下 30.53 real、157.66 user、17.28 sys，约 5.73 个平均 CPU，但因本机
`kern.clockrate` 权限仍以 1 退出且没有 MaxRSS；该状态现在单独记录为
`resource_time_exit_status=1`，科学程序和 wrapper 仍明确为 0。1 秒临时目录采样峰值为
14,344 KiB，完成后临时目录为 0；这只是 450 × 10,001 本地结果，不能外推到旧 3 TB OOM
规模。

随后把同一 raw-status wrapper 同时应用到 GNU time 分支，使生产 Linux 也不会因计量工具
自身失败覆盖已明确写出的程序状态。再次用 1-frame water smoke 验证：science=0、
resource-time=1、wrapper=0 且 manifest 存在；`bash -n` 和 `git diff --check` 通过。
10,001-frame 原输入大小/mtime/SHA-256 未改变。上述两次大轨迹结果、一次 water 结果、
24 MiB 验证目录、432 KiB 字体 cache 和 4 KiB time probe 已按精确 `/private/tmp` 路径
清理；原轨迹和旧保留物均未触碰。

## 2026-08-08：n3 上的 48 CPU Slurm 生产形态验证

### 节点、提交链路与环境引导

按用户提供的 SSH host `n3`，使用 DPDispatcher 1.0.3 的 SSHContext/Slurm 后端提交。
`node3` 是单节点、48 个物理 CPU（AMD EPYC 7K62，threads/core=1）、约 240 GiB Slurm
内存，因此本节点的满节点验证必须是 48 CPU；64 CPU 不是该节点可提供的 allocation。
DPDispatcher 默认把 `cpu_per_node` 写成 `--ntasks-per-node`，提交驱动在其后显式覆盖为
`--ntasks=1 --ntasks-per-node=1 --cpus-per-task=N`。任务 3615/3619/3621 的 `scontrol`
均确认 `NumTasks=1`、`CPUs/Task=48`、`TRES=cpu=48`，程序内再启动 Python workers。

首次环境引导任务 3608 因 conda 的 ReacNetGenerator 1.6.15 包没有自动带入 `h5py` 而
失败；当时 DPDispatcher 默认 retry 又产生 3609--3611。之后把提交驱动默认
`retry_count` 改为 0，并在引导脚本中显式安装 ase、h5py、hmmlearn、OpenBabel、RDKit、
NumPy、SciPy 等完整运行依赖。任务 3612 成功，使用 Python 3.12.13、h5py 3.16.0、
NumPy 2.5.1、SciPy 1.18.0 和已安装包的 Linux `dps.abi3.so`；Python 源码则来自当前
工作区快照。所有提交及远端工作目录均保留，未取消或改动用户既有作业。

新增 `scripts/dpdispatcher-submit-slurm.py`、
`scripts/bootstrap-remote-reacnetgenerator.sh` 和
`scripts/run-remote-reacnetgenerator-validation.sh`，并增强生产包装：GNU time 识别忽略
大小写，记录 `scontrol` 起止快照、`sstat`、临时盘和进程树聚合 RSS。n3 关闭了 Slurm
accounting，`sacct` 不可用，`sstat` 的 AveCPU 也返回无效超大值；因此本轮 CPU 采用 GNU
time 的 user+system/wall，内存同时报告 GNU time 单进程 MaxRSS 和 5 秒进程树 RSS 采样。
后者会重复计算 fork 共享页，只能视为保守聚合指标，不能冒充 USS 或 Slurm MaxRSS。

### 10,001 帧的 8/48 CPU 对照与确定性

输入为只读 `ch4_o2_3000K.lammpstrj`，135,333,201 bytes、10,001 frames、450 atoms，
SHA-256 为
`20f0fee8a53778e0a3a6929032f8f57a1c5f53908ef8ddf6e53838b35092de06`。no-HMM、timeline
和 reaction event 参数保持一致，BLAS/OpenMP 线程固定为 1。

| Job | CPU | RNG wall | GNU wall | 平均 CPU | Step 1 | Step 3 | 单进程 MaxRSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3616 | 8 | 23.731 s | 25.97 s | 6.92 cores | 20.785 s | 1.451 s | 263,780 KiB |
| 3615 | 48 | 9.176 s | 未采集 | 未采集 | 5.962 s | 1.467 s | 未采集 |
| 3619 | 48 | 9.129 s | 11.40 s | 17.30 cores | 5.957 s | 1.473 s | 265,064 KiB |

48 CPU 相对 8 CPU 把 Step 1 wall 缩短约 71.3%（3.49×），完整 RNG wall 缩短约 61.5%
（2.60×）；代价是 48 CPU 任务的 allocation core-seconds 更高。Step 3 的 8/48 wall
几乎相同不是 48 进程失效：该 10k 工作量自动选择 1 个 SMILES worker、8 个 route worker
和 7 个 reaction worker，避免为很短的子阶段支付 48 个进程的调度与 IPC 成本。任务
3619 的进程树 RSS 采样峰值为 8,342,684 KiB，包含共享页重复计数；没有 OOM 或 swap。

三次 n3 输出的 HDF5 语义指纹都为
`79ebc6f5748ad7d46f585f239fcb65a65fb155a7088f7fe1362c661bdf4b23f8`，计数都为 3,156
molecules、154,109 ranges、499 reaction types、43,589 compressed reaction rows。
`.moname`、`.route` 和 `.reactionabcd` SHA-256 逐字节相同，分别为
`9887abea6eb35006cf9c05170d709f38746db2f623b875d7d1617cd0b722e282`、
`0d8f841ff326ba04d031e060fd01d7667e52d2bbeba93c943bc2f82e495aed96`、
`ff2bdf6078401c9975553313bf68e7d2df44f3fa33ee338257456a2ac0a12290`。这直接排除了
8/48 进程数引入的非确定性。

n3 的结果与此前 macOS 基线相差 2 个 molecules、1 个 range、2 个 reaction types 和 1
个 compressed reaction row。输入哈希相同，但本地 OpenBabel 为 3.2.1，n3 为 3.1.0；
全部 10,001 帧都走 OpenBabel bond perception。因此跨平台指纹不能直接互作严格基线，
该小差异已有明确依赖版本因素，不能归因于 48 进程或 HDF5 批量 writer。生产的严格候选/
基线比较必须固定同一 OpenBabel 版本和平台。

### 50,005 帧的 48 CPU 扩展结果

任务 3621 把同一只读输入按顺序重复 5 次，`scontrol` 和程序日志都确认 48 CPU、affinity
0--47。HDF5 validator 完成，`.route`、`.reactionabcd` 和报告均生成，wrapper/science/
GNU time 均退出 0。结果如下：

- RNG wall 36.178 秒，GNU wall 38.44 秒，user+system 938.40 CPU-s，平均 24.41 cores，
  即整个 48 CPU allocation 平均约 50.9%；
- Step 1/2/3 为 28.595/0.701/3.581 秒；Step 3 的 molecule/matrix/route/reaction 为
  0.456/0.419/0.534/2.068 秒；
- reaction 根据 1,802,953 modified events 自动选择 36 workers；route 选择 8 workers；
- 770,489 molecule ranges 只用 12 个写批次，217,953 compressed reaction rows 用 54 个
  写批次；timed-output 累计 write wall 为 1.720 秒，没有出现逐条 resize/write 的长尾；
- GNU time 单进程 MaxRSS 为 320,456 KiB；进程树 RSS 采样峰值 9,538,724 KiB（共享页重复
  计数）；临时目录采样峰值 18,248 KiB；无 swap、OOM 或错误日志；
- manifest 语义指纹为
  `3040b7cb08bbb930e9d3216fb781a324b3df2928cae319659f32501ad94f3ea1`，完整计数为
  50,005 frames、7,939,090 logical molecule rows、770,489 ranges、500 reaction types、
  217,953 compressed reaction rows；
- `.moname`、`.route`、`.reactionabcd` SHA-256 分别为
  `9887abea6eb35006cf9c05170d709f38746db2f623b875d7d1617cd0b722e282`、
  `d48b4bcd0ff3225da65abb6962d91837e3cd60a4e309efa4ef85592855f78345`、
  `d6929db0930c382f008c9bdc23a946f26d22519907906ae9f409b9af4e9e347f`。

相对 10,001 帧，帧数放大 5× 时 Step 3 只放大约 2.43×；molecule writer 从约 0.333 秒
增至 0.456 秒，说明当前批量 range/HDF5 路径没有复现旧作业的数十小时串行写入长尾。
这证明修复在 n3 的 48 CPU 生产形态下可运行、可扩展且结果对进程数确定；但测试只有
450 atoms/50k frames，仍不能用 38 秒反馈环直接外推旧全轨迹的绝对完工时间。对旧作业
是否取消的决策应结合剩余 Slurm 时限：若仍处于连续数十小时约 2.1 cores 的旧 Step 3，
当前数据支持取消后用修复版重算；若旧作业已接近生成最终 `.route`，则先保留其输出作为
同平台基线再切换。

本地回传结果保存在 `/private/tmp/rng-dpdispatcher-n3-20260808`；为防 DPDispatcher 在重复
输入测试中把只读结果 symlink 解引用并下载五份 135 MB 轨迹，远端 wrapper 完成验证后只
unlink 自己生成的五个结果目录输入链接，原输入、远端 task input 和所有科学输出均未修改。

n3 验收后重新运行全部非 GUI 测试文件，结果为 255 passed in 25.83s；`compileall`、三个
shell 脚本的 `bash -n`、DPDispatcher 驱动的 Black/py_compile 和 `git diff --check` 均
通过。本机仍未安装 ShellCheck。本轮没有生成 `uv.lock`，也没有删除此前按约束保留的五个
旧临时路径。

### code-review 后续：同环境固定点与 range 量级补测

按 `code-review` skill 以 `b332ce99` 为固定点复核工作树。Standards 轴没有仓库规范 hard
violation，给出三项 judgement call：bond-detection mode 裸整数、unsigned dtype 选择重复、
四阶段 pool-plan 形态分散。前两项已修复：mode 改为携带生产日志 label 的 `IntEnum`，同时
保留既有内部常量兼容；molecule name/matrix/path 统一使用“可容纳最大值”的 unsigned dtype
helper，调用点显式做 count-to-maximum 换算。跨四模块统一 `PoolPlan` 会改变大量调度入口，
在生产压力任务前引入的回归面过大，本轮只记录，不做无性能证据的结构重构。

Spec 轴指出 50,005-frame 只有约 77 万 ranges，尚未覆盖旧问题约 4,900 万 ranges；还缺
n3 同依赖环境的固定点输出对照，且此前 Step 3 实际最多 36 reaction workers。为处理这些
缺口，新增 320× 输入压力任务 3631（3,200,320 frames，目标约 4,900 万 canonical ranges），
请求一个 task/48 CPUs；最终结果见下节的代表性压力验收。Step 1 periodic-cKDTree 是独立
的化学成键加速与风险面，最终交付应与 Step 3 writer 修复拆成独立逻辑 commit/评审单元，
不用其收益掩盖 Step 3 结论。

固定点源码由 `git archive b332ce99` 得到，只补入当前 validator 和未被 Git 跟踪的静态
HTML bundle，不改科学计算模块。第一次任务 3626 在 Step 5 后因缺少 bundle 失败；第二次
任务 3630 完成 Step 1--6，科学程序退出 0，但旧 HDF5 schema 缺少当前 validator 要求的
`molecule_count` 等属性，wrapper 最终按兼容错误退出。必要 HDF5/文本/资源文件已从保留的
远端目录精确回传，没有回传或修改轨迹链接。

同一 n3/OpenBabel 3.1.0/48 CPU 环境下，`b332ce99` 的 Step 1/2/3 为
6.518/0.624/57.925 秒，完整 RNG 为 66.453 秒；当前任务 3619 为 5.957/0.425/1.473 秒，
因此 Step 3 wall 缩短约 97.5%（39.3×）。固定点 `.route`、`.reactionabcd` 与当前逐字节
相同；`.moname` 都有 3,156 行，排序后的行多重集合完全相同，只有旧并行完成顺序与当前
确定顺序不同。只读兼容比较进一步得到：固定点 164,885 个物理 ranges 合并乱序/相邻分段
后恰好为 154,109，当前也是 154,109；两者 logical molecule rows 都为 1,587,818，frames、
分子定义及 frame coverage、reaction events 全部相等。比较脚本及回传证据保存在
`/private/tmp/rng-dpdispatcher-n3-20260808`。

validator 现在把同一 molecule 内有序且相邻的 ranges 归一化后计算语义 fingerprint，并
继续拒绝 overlap/逆序；新增回归确认物理 1/2 个 range 表示相同连续 coverage 时 semantic
fingerprint 和 manifest compare 一致。code-review 修复后的非 GUI 回归为 256 passed in
26.55s。

### 3,200,320-frame / 49,310,414-range 的 48 进程压力验收

任务 3631 把同一 10,001-frame 真实轨迹按顺序重复 320 次，形成 3,200,320 frames、
508,101,760 molecule-frame occurrences 和 49,310,414 canonical molecule ranges。该用例
的 range 数与旧 5% 作业的 49,047,393 基本同量级，专门闭环旧 Step 3 串行 range/HDF5
长尾；重复输入不能替代真实全轨迹的化学分布验收，因此这里只称“代表性 range 规模”，
不称真实全轨迹。

Slurm 配置为一个 task、`CPUs/Task=48`、affinity `0-47`，程序 `nproc=48`。Step 1 日志
确认 48 workers；Step 3 reaction 日志也实际确认
`48 workers, chunksize=32, max_inflight=3072`，不再只是传入 48 的参数测试。主要 wall
如下：

| 阶段 | wall |
| --- | ---: |
| Step 1 | 2102.599 s |
| Step 2 | 119.126 s |
| Step 3 molecule | 7.846 s |
| Step 3 matrix | 18.692 s |
| Step 3 route | 27.076 s |
| Step 3 reaction | 101.086 s |
| Step 3 total | 156.024 s |
| Step 4 | 163.036 s |
| RNG total | 2541.378 s |

49,310,414 ranges 只用 753 个 HDF5 batches，molecule pipeline 的 6.642 秒记账时间中最大
单批为 65,536 ranges / 1.513 MiB；13,949,118 compressed reaction rows 使用 3,406 个
batches。route 对 1,440,144,000 atom-frame values 使用 8 workers，在 27.076 秒内把
115,415,578 event rows 压为 17,239 pair rows；reaction 对 3,200,319 active transitions
使用全部 48 workers。Step 3 总计 2.60 分钟，没有复现旧作业连续约 48 小时、约 2.1 核
的串行长尾。

科学程序与 wrapper 均退出 0，`.route`（2,346,646,310 bytes）、`.reactionabcd`、报告和
HDF5 均生成。validator 得到 3,156 molecules、500 reaction types、13,949,118 compressed
reaction rows，HDF5 `status=complete`，语义指纹为
`90e9618925ec7eab870235643a35f08bc704a5e14669408e5cbdc25f8dee012b`。回传后
`output.sha256` 全部逐文件校验通过。

GNU time 为 42:23.98 wall、61,057.74 user seconds、943.06 system seconds，平均约
24.37 cores，即 48 CPU allocation 的约 50.8%；单进程 MaxRSS 为 4,002,892 KiB，swap
为 0。5 秒进程树 RSS 求和峰值约 160.2 GiB，但会把 fork 共享页重复计算；运行中一次
49-Python-process PSS 抽样仅 4.746 GiB（private 3.284 GiB），节点 `free` 同期 used 约
9--13 GiB、available 235--238 GiB。因此不能把聚合 RSS 当作实际物理内存。节点临时目录
采样峰值 4,929,212 KiB，结束时只余 120 KiB。

完整回传证据保存在
`/private/tmp/rng-dpdispatcher-n3-20260808/validate-48-3m/results/run-3631`，约 2.9 GiB；
远端保留目录为
`/home/chuang/dpdispatcher_works/9277de915148bd8f16124ce60fa0baa9ee73096c/validate-48-3m`。
这项结果处理了 code-review Spec 轴的 range 量级和实际 48 reaction workers 两项缺口；
同环境固定点语义与 39.3x Step 3 对照由任务 3630/3619 提供。真实全轨迹仍应按生产包装
再跑一次，Step 1 periodic-cKDTree 变更也继续作为与 Step 3 writer 分离的逻辑评审单元。

回传后又用当前工作树（包含 adjacent-range canonicalization）的 validator 重读 HDF5 并以
远端 manifest 为 baseline；退出 0，生成的 JSON 与远端文件逐字节相同，SHA-256 都为
`8f5279bfc2c3ca19bd4028cb0647f1bb35cf4ed6a3ddb5d50d8956732d6e09b0`。最终核心非 GUI/
非 benchmark 回归为 261 passed、48 deselected（16.74 秒）；本次修改的 18 个 Python 文件
通过 Black，isort、compileall、三个 shell 脚本的 `bash -n` 和 `git diff --check` 通过。
直接运行整个仓库的系统 Black/flake8 会命中未修改的历史文件和 vendored `node_modules`，
不作为本 diff 的失败；先前针对修改集的 flake8 结果仍为通过。

### Step 3 与 Step 1 的独立提交边界

按 code-review 的 P2 scope finding 将算法变更拆成独立本地提交：

- `1aa2e370`（`perf: optimize Step 3 streaming and timed output`）包含 Step 3、HMM/matrix/
  route/reaction、timed-output validator 和共享有界多进程基础设施，不含 cKDTree、
  periodic-neighbor 或 SciPy 新依赖；从该提交单独导出、使用旧 `_detect.py` 运行核心测试，
  结果为 220 passed、48 deselected；
- `96768c04`（`perf: accelerate Step 1 molecule detection`）只包含 SciPy 直接依赖、
  `_detect.py` 的 Step 1 检测/压缩/解析/cKDTree 和 `tests/test_detect.py`，专用测试为
  58 passed；
- DPDispatcher/Slurm 脚本和本地性能证据保留在后续 docs 提交，不参与两个算法提交的
  回退边界。

因此 Step 1 若在后续大原子体系验证中出现化学语义差异，可以单独 revert `96768c04`，
不会撤销已经在 n3 上验证的 Step 3 修复。
