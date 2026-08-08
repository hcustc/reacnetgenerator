# 全轨迹 Slurm 生产验证

本地回归已经覆盖输出语义、内存上界和多进程调度，但不能替代真实 Slurm 节点上的 CPU
affinity、共享文件系统和全轨迹规模。仓库中的
`scripts/slurm-production-validation.sh` 将一次复算隔离到新的作业目录，保存代码、输入、
命令、资源和 HDF5 语义证据，避免只凭 `top` 或单个 CPU 百分比判断修复效果。

## 提交前

先在当前分支安装或以 editable 模式安装项目，并固定 Python 路径。若旧作业能完成，可先
为其最终 HDF5 建立可信基线：

```bash
/path/to/venv/bin/python -m reacnetgenerator.timedoutputcheck \
    /path/to/trusted.timeline.h5 \
    --output /path/to/trusted.timeline.manifest.json
```

基线只比较规范化语义；默认不要求 HDF5 中记录的输入路径逐字相同。输入轨迹、atom type
顺序、HMM 开关、cell、抽帧和 split 等计算参数仍必须与候选作业一致。

## 提交作业

分区、账号、wall-time、内存和输出路径依集群设置，不写死在仓库。下面的 `--` 之后直接
传给 ReacNetGenerator；示例沿用 no-HMM，如原作业启用 HMM，应删除 `--nohmm`：

```bash
RNG_PYTHON=/path/to/venv/bin/python \
sbatch \
    --partition=PARTITION \
    --account=ACCOUNT \
    --time=TIME_LIMIT \
    --mem=MEMORY \
    --cpus-per-task=48 \
    scripts/slurm-production-validation.sh \
    --input /path/to/full.lammpstrj \
    --output-root /path/to/validation-results \
    --atoms C,H,O \
    --type dump \
    --baseline /path/to/trusted.timeline.manifest.json \
    -- --nohmm
```

多段轨迹可重复给出 `--input`，顺序就是拼接顺序。`--atoms` 使用逗号分隔。需要 cell、
step interval 或 split 时也放在 `--` 后，例如 `-- --nohmm --stepinterval 2`。包装脚本管理
输入、atom、进程数和 timed-output 参数，拒绝在 `--` 后重复这些参数，避免验证产物写到
未跟踪位置。

### 通过 DPDispatcher 提交 n3

仓库同时提供基于 [DPDispatcher](https://github.com/deepmodeling/dpdispatcher) 的 SSH/Slurm
驱动。先把当前源码放在本地 staging task 的 `source/`，轨迹放在 `input/`；驱动会把一个
Slurm task 映射为 `--cpus-per-task=48`，程序内部再启动 48 个 worker。环境首次引导示例：

```bash
uv run --no-project --with dpdispatcher python \
    scripts/dpdispatcher-submit-slurm.py \
    --local-root /path/to/staging \
    --task-work-path bootstrap \
    --command 'bash bootstrap-remote-reacnetgenerator.sh /home/chuang/miniconda3/envs/reacnetgenerator-perf' \
    --forward bootstrap-remote-reacnetgenerator.sh \
    --backward bootstrap-report.txt \
    --backward bootstrap-explicit-spec.txt \
    --cpus 1 --memory-gib 8 --time-limit 02:00:00 \
    --job-name rng-env-bootstrap --retry-count 0
```

48 CPU 候选示例：

```bash
uv run --no-project --with dpdispatcher python \
    scripts/dpdispatcher-submit-slurm.py \
    --local-root /path/to/staging \
    --task-work-path validate-48 \
    --command 'bash source/scripts/run-remote-reacnetgenerator-validation.sh /home/chuang/miniconda3/envs/reacnetgenerator-perf input/trajectory.lammpstrj 1' \
    --forward source --forward input \
    --backward results --backward environment.txt --backward source.sha256 \
    --cpus 48 --memory-gib 220 --time-limit 06:00:00 \
    --job-name rng-48-validation --retry-count 0
```

提交脚本默认保留远端工作目录，科学或环境错误不会自动重试。严格比较时必须让候选和基线
使用相同版本的 OpenBabel 等键感知依赖；相同输入在不同 OpenBabel 版本上可能产生少量
分子/反应差异。

#### n3 已完成的 48 CPU 代表性规模结果

任务 3631 使用一个 Slurm task、48 CPUs/Task，把 10,001-frame 真实输入顺序重复 320 次，
生成 3,200,320 frames 和 49,310,414 canonical molecule ranges。该 range 数与旧问题的
49,047,393 同量级，但重复轨迹不能替代最终真实全轨迹验收。

Step 3 总计 156.024 秒，其中 molecule/matrix/route/reaction 分别为
7.846/18.692/27.076/101.086 秒；molecule writer 用 753 batches，reaction writer 用
3,406 batches。Step 1 和 reaction 日志均实际确认 48 workers。科学程序、wrapper 和
HDF5 validator 均退出 0，最终 `.route` 约 2.35 GB，manifest `status=complete`。完整证据
已回传到
`/private/tmp/rng-dpdispatcher-n3-20260808/validate-48-3m/results/run-3631`；详细资源与
语义数据见 `docs/step3-performance-optimization-log.md`。

脚本在 Slurm 内默认采用 `SLURM_CPUS_PER_TASK` 作为 `--nproc`，脱离 Slurm smoke test 时
回退为 1。n3 的单节点上限为 48 CPU；做较低 CPU/48 CPU 资源对照时，应分别改变
`sbatch --cpus-per-task`，让 allocation 与默认 `nproc` 一致。显式 `--nproc` 只用于诊断
“同一 allocation 内限制 worker”的效果，不能当作较小 allocation 的资源成本。
`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 和 `MKL_NUM_THREADS` 在未设置
时固定为 1，防止每个 Python worker 再启动一组底层线程。若集群提供 `SLURM_TMPDIR`，
中间 memmap 自动放入节点临时盘；否则放在该次结果目录内。可用
`RNG_MONITOR_INTERVAL_SECONDS` 调整默认 60 秒的临时目录采样间隔。包装脚本只在调用者
未设置时为 `MPLCONFIGDIR` 建立作业级缓存；希望多轮不同 CPU 数测试复用字体缓存时，可将
它显式指向同一个已预热、可写目录。

## 产物和验收

每次提交创建 `run-<job-id>`，已有目录绝不复用。输入以只读符号链接接入，科学结果、日志
和 manifest 都留在该目录；原始轨迹不会被复制或修改。关键文件包括：

- `metadata.txt`：commit、软件位置、CPU affinity、Slurm allocation 和线程设置；
- `git-status.txt`、`command.txt`、`input.sha256`：代码脏状态、精确命令和输入哈希；
- `run.log`、`resource-time.txt`：阶段 wall、实际 worker、writer 统计，以及 Linux GNU
  time 或本地 macOS BSD time 可用时的进程资源指标；
- `run-status.txt`、`run-status.raw`：科学程序与资源采集器的退出状态；采集器失败不会
  覆盖已经明确写出的科学程序状态；
- `tmp-usage-kib.tsv`：中间目录用量采样；
- `process-stats.tsv`：包装进程及其全部后代的进程数、聚合 RSS/VSZ 和单进程最大 RSS
  采样，用于 accounting 关闭时补足节点内存证据；
- `slurm-stats.tsv`、`slurm-job-start.txt`、`slurm-job-end.txt`：Slurm 作业运行中的
  `sstat` 采样及 allocation 快照；若 accounting 关闭，`sstat` 数值可能不可用，应改用
  GNU time、进程树采样和 `scontrol` allocation，不得把异常值当作 CPU 证据；
- `candidate.timeline.manifest.json`、`validation.log`：HDF5 完整性与基线语义比较；
- `output.sha256`、`final-status.txt`：顶层文件哈希和最终退出状态；
- `post-job-sacct-command.txt`：作业结束后应执行的最终 Slurm 计量命令。

若集群启用了 accounting，作业结束后执行该文件中的 `sacct` 命令并保存输出；n3 当前
关闭 accounting，应保存 GNU time、`process-stats.tsv` 和 `scontrol` 快照。只有同时满足
以下条件，才算生产验证通过：

1. 作业和 HDF5 校验退出码均为 0，manifest 与可信基线一致；
2. `.route`、`.reactionabcd` 等最终输出存在，日志没有 incomplete/failed 状态；
3. 启动日志中的 visible affinity 与 `SLURM_CPUS_PER_TASK` 一致；
4. Step 1 日志确认 `periodic-cKDTree` 命中率；若回退，按日志原因解释；
5. 对比旧作业和较低 CPU/48 CPU 候选的各阶段 wall、`TotalCPU`、MaxRSS、临时盘峰值及总
   core-hours，而不是只看整个作业的平均 CPU；
6. 若较低 CPU 与 48 CPU wall 接近，生产配置选择较低 allocation；若 48 CPU 确有阶段性
   加速，再保留满节点申请。

accounting 可用时，最终 CPU 效率应从作业结束后的 `sacct` 计算：

```text
CPU efficiency = TotalCPU seconds / (ElapsedRaw * AllocCPUS)
```

accounting 不可用时，用 GNU time 的 `(user seconds + system seconds) / wall seconds` 得到
平均使用核心数，再除以 `AllocCPUS` 得到近似效率；同时用分阶段 wall 解释串行尾部。

这项比值应结合分阶段日志解释。HDF5 写入、route 汇总等有界串行尾部不可能维持 100%
利用率；需要排查的是像旧作业那样连续数十小时仅约 2.1 核且 wall 几乎不下降的状态。
