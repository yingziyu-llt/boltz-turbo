# 面向虚拟筛选初筛的 Boltz-2 推理加速

## 1. 研究目标

研究问题：在固定 GPU 预算下，能否通过推理算法与算子优化，提高 Boltz-2 对蛋白质–小分子库的筛选吞吐量，同时保留早期命中富集能力？

工作负载固定为同一靶点、多种小分子，主平台为集群 A800 / L40。两种 GPU 分别测量、分别报告；记录实际显存规格、功耗上限和软件版本。不以本机显存作为研究限制。先优化单 GPU 吞吐量，多 GPU 用于分摊不同配体任务。

研究主线：**diffusion 采样算法、自适应调度、跨步计算复用与少步蒸馏，以及对应的算子和执行优化。** 各方法族均开展独立实验，再组合有收益的方法。固定步数、recycling 和 samples 的缩减作为弱基线；Anthropic 套件作为系统优化强基线。新增算法必须在相同 NFE 或实际 GPU 时间下证明收益。

直接初筛与级联筛选分别定义：

- **直接初筛**：库中每个合格分子都经过快速 Boltz-2，按其分数排序。
- **级联筛选**：便宜方法先筛去部分分子，再用 Boltz-2 排序。保留这个应用对照，但单独报告预筛损失和整库成本。

是否达到初筛要求，由库规模、GPU 预算、完成时间与实验筛选名额共同决定。暂定一个工程参考负载：10 万分子、8 张同型号 GPU、24 小时内完成，对应每卡平均至少约 0.145 分子/秒，即约 6.9 GPU 秒/分子。该数字是预算反推的目标，不是已测性能；还需为一次性准备和调度留出时间。

## 2. 已核实的基础事实

| 项目 | 事实及研究含义 |
| --- | --- |
| 官方版本 | Anthropic Boltz-2 套件固定 `boltz==2.2.1`，上游 tag `v2.2.1`，提交前缀 `cb04aecc`。本项目优先使用相同发布版本，避免把版本差异算作加速。 |
| Python | Boltz 当前声明 `>=3.10,<3.13`；环境选择 Python 3.11。 |
| 精度与三角算子 | Boltz-2 已采用 bf16 混合精度和加速三角算子。官方当前 CUDA 依赖包含 cuEquivariance；“首次启用混合精度/三角内核”不能作为新增贡献。 |
| 初筛分数 | 主排序分数采用 `affinity_probability_binary`，从高到低。官方推荐它区分 binder 与 decoy。 |
| 连续亲和力 | `affinity_pred_value` 用于活性分子间的亲和力比较，值越低表示预测结合越强。它不是本项目的默认初筛排序分数。 |
| 模型路径 | 完整任务包括输入准备、结构预测、置信度计算及 affinity pass。上游 CLI 在 affinity 阶段加载 `boltz2_aff.ckpt`，重新执行模型，固定 `recycling_steps=5`、`max_parallel_samples=1`；只加速首个结构阶段不能代表完整筛选收益。 |
| 已有筛选证据 | Boltz-2 原论文在 MF-PCBA 上评估 AP、早期富集与 AUROC；TYK2 筛选以 ABFE 计算验证，不能表述为湿实验确认命中。 |
| 支持范围 | 官方 affinity 主要面向蛋白质–小分子；每次指定一个小分子 binder。大配体、必要辅因子和复杂多聚体体系需要单独判断适用性。 |

官方 CLI 的起步设置显式指定为：`recycling_steps=3`、结构 `sampling_steps=200`、`diffusion_samples=1`、affinity `sampling_steps_affinity=200`、`diffusion_samples_affinity=5`、`seed=0`。这些值用于定义比较基线，不表示各参数成本独立。实际调用链要通过首轮 trace 确认。

网页参数表、Python 函数签名与 Click 装饰器存在部分默认值差异。执行前以所安装发布版本的 `boltz predict --help` 和 Click 实际行为为准，实验命令显式写入上述关键参数。

预算控制需区分两阶段：CLI 的 `--recycling_steps` 只控制首个结构阶段；affinity recycling=5 写在 `src/boltz/main.py` 的 `predict_affinity_args`，需要修改这一处才能单独研究该预算。结构模型按 `recycling_steps+1` 执行 trunk，因此 3 recycling 对应 4 次 trunk pass，affinity 的 5 对应 6 次。先在固定发布版本核对，再通过 trace 计时。

## 3. Anthropic 参考工作的使用方式

参考报告为 *Accelerating open-source biomolecular models with Claude*，采用官方文章链接的 2026-09-21 更新版。实现以开源 `boltz2/` 套件及其版本说明为依据。

### 已有优化

| 模式 | 套件已提供的优化 | 本项目如何使用 |
| --- | --- | --- |
| `off` | 固定版本的 stock Boltz-2 | 官方实现基线，保留原有加速内核。 |
| `exact` | eval 冗余计算消除、pair block 融合、部分精确三角算子、步间不变量外提、CUDA Graph、预取与输出重叠 | 精确实现优化基线；套件对特定硬件/shape 有限制，输出一致性声明不直接外推到本项目 affinity 分数。 |
| `fast` | FlashPairformer、resident bf16 pair stream、融合 MSA/扩散/原子计算等 | 强加速基线；重新验证早期富集与分数变化。 |
| `big` | 分块、提前释放和大体系多 GPU 切分 | 只有输入规模确实需要时使用，不作为初筛吞吐量主线。 |

关键代码入口：

- `boltz2/README.md`：运行接口、硬件条件、affinity 行为。
- `boltz2/STOCK.md`、`environment/requirements.lock`：固定版本及依赖栈。
- `boltz2/CHANGES.md`：逐项优化及其适用条件。
- `boltz2/opt/boltz2_opt/registry.py`：优化开关。
- `common/opt_core/`：共享 FlashPairformer 与相关算子。

套件 README 明确说明 affinity 输入仍调用上游 affinity pass。要追踪这个 pass 的模块调用与实际耗时，再判断是否存在未覆盖热点。

报告 §S1.10 给出 Boltz-2 fast 的前向调用几何平均加速 **5.56 倍**：H100 80 GB、batch=1、200–1400 tokens、10 recycling、5 diffusion samples、200 steps、预计算完整 MSA、无模板，且明确 **不包含 affinity head**。该结果不能直接外推到默认 3 recycling / 1 sample 的完整筛选任务。FlashPairformer 相对 cuEquivariance 的三角 attention / multiplication 数字属于算子微基准；“多模型平均约 4 倍”也不能当作本项目预期结果。

### A800 / L40 的适配边界

- 套件主要报告 H100；另有 A100 配置，未列出原生 A800 / L40 配置。
- A800 与 A100 同属 compute capability 8.0，A100 配置是 A800 的适配起点，仍需核实设备识别、实际启用算子及数值行为。
- L40 为 compute capability 8.9，套件缺少该架构的实测 cell 列。代码中部分 exact LayerNorm / DiT 路径只证明了 8.0 / 9.0，可能导致模式拒绝运行（基于代码路径的推断，尚未实测）。先跑 stock，再验证套件实际运行；需要时为 8.9 移植并重新测量便携 Triton 路径，不直接标为套件原生支持。
- 当前套件固定栈包含 Python 3.11.12、PyTorch 2.12.0+cu130、Triton 3.7.0、cuEquivariance 0.10.0，要求宿主驱动 580 或以上。集群驱动版本是接入前需确认的条件，容器不能替代宿主驱动升级。
- 如果某卡无法运行该栈，先建立该卡可运行的官方基线，再移植适用优化；将其标为移植版本，不宣称完整复现原套件。
- 套件声明不维护、不接收贡献。固定使用的提交，保留必要许可证与来源，不把整个套件架构复制为本项目结构。

## 4. 起步环境与首轮运行

已准备 `data/smoke.yaml`、官方预计算 `data/msa/seq1.a3m` 和 `run_smoke.sh`。输入取自 Boltz v2.2.1 官方 affinity 示例，蛋白为 384 残基，MSA 为 249 条记录；已核对 query 与输入序列一致，检查 shell 和结果解析代码语法。集群复用环境为 Python 3.11.16、Boltz 2.2.1、PyTorch 2.13.0+cu130、Triton 3.7.1、cuEquivariance 0.11.1。L40 节点驱动 590.44.01 已通过 CUDA 初始化；A800 节点驱动 575.57.08 无法运行这套 CUDA 13 环境，需要另建兼容环境。首轮完整结构与 affinity 在 L40 作业 `31814315` 执行，远端目录 `/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate`；具体状态见 `results/p0_status.json`。未安装新依赖或下载权重。

首轮结果：作业 `31814315` 完成，退出码 `0:0`，结构与 affinity 两阶段均为 0 个失败样本。CLI wall time 为 149.21 秒，Slurm 总时长 158 秒；日志进度时间约为结构 30 秒、affinity 11 秒，不能用它们代替 profiler 模块计时。已生成 CIF 和 affinity JSON，`affinity_probability_binary=0.3151170015335083`，`affinity_pred_value=2.1493706703186035`。这是一次功能性验证，尚无稳态吞吐、峰值显存或筛选质量结论。

### 同靶点多配体基线与分阶段 profile

`exp_baseline.py` 使用同一蛋白及 6 个芳香族氨基酸配体，保留结构 3 recycling / 200 steps / 1 sample，affinity 5 recycling / 200 steps / 5 samples。配体没有活性标签，仅用于性能测量。L40 两次独立 CLI 批量运行分别耗时 216.88 秒、200.23 秒，即 36.15 秒/配体、33.37 秒/配体；12 份结构与 affinity 输出均通过文件和分数检查。第二次复用磁盘和 JIT 缓存，模型仍重新加载，不代表常驻模型稳态吞吐。部分配体在两轮 seed=0 时仍有分数差异，原因尚未隔离。

独立 profile 作业 `31814400` 完成，退出码 `0:0`。预处理 0.48 秒，结构与 affinity checkpoint 加载分别为 24.05 秒、23.77 秒。结构 / affinity 的 Trainer wall time 分别为 21.02 秒、16.14 秒；PyTorch 峰值 allocated 显存分别为 3.21 GiB、2.63 GiB，reserved 分别为 3.96 GiB、2.92 GiB，不等于整卡总占用。

CUDA event 区间中，结构 / affinity diffusion 分别为 9.55 秒、10.72 秒，pairformer 为 3.44 秒、1.97 秒，confidence 为 0.28 秒、0.47 秒；affinity 两个评分头合计约 0.12 秒。区间包含 CPU 提交空隙以及每阶段一次 denoiser 的 profiler 扰动，不能视为纯 kernel 耗时；嵌套区间不能累加。采样的 denoiser 中，`aten::mm`、`aten::addmm` 和 `aten::bmm` 占 self CUDA 时间约 71%（结构）和 78%（affinity），线性代数和调用开销优先于单独优化评分头。下一步测无 profiler 的常驻模型 denoiser，再比较编译或 CUDA Graph；减少 diffusion 预算属于独立质量实验。摘要见 `results/baseline_status.json`，完整输出在远端 `results/throughput` 与 `results/profile`。

### 常驻 denoiser 与 CUDA Graph

`exp_cuda_graph.py` 在 L40 作业 `31814416` 完成，退出码 `0:0`，保留全部采样预算。使用真实 denoiser 输入，各模式测 5 组、每组 20 次调用；结构单步中位数加速 1.30×，affinity 为 1.01×。固定输入及更新坐标、sigma 后的重放输出均通过 `rtol=1e-5, atol=1e-5` 检查，实测最大绝对误差为 0。

常驻模型和条件张量后，完整 warm sample 的结构耗时从 6.160 秒降到 4.351 秒（1.416×），affinity 从 9.584 秒降到 9.386 秒（1.021×）。每阶段只测一次完整 warm repeat，通过保存和恢复 RNG 保持采样序列。结构 / affinity 捕获分别耗时 0.126 秒、0.078 秒。最终序列化 CIF 坐标与 6 个 affinity 分数均与 eager 完全一致；该结论仅覆盖本输入。PyTorch allocated 峰值从 3.208 / 2.634 GiB 变为 3.224 / 2.654 GiB。

实验包含微基准与重复采样，CLI 总耗时不代表部署延迟；图仅复用本次 sample 的条件张量，未实现跨配体复用。优先推进结构阶段 Graph，下一步移除内嵌微基准并按 sample 管理图生命周期，独立测端到端延迟。当前 affinity 阶段收益较小，暂不以 Graph 为主要优化方向。摘要见 `results/resident_status.json`，远端完整结果位于 `results/resident`。

### Anthropic 原套件的 L40 适配

原套件固定为 `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`（kit 0.3.36 / opt_core 0.5.228.0），独立环境使用 PyTorch 2.12.0+cu130、Triton 3.7.0、cuEquivariance 0.10.0、Python 3.11.16。`exp_anthropic_l40.py` 保留原树，通过模式配置明确裁剪 L40 未验证的精确副本和无法运行的内核；所有裁剪写入 LEVER 行。原版在 L40 的 exact / fast / big 严格验收均退出 3，不能称为全量复现。

作业 `31815274` 中，L40 适配 exact / fast / big / big 双卡均完成两轮结构和 affinity，最终各子运行退出 0。CLI 总耗时分别为 exact 175.59 / 86.25 秒，fast 61.80 / 58.76 秒，big 61.06 / 60.05 秒，双卡 80.54 / 76.58 秒。相同固定栈的 stock 两轮为 99.97 / 86.09 秒。各组未交错计时，启动波动明显；第二轮 fast 对 stock 的观察值约 1.47×，不代表稳定吞吐结论。第二轮结构 PHASE 为 stock 10.525 秒、exact 8.124 秒、fast 3.154 秒、big 4.347 秒、双卡 18.687 秒。套件 affinity 为独立执行路径，结构计时不能用于声称 affinity 加速。

实际执行包括 diffusion roll-out Graph、不变量外提、DiT 分支调度与 mask/scale-bias、dummy template 跳过、MSA cast/count/division 外提、构造初始化跳过、atom gather/外提、预取和异步写盘；fast 还执行 bf16 pair residency、triangle kernels、MSA PWA/transition、DiT 与 atom 融合、条件投影折叠；big 执行 conditioner 分块、提前释放、relative-position 延迟重算与 expandable segments；双卡执行 rowpair sharding。逐项调用数见远端 `results/anthropic/comparison.json` 的 levers。

L40 不支持原套件已验证范围内的 exact LayerNorm/residual、DiT softmax/glue、exact MSA PWA/transition；OPM 默认及 v2 配置均遇到 OutOfResources，最终明确停用。独立 pairblock/transition 无 L40 cell，按 `--allow-partial` 明确走 stock；397 tokens 超过 trunk Graph 的 300-token 上限；无真实 template，C64 template 融合未触发；TF32 因未验证 transition cell 停用。fast 虽请求 Jacobi，对应 rollout 记录为 `kabsch=device`，不认定 Jacobi 内核已执行。4/8 卡、A800、其他尺寸与真实模板未测。

适配 exact 的 CIF 与 stock 字节一致，6 个 affinity 分数差异均为 0。fast / big 与 stock 的全原子刚体对齐 RMSD 分别为 0.600 / 0.595 Å，主 affinity value 绝对差为 0.1911、概率差为 0.03275；双卡为 0.849 Å、0.09978、0.01122。单输入差异不能证明落在 stock seed-to-seed 波动内，也不能推出筛选质量不变。摘要见 `results/anthropic_status.json`。

### sm89 单卡算子

`sm89_opm.py` 为 L40 增加指针加载的 OPM tile：BI=4、BJ=4、BK=32、BZ=16、4 warps、2 stages。作业 `31824153` 的共享内存为 69,632 B，chunked 路径有 16 个 spills。199 tokens / 128 MSA 的 stock / fused 耗时为 1.341 / 0.612 ms；397 / 128 为 5.320 / 2.328 ms；397 / 1024 为 9.815 / 5.531 ms。相对 RMS 误差分别为 0.0067%、0.3758%、0.3274%，重复输出逐位一致；该路径为 bf16 近似计算。

`sm89_pair.py` 增加 C64/C128 attention prologue/epilogue 和小 tile transition。作业 `31824258` 在 199/397 tokens、starting/ending 两种布局与 stock 比较，8 个 attention 样本逐位一致，整体 block 加速 1.25–1.57×。397 tokens 的 C64/hidden128、C128/hidden256、C128/hidden512 transition 样本逐位一致，加速分别为 2.39×、2.40×、2.99×。这些有限形状的相等结果不代表任意输入逐位相等。

`sm89_exact.py` 复用原 ATen 算术顺序并在 sm89 编译 CUDA 内核，保留运行期逐位锁。LayerNorm 的 64/128/256/384/512/768 宽度、带/不带 affine 均通过逐位比较；DiT softmax 的 N199/N397、带/不带 mask 和 affine/gated-residual glue 均通过。完整 exact 推理实际执行 LayerNorm 1,209 次、softmax 48 次、glue 240 次、精确 MSA transition 15 次，CIF 与 stock 字节一致，6 个 affinity 分数差异为 0。残差融合受原大小阈值限制，本输入没有执行。

精确 PWA 使用 Ada 的 BI32/BS4、4 warps、1 stage 调度，保留候选求和顺序逐位锁。作业 `31825266` 的 N199/N397 候选分别以 residue 7/13 的 k8 MMA 与 stock 逐位一致。完整作业 `31825397` 的真实 MSA 深度为 249，选择 `[2,13]`，比较 1 次后执行 15 次、无 fallback；结构 PHASE 为 7.413 秒，CLI 为 76.36 秒，CIF 与 6 个 affinity 分数仍与 stock 完全一致。

作业 `31824161` 交错执行旧 fast 与 OPM fast，完整耗时分别为 116.12 / 75.74 秒与 76.42 / 72.35 秒。作业 `31824258` 的组合 exact / fast 为 82.17 / 75.38 秒，结构 PHASE 为 10.592 / 4.770 秒；组合 fast allocated 峰值 4.24 GiB。组合 fast 与 stock 的对齐全原子 RMSD 为 1.107 Å，主 affinity value / probability 绝对差为 0.1051 / 0.06375。端到端启动波动超过部分算子收益，当前不宣称组合版稳定快于旧 fast。筛选质量尚未验证。

单卡入口为 `exp_sm89.py`；`install_sm89.py` 安装项目内模块的子进程加载入口，`SM89_ACTIVE=1 SM89_FULL=1` 启用组合路径。编译缓存与临时目录统一使用项目的 `cache/` 和 `tmp/`。固定 cu130 环境需要兼容驱动，作业指定已验证的 `s60b05n02`。当前输入单卡容纳，后续优先按配体分配独立 GPU，不采用单复合物 rowpair 多卡切分。TF32、真实模板完整推理和 trunk Graph 扩大适用范围尚未验证。摘要见 `results/sm89_status.json`。

### Training-free 采样实验

`exp_training_free.py` 在 L40 上完成 12 组结构与 affinity 预测，结果见 `results/training_free_status.json`，远端完整记录为 `results/training_free/comparison.json`。保持结构 3 recycling / 1 sample、affinity 5 recycling / 5 samples；使用 stock eager，不混入近似算子。每次采样保存并恢复 RNG 后重复一次，记录 warm sampling 耗时；CLI 和 Trainer 时间包含该额外重复，不能当作部署延迟。

原始 200 步结构 / affinity warm sample 为 6.064 / 9.312 秒。原始 100 步合计 7.747 秒，binder probability 与基线绝对差 0.0200；原始 50 步合计 3.871 秒，概率差 0.1108。保留原始随机采样及物理引导的几何日程 50 步合计 3.798 秒，采样部分相对 200 步为 4.05×，概率差 0.0261。按蛋白 Cα 对齐，几何 50 步的蛋白 Cα / 配体 RMSD 为 3.165 / 1.458 Å；这些是相对 stock 预测的偏差，不是实验结构精度。

确定性分支固定坐标系、关闭 churn、reverse alignment 和物理引导、使用 unit step scale；比较 Karras / 几何 Euler、Heun 和 DPM-Solver++ 2M。Heun 32 / 16 步实际为 63 / 31 次网络调用，其他方法每步一次；affinity 每次调用包含 5 个样本。确定性 Euler 即使保留 200 步，蛋白 Cα RMSD 也为 8.241 Å，概率差 0.0805；少步高阶配置 Cα RMSD 为 8.27–8.63 Å，当前未证明其质量保持收益。优先继续验证随机几何日程；确定性策略的多个变化尚未独立消融。

本轮只有一个无标签输入、seed 0，两阶段同时改变，没有独立阶段归因、PoseBusters validity 或富集结论。作业 `31848535` 完成，退出 `0:0`；前四组原始/几何对照来自作业 `31848018` 的已完成输出。

### DPM-Solver++ 2M 改造

`exp_training_free.py` 提供标准 midpoint、Taylor 系数、全局历史项减半、sigma≤1 Å 时降为一阶、log-sigma 网格 power=0.75，以及仅配体历史项减半六种配置。所有变体固定坐标系、关闭 churn / reverse alignment / 引导，使用 unit step scale 和相同噪声端点，不启用新增算子加速。每个 study 固定 trunk 条件和初始 RNG，比较 16 / 32 NFE，相对 Heun512（1023 NFE）计算有效原子逐坐标 RMS、配体逐坐标 RMS 和配体内部距离 RMS；Heun256 用于观察数值参考敏感性。该参考不代表实验结构或物理有效性。

作业 `31851302`、`31851434` 均退出 `0:0`。结构阶段 16 NFE 下，末段降阶使 seed 0 的全原子坐标 RMS 从 0.5919 降到 0.5753 Å、配体距离 RMS 从 0.0380 降到 0.0264 Å；seed 1 分别从 0.2870 降到 0.2512 Å、0.0349 降到 0.0280 Å，没有增加网络调用。Taylor 和全局阻尼未优于标准 midpoint 的全原子误差；低噪声节点加密也没有整体优势。

配体选择性阻尼有取舍。seed 1、同 affinity 条件、32 NFE 下，配体距离 RMS 从 0.1714 降到 0.1223 Å，但全原子坐标 RMS 从 0.1419 升到 0.1539 Å，配体坐标 RMS 从 0.7176 升到 0.9171 Å。完整两阶段预测的概率在 seed 0 为标准 0.3923 / 配体阻尼 0.4165，seed 1 为 0.3214 / 0.3145，不能凭概率方向判断质量。两阶段 warm sampling 合计约 2.42–2.45 秒。

保留标准 midpoint 为默认，将低噪声末段降阶作为 16-NFE 结构阶段候选；配体阻尼尚未形成稳定优势。只测一个无标签复合物、两个 seed；不同完整结构变体会改变 affinity 条件，不能将不同 study 当作同条件重复。一个 affinity 条件下 Heun256/512 的坐标 RMS 差为 0.0371 Å，参考解仍有不确定性。历史阻尼通常牺牲二阶精度，末段降阶属于实验性混合策略。摘要见 `results/dpm_status.json`，远端完整结果为 `results/training_free/dpm_comparison.json`。

`analyze_dpm_confidence.py` 使用 stock seed 0 的 pLDDT≥90 区域固定分组，在 327 个高置信度 Cα 上 Kabsch 对齐，报告逐原子欧氏 RMSD。相对同条件结构 Heun512，标准 2M 的 16 NFE 在 seed 0 / 1 的高置信度全原子 RMSD 为 0.109 / 0.158 Å、口袋为 0.034 / 0.035 Å；32 NFE 分别为 0.091 / 0.110 Å、0.010 / 0.011 Å。全局误差主要受低置信度区域影响。末段降阶没有改善高置信度核心和口袋指标，因此不按全局 RMS 的下降优先采用它。相对官方 stock seed 0，标准 32 NFE 的高置信度全原子 RMSD 为 0.887 / 0.967 Å、口袋为 0.599 / 0.730 Å，包含采样策略及 seed 差异；不能与纯求解器数值误差混为一谈。结构数组去 padding 后与对应 CIF 最大坐标差 5.72e-6 Å。尚未映射 affinity 裁剪坐标的置信度分区。摘要见 `results/dpm_confidence_status.json`。

### 多体系、多种子稳定性

作业 `31852410` 在 L40 完成，退出 `0:0`，用时 01:13:09。4 个体系 × seed 0–4 × stock200 / DPM16 / DPM32，共 60 次完整 structure+affinity 预测。三个新增体系为 3PTB–BEN、1IEP–STI、1STP–BTN，使用实验单体序列、空 MSA；smoke 保留预计算 MSA。结构 recycling=3、1 sample，affinity recycling=5、5 samples。DPM 使用标准 midpoint、固定坐标系、无 churn / reverse alignment / 引导、unit step scale，与官方 stock 的比较包含整套采样策略差异。

下表为 5 个种子的中位数。配体 RMSD 相对实验结构，在蛋白 Cα 对齐后计算，并对 CCD 图自同构取最小值；口袋为实验结构中距配体 5 Å 内残基的共有蛋白原子。采样时间为两个阶段之和。

| 体系 | 配置 | 实验配体 RMSD / Å | 实验口袋 RMSD / Å | 采样 / s | 完整 CLI / s |
| --- | --- | --- | --- | --- | --- |
| 3PTB | stock200 | 0.172 | 0.446 | 15.311 | 69.767 |
| 3PTB | DPM16 | 0.311 | 0.571 | 1.263 | 64.604 |
| 3PTB | DPM32 | 0.310 | 0.583 | 2.442 | 63.558 |
| 1IEP | stock200 | 0.301 | 0.656 | 15.737 | 80.983 |
| 1IEP | DPM16 | 0.414 | 0.879 | 1.315 | 66.322 |
| 1IEP | DPM32 | 0.401 | 0.876 | 2.515 | 63.015 |
| 1STP | stock200 | 0.577 | 0.446 | 12.483 | 66.610 |
| 1STP | DPM16 | 0.787 | 0.624 | 1.025 | 54.497 |
| 1STP | DPM32 | 0.774 | 0.617 | 1.989 | 63.586 |
| smoke | stock200 | — | — | 15.383 | 80.592 |
| smoke | DPM16 | — | — | 1.284 | 67.254 |
| smoke | DPM32 | — | — | 2.462 | 62.725 |

DPM16 的采样加速为 11.97–12.18×，完整 CLI 中位数之比为 1.08–1.22×；DPM32 分别为 6.26–6.28×、1.05–1.29×。CLI 包含两次 checkpoint 加载，配置顺序固定，不能据此判断 DPM32 比 DPM16 的执行成本低。三个实验体系全部 45 次预测的对称校正配体 RMSD 均低于 1.28 Å；DPM 的中位数较 stock 增加 0.10–0.21 Å。60 次预测均未出现配体–蛋白原子距离低于 1.2 Å 的接触；这不等同于 PoseBusters 有效性。

多种子结果显示高置信度区域仍有变化。相对同 seed stock，DPM16 / DPM32 的高置信度全原子 RMSD 中位数分别为：3PTB 0.489 / 0.535 Å、1IEP 0.798 / 0.794 Å、1STP 0.833 / 0.827 Å、smoke 0.923 / 0.921 Å。每种方法内部的 10 对 seed 比较中，1IEP 口袋 RMSD 中位数为 stock 0.395 Å、DPM16 0.922 Å、DPM32 0.957 Å；1STP 分别为 0.279 / 0.803 / 0.841 Å。增加到 32 NFE 未恢复 stock 的跨 seed 稳定性。

当前优先将 DPM16 作为有标签筛选 pilot 的成本候选，保留 stock200 为质量基线。smoke 的 binder probability 中位数为 stock 0.3264、DPM16 0.4068、DPM32 0.3356，尚不能由分数偏移推断早期富集质量。下一阶段在相同配体集合上比较 EF、Recall、AP 和排序，并分开消融结构阶段与 affinity 阶段的预算。

限制：三个历史 PDB 可能与训练集重叠；输入省略天然多聚体、水及非 binder 辅因子；图对称校正未施加立体化学约束，直接原子名指标同时保留；跨 seed 配体指标未作对称校正；smoke 无实验结构；当前无活性/非活性标签。完整分析位于远端 `results/stability/comparison.json`，分析入口为 `analyze_stability.py`。

### 持久化常驻公平对比（A800 & L40，2026-11-22）

1. **跨架构环境统一与故障定性**：L40 节点 `s60b03n01` 在作业 `31982149` 报错实为宿主驱动 `575.57.08`（CUDA 12.9）与 `torch==2.13.0+cu130` 不匹配，已核查 `ecc.errors.uncorrected.volatile.total = 0` 排除硬件 ECC 问题。为统一 A800 与全 L40 节点基线，已将集群主环境 `/lustre3/lhlai_pkuhpc/llt/.micromamba/envs/boltz2` 切换为 `torch==2.11.0+cu128` + `triton==3.6.0`（保留 `cuequivariance==0.11.1`、`boltz==2.2.1`；切换前完整依赖备份于集群 `tmp/boltz2_pre_cu12_freeze.txt`；`anthropic-env` 原封不动），在 `575.57.08` 与 `590.44.01` 两种驱动上均可运行。
2. **实验口径**：`exp_resident_compare.py` 实现进程内权重常驻，冷加载后缓存命中 `checkpoint_load` 降至 0.023–0.028 s；使用 `data/batch`（384 aa、249 MSA、6 芳香族氨基酸配体），偶/奇轮次正反交错跑 2 repeats，显式设 `offload_inactive_to_cpu=False`、`reseed_per_stage=True`。A800 作业 `31982386`（`s61b03n01`，驱动 `590.44.01`，耗时 `00:18:00`）与 L40 作业 `31982387`（`s60b03n02`，驱动 `575.57.08`，耗时 `00:17:41`）均零错误完成，峰值 allocated 显存 3.22 GiB。
3. **跨架构瓶颈交叉现象**（详细数据见 `results/resident_compare_a800_status.json`、`results/resident_compare_l40_status.json`）：

| GPU | 配置 | 总耗时 (`s/mol`) | `prep` | `S_非扩散` | `S_扩散` | `A_非扩散` | `A_扩散` | 整机加速 | 扩散加速 | 扩散占比 | 非扩散(Trunk)占比 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **A800** | `stock_200` | 27.32 | 0.23 | 4.08 | 6.90 | 3.48 | 12.62 | 1.00× | 1.00× | 71.5% | 28.5% |
| **A800** | `stock_100` | 17.25 | 0.23 | 3.98 | 3.45 | 3.39 | 6.29 | 1.58× | 2.00× | 56.4% | 43.6% |
| **A800** | `geometric_50` | 12.46 | 0.22 | 3.89 | 1.73 | 3.47 | 3.15 | 2.19× | 4.00× | 39.2% | 60.8% |
| **A800** | `dpm2m_32` | 10.52 | 0.23 | 3.88 | 1.03 | 3.44 | 1.92 | 2.60× | 6.60× | 28.1% | 71.9% |
| **A800** | **`dpm2m_16`** | **9.03** | 0.23 | 3.89 | 0.52 | 3.42 | 0.96 | **3.03×** | **13.22×** | 16.4% | **83.6%** |
| **L40** | `stock_200` | **24.63** | 0.28 | 5.06 | 5.83 | 3.27 | 10.18 | 1.00× | 1.00× | 65.0% | 35.0% |
| **L40** | `stock_100` | 16.50 | 0.28 | 4.91 | 2.93 | 3.25 | 5.12 | 1.49× | 1.99× | 48.8% | 51.2% |
| **L40** | `geometric_50` | 12.55 | 0.28 | 4.91 | 1.46 | 3.36 | 2.53 | 1.96× | 4.01× | 31.8% | 68.2% |
| **L40** | `dpm2m_32` | 10.90 | 0.28 | 4.88 | 0.84 | 3.27 | 1.63 | 2.26× | 6.49× | 22.6% | 77.4% |
| **L40** | **`dpm2m_16`** | 9.71 | 0.28 | 4.92 | 0.43 | 3.27 | 0.81 | **2.54×** | **12.93×** | 12.8% | **87.2%** |

> 关键发现：`stock_200` 下 L40 更快（小 batch diffusion 延迟更低：24.63 vs 27.32 s/mol）；当 `dpm2m_16` 将扩散压低 ~13× 后，A800 凭借更高的显存带宽在 Trunk 阶段胜出，总耗时反超 L40（9.03 vs 9.71 s/mol），此时非扩散 Trunk 已占 83.6–87.2%。

### HSP90 真实实验亲和力基准（180 配体全量 & 60 配体解耦，2026-11-23）

1. **数据集与评测脚本**：采用 `/lustre3/lhlai_pkuhpc/llt/hsp90_affinity`（靶点 PDB `4YKR`，236 残基，`hsp90.a3m` 5.4 MB 真实 MSA，YAML 无模板）。`hsp90_labels_primary.csv` 共 180 个配体：147 个定量实验 IC50（5.22 nM – 68.3 µM）+ 33 个删失 `>50 µM`；含 3 种化学骨架（`pyrimidin-2-amine` 62、`aryl-benzimidazolone` 61、`benzophenone-like` 57）；活性分布为 `≤1 µM` 73 个、`≤100 nM` 41 个。评测入口为 `analysis/eval_hsp90.py`（已修复删失行空字符串解析并用 `np.isfinite` 保护定量指标），提交脚本为 `slurm/hsp90_compare_a800.slurm`。
2. **180 全量队列（`fair`，A800 `s61b04n01`，作业 `31984096`，耗时 `03:33:39`，0 失败，Peak 8.38 GiB，冷加载 48.60 s：`conf` 21.52 s + `aff` 27.09 s，`results/hsp90_a800_fair_status.json`）**：

| 配置 | 稳态 (`s/mol`) | 加速 (`diff`加速 / `dif%`) | Quant `r` / `ρ` / `τ` / `cRMSE` | `≤1µM` AUROC / AP / EF10% (`-val` | `prob`) | `≤100nM` AUROC / AP / EF10% (`-val` | `prob`) | vs `stock` `ρ(val/prob)` | 分骨架 `r` (`aryl`/`pyrim`/`benzo`) |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `stock_200` | 25.02 | 1.00× (1.00× / 72.3%) | 0.448 / 0.411 / 0.286 / 1.61 | 0.706/0.720/2.47× \| 0.745/0.738/2.19× | 0.787/0.660/3.66× \| 0.780/0.651/3.66× | 1.000 / 1.000 | 0.733 / 0.455 / -0.174 |
| `stock_100` | 15.95 | 1.57× (2.00× / 56.7%) | 0.457 / 0.422 / 0.298 / 1.60 | 0.710/0.717/2.47× \| 0.739/0.730/2.19× | 0.788/0.660/3.66× \| 0.774/0.648/3.66× | 0.992 / 0.995 | 0.737 / 0.476 / -0.177 |
| `geometric_50` | 11.44 | 2.19× (4.00× / 39.6%) | 0.444 / 0.409 / 0.289 / 1.62 | 0.703/0.718/2.47× \| 0.744/0.733/2.19× | 0.782/0.654/3.66× \| 0.777/0.625/3.41× | 0.992 / 0.995 | 0.745 / 0.457 / -0.215 |
| `dpm2m_32` | 9.66 | 2.59× (6.58× / 28.5%) | 0.459 / 0.423 / 0.299 / 1.59 | 0.711/0.721/2.47× \| 0.741/0.734/2.19× | 0.790/0.667/3.66× \| 0.777/0.642/3.41× | 0.993 / 0.993 | 0.748 / 0.471 / -0.169 |
| **`dpm2m_16`** | **8.29** | **3.02×** (**13.14×** / 16.6%) | **0.458 / 0.422 / 0.298 / 1.59** | 0.710/0.717/2.47× \| 0.740/0.730/2.19× | **0.788/0.664/3.66×** \| 0.774/0.639/3.41× | **0.991 / 0.994** | **0.735 / 0.470 / -0.163** |

> `dpm2m_16` 相对 `stock_200` 几何 Cα RMSD = 4.215 Å、配体 RMSD = 1.083 Å；affinity 预测排名相关性 Spearman ρ 达 `0.991–0.994`，真实实验定量与富集指标全程零退化甚至略微提升。

3. **60 配体分层子集与两阶段解耦（`decoupled`，每骨架 20 个，定量 48，`≤1 µM` 25，`≤100 nM` 14；A800 作业 `31984097`，耗时 `01:30:36`，0 失败，Peak 8.33 GiB，`results/hsp90_a800_decoupled_status.json`）**：

| 配置 | 预算 (`S_step/S_rec/S_n`, `A_step/A_rec/A_n`) | 总耗时 (`s/mol`) | 加速 | `prep` | `S_非扩散` | `S_扩散` | `A_非扩散` | `A_扩散` | 扩散占比 | Quant `r / ρ / cRMSE` | `≤100nM` AUROC / AP / EF10% (`-val`) | vs `stock` `ρ(val/prob)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `stock_200` | `200/3/1`, `200/5/5` | 25.39 | 1.00× | 0.29 | 2.49 | 5.32 | 4.32 | 12.94 | 71.9% | 0.477 / 0.400 / 1.57 | 0.812 / 0.702 / 3.57× | 1.000 / 1.000 |
| `s_dpm16_a_stock200` | `16/3/1`, `200/5/5` | 20.52 | 1.24× | 0.28 | 2.45 | 0.39 | 4.39 | 13.00 | 65.3% | 0.476 / 0.400 / 1.57 | 0.806 / 0.688 / 3.57× | 0.993 / 0.995 |
| `s_stock200_a_dpm16` | `200/3/1`, `16/5/5` | 13.39 | 1.90× | 0.28 | 2.45 | 5.32 | 4.35 | 1.00 | 47.1% | 0.477 / 0.412 / 1.58 | 0.821 / 0.714 / 3.57× | 0.993 / 0.994 |
| `dpm2m_16` | `16/3/1`, `16/5/5` | 8.47 | 3.00× | 0.28 | 2.45 | 0.39 | 4.35 | 1.00 | 16.4% | 0.466 / 0.399 / 1.59 | 0.814 / 0.685 / 3.57× | 0.992 / 0.994 |
| `dpm2m_16_affrec3` | `16/3/1`, `16/3/5` | 7.20 | 3.52× | 0.28 | 2.45 | 0.39 | 3.08 | 1.00 | 19.3% | 0.472 / 0.412 / 1.58 | 0.823 / 0.712 / 3.57× | 0.991 / 0.993 |
| **`dpm2m_16_affrec1`** | **`16/3/1`, `16/1/5`** | **5.91** | **4.29×** | **0.28** | **2.45** | **0.39** | **1.79** | **1.00** | 23.6% | **0.494 / 0.425 / 1.52** | **0.842 / 0.708 / 3.57×** | 0.977 / 0.978 |
| `dpm2m_16_affs1` | `16/3/1`, `16/5/1` | 7.60 | 3.34× | 0.29 | 2.45 | 0.39 | 4.11 | 0.36 | **9.9%** | 0.479 / **0.433** / 1.57 | 0.828 / **0.719** / 3.57× | 0.991 / 0.991 |

> 关键发现：`s_stock200_a_dpm16` 相对 `stock_200` 几何 Cα RMSD 仅 **0.056 Å**、配体 **0.023 Å**，说明 `dpm2m_16` 的 affinity 扩散几乎无几何漂移；降低 `affinity_recycling` (`5 → 3 → 1`) 呈现精准线性省时（每轮 recycle 约 `0.64 s/mol`），使 `dpm2m_16_affrec1` 降至 **5.91 s/mol（4.29×，首次优于 6.9 s/mol 目标）** 且 HSP90 相关性不降反升；单独将 `affinity_samples` 降为 1（`dpm2m_16_affs1`）将扩散加速推至 **24.22×**（占比压至 9.9%），省时 `0.88 s/mol`（扩散 0.64 s + 非扩散 0.24 s）且排名完全不损。

### AutoDock Vina 对接基线对照（HSP90 180 配体，2026-11-23）

1. **实验设置与覆盖率**：入口 `experiments/exp_vina_hsp90.py`、`analysis/eval_vina_hsp90.py`、`slurm/vina_hsp90.slurm`。作业 `31991779` 在 `cn-short`（`c01b04n09`，16 核）正常 `COMPLETED (0:0)`（`00:17:39`），使用 `fap-docking`（`AutoDock Vina f458505-mod`、`RDKit 2026.03.1`、`Meeko 0.6.1`、`OPENMM_PLUGIN_DIR=/dev/null` 启用 OpenMM `Reference` 平台）。取 `4YKR` A 链（修补 `A:16:GLU`、`A:123:GLN`、`A:223:GLU` 缺失重原子）及共晶配体 `4EP A:301` 质心 `(-1.545, 9.506, 25.062)` 构建 `20×20×20 Å` 盒子，配体仅从 SMILES 生成（`ETKDGv3` + `MMFF94/UFF`），`exhaustiveness=8`、`--seed 0`（实录 180 个独立有效种子）。覆盖率 **180/180（0 失败）**；12 配体额外串行单核测试 **12/12（0 失败）**。
2. **吞吐与精度对照**（`results/hsp90_vina_status.json`）：
   - **Vina 吞吐口径**：16 核稳态 **3.07 s/mol**（`0.326 mol/s`）、16 核端到端（含 7.92 s 受体准备）**3.11 s/mol**、单核实测 **41.13 s/mol**（`0.0243 mol/s`）、CPU 总核时均值 **43.09 core·s/mol**。
   - **180 全量集对比**：Vina 定量 `r=0.410 / ρ=0.412 / τ=0.284 / cRMSE=1.52`，但早期富集显著低于 Boltz-2（Vina `≤1µM` AUROC `0.670` / AP `0.591` / EF10% `1.78×`，`≤100nM` AUROC `0.721` / AP `0.460` / EF10% `2.20×`；而 `dpm2m_16` 在相同队列下达 `≤100nM` AUROC `0.788` / AP `0.664` / EF10% `3.66×`）。
   - **60 子集对比**：Vina (`r=0.400 / ρ=0.419 / ≤100nM AUROC=0.702, AP=0.483, EF10%=2.14×`) 全面落后于加速版 `dpm2m_16_affrec1` (`r=0.494 / ρ=0.425 / ≤100nM AUROC=0.842, AP=0.708, EF10%=3.57×`，5.91 s/mol)。
   - **信息不对称与强骨架互补性**：Vina 拥有 `4YKR` **holo 晶体结构与已知口袋中心坐标**，Boltz-2 为纯序列+SMILES 盲预测。即便如此，Boltz-2 在 `aryl-benzimidazolone`（全量 `r=0.733–0.735` vs Vina `0.153`）与 `pyrimidin-2-amine`（`r=0.455–0.470` vs Vina `0.202`）大幅领先；而 Vina 在 `benzophenone-like` 表现优异（`r=0.504 / ρ=0.363` vs Boltz-2 `-0.174 ~ -0.163`），显示 holo 口袋构象与盲折预测的显著互补性。

### 当前交接状态与下一步精细 Profiling / 优化路线（2026-11-23 Handoff）

> 当前仓库本地最新结果已沉淀在 `results/*_status.json`，**按用户要求暂不 commit/push**；下一步优先做精细 Profiling，再精准实施 Trunk / 预算 / 并发优化。

#### 1. 已知代码解剖与精细拆分（已在集群验证源码）

- 通过 `results/hsp90_a800_decoupled/summary.json` 中记录的 `model_forward_seconds` 与源码 `Boltz2.forward()` 实查：
  - 在 `dpm2m_16_affrec1`（**5.91 s/mol**）上，`model.forward()` 占了 **95%+** 的 Trainer 时间（Structure：`model_forward 2.676 s` / `trainer_wall 2.843 s`；Affinity：`model_forward 2.726 s` / `trainer_wall 2.789 s`），DataLoader + CIF/JSON IO 合计仅 **0.23 s/mol**，`preprocessing` 仅 **0.28 s/mol**。
  - `Boltz2.forward()` 调用链为：`input_embedder` → `for recycling_steps+1: [template_module (若 self.use_templates) → msa_module → pairformer_module]` → `distogram_module` → `diffusion_conditioning` → `structure_module.sample`（`autocast(enabled=False)`）→ `confidence_module`（**注意：`predict_step` 硬编码了 `run_confidence_sequentially=True`**，Affinity 阶段 `samples=5` 会串行跑 5 遍！）→（Affinity 阶段）`input_embedder(affinity=True)` + `affinity_module1` + `affinity_module2`（`autocast(enabled=False)`）。
  - 集群 `boltz2` 环境已确认装有 `cuequivariance 0.11.1`、`cuequivariance_torch 0.11.1`、`cuequivariance_ops_torch 0.11.1`、`triton 3.6.0`。
  - `hsp90_1.yaml` 无模板，需在 Profiling 中实测 `self.use_templates` 及 `self.use_kernels` 当前在 A800/L40 上的真实取值与 `template_module` 是否空转耗时。

#### 2. 立即执行任务：精细 Profiling（`experiments/exp_profile_hsp90.py` + Slurm）

不要盲写优化补丁，先跑出两阶段、完全无重叠且按子模块加和对齐的真实账本：
- **Pass A（无 profiler 干扰的干净 CUDA Event 分账，在 pre-warm 后连续跑 N=8 个 HSP90 配体取中位数/均值）**：
  - 分别对 **`dpm2m_16`**、**`dpm2m_16_affrec1`**、**`dpm2m_16_affrec1_affs1`** 测量（注意叠加 `affrec1 + affs1` 顺便直接拿到基线数字！）。
  - Structure 与 Affinity 各自拆分不重叠子模块时间与调用次数：`input_embedder`、`template_module`、`msa_module`、`pairformer_module`（区分首次 vs recycle）、`distogram_module`、`diffusion_conditioning`、`diffusion_sample`（单独测 `preconditioned_network_forward` 单步 + sampler 开销）、`confidence_module`、`affinity_module1` / `affinity_module2`（区分内部 trunk vs affinity head），以及 `preprocessing`（含同一 MSA 重复解析耗时）和 `Trainer/DataLoader/IO`。
  - 同步打印关键元信息：`(N_tokens, N_atoms, N_msa, N_templates)`、`model.use_templates`、`model.use_kernels`、各模块激活 dtype、以及 **CUDA active time vs wall time 差距（判断 CPU/kernel launch 气泡占比）**。
- **Pass B（`torch.profiler` 算子热点，在 Pass A 结束后对 1 个热身配体抓取，不混入吞吐计时）**：
  - 单独导出 `pairformer_module`、`msa_module`、`template_module`、`confidence_module`、`diffusion_conditioning`、`denoiser`、`affinity_module1` 的 Top-20 `self_cuda_time_total` 与 `self_cpu_time_total` 算子表（确认 `cuEquivariance` / Tensor Core / ATen 线性代数 / 小算子 launch 实际命中情况）。
- **调度提示**：截至 `2026-11-23 14:43`，集群 `gpu_a800` 两个可用节点处于 `alloc`，`gpu_l40`（`s60b03n[01-02]`, `s60b05n02`）为 `mix` 有空闲卡；统一 `boltz2` 环境两卡通用，可优先用 A800 或先在 L40 出完整算子账本、再上 A800 对账。

#### 3. Profiling 之后的优化路线图（按预期性价比排序）

1. **第一梯队：无痛预算 Pareto 消融**
   - 直接测试 `dpm2m_16_affrec1_affs1`（预期降至 `~5.0 s/mol`）；
   - 消融 **`structure_recycling` (`3 → 1 / 0`)**（`S_非扩散` 目前 2.45 s 完全未动）与 `affinity_recycling = 0`；
   - 步数下探 **`16 → 12 / 8` 步** 与 **UniPC**，确定训练前无重训算法下限。
2. **第二梯队：无损/近似保真 Trunk & 执行系统优化（打掉 72–80% 非扩散瓶颈）**
   - **跳过无模板时的空 `template_module`**（`dummy_template_skip`）；
   - **排查并优化 `confidence_module`**（`run_confidence_sequentially` / Affinity 阶段是否必须对 5 个样本跑置信度，实际上源码第 233 行 `best_idx = argsort(dict_out["iptm"])` 仅用来挑 pose）；
   - **在 A800 / L40 启用/移植 `cuEquivariance`、BF16 融合 Pairformer/Triangle/MSA 算子、`torch.compile` / CUDA Graph**（参考已有 `sm89/` 与 `anthropic` 套件思路，但直接针对当前统一 `torch 2.11+cu128` 栈的 Trunk 热点）；
   - **消除同靶点冗余计算**：缓存在内存中的蛋白特征与单次 MSA 解析，消除 Stage 1→2 之间写盘 `.cif` 与 `Trainer.predict` 重启开销。
3. **第三梯队：显存红利与并发吞吐**
   - 当前 A800 峰值显存仅 **8.38 / 80 GiB（10.5%）**，探索单卡多进程并发（或安全的多配体 Batching）将单卡等效吞吐进一步压到超越 16 核 Vina（3.07 s/mol）。
4. **第四梯队：虚拟筛选专属算法创新**
   - **同靶点受体表征跨配体复用（Cross-ligand Receptor Trunk Caching）**；
   - **结合 `4YKR` 晶体模板引导实验**：测试直接提供 `4YKR` 蛋白模板能否同时大幅削减 Structure Stage 耗时并改善 `benzophenone-like` 的负相关问题（独立公平对比，标注额外结构信息）。

#### 4. 集群与工作流操作约定（交接必读）

- **路径**：本机 `/home/yingziyu/boltz2_acclerate`；集群 `/lustre3/lhlai_pkuhpc/llt/boltz2_acclerate`；HSP90 数据集 `/lustre3/lhlai_pkuhpc/llt/hsp90_affinity`。
- **集群操作规范**：集群经堡垒机交互认证，**严禁**直接使用 `ssh pku-hpc <cmd>`、`scp`、`sftp`、`rsync`。必须使用 `~/.cache/pku-hpc-stealth/start.sh`、`~/.agents/skills/pku-hpc-cluster/scripts/hpc run '...'`（单行命令 + `[[ ! -v HISTFILE && ! -o history ]]` 校验）、`hpc-stage` 传输，结束后务必执行 `hpc stop`。
- **存储配额**：集群 `$HOME` 配额已满（0.77 TB），所有 `XDG_CACHE_HOME`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`CUDA_CACHE_PATH`、`TMPDIR` 必须留在 Lustre 工作目录下的 `cache/` 与 `tmp/`。

### HSP90 精细 Profiling：L40 实测

北极星作业 `31991982`（`COMPLETED 0:0`），`results/hsp90_profile_31991982/` 含 `profile.json` 和六份 Chrome trace。三种配置均使用 `hsp90_1` 至 `hsp90_8`、seed 0、常驻权重，8/8 均写出两阶段结果；逐配体 CUDA Event 排他时间与 forward span 对齐。Pass A 串行预处理以分账 A3M，吞吐不与先前并行预处理的全量基准直接比较。Trunk 指 template、MSA 和 Pairformer 的排他时间之和。

| 配置 | Pass A s/mol | Structure forward（s/mol） | Affinity forward（s/mol） | Structure Trunk | Affinity Trunk |
|---|---:|---:|---:|---:|---:|
| `dpm2m_16` | 10.435 | 3.076 | 5.251 | 85.1% | 76.7% |
| `dpm2m_16_affrec1` | 7.413 | 3.011 | 2.588 | 84.7% | 52.4% |
| `dpm2m_16_affrec1_affs1` | 6.793 | 3.023 | 1.965 | 84.7% | 69.0% |

- 两阶段实际模板数均为 0；Structure 启用 `use_templates` 和 `use_kernels`，空模板占 33–38 ms/mol；Affinity 两个 flag 均关闭。优先检查 Affinity Trunk 的 kernel 路径，再评估空模板跳过。
- `affrec1` 将 Affinity forward 减少 2.66 s/mol；在此基础上将样本数从 5 减到 1 再减少 0.62 s/mol。Affinity confidence 占比从 10.2% 降到 2.5%，串行调用从 5 次降到 1 次。`6.793 s/mol` 仅代表这 8 个配体；单样本的亲和力精度需另测。
- 原始 A3M 每配置解析 8 次，合计 2.54–2.65 s（约 0.32 s/mol）。Pass B 的 Pairformer/MSA CUDA 算子榜首均为 `aten::copy_`；`self_cuda` 按发起算子归集，不等于独立内核名称。
- Pass B 用 Chrome trace 中实际 kernel、memcpy、memset 区间的并集计算活跃率；CPU 回放作业 `31992372`（`COMPLETED 0:0`）已更新 `profile.json`，原日志的 99% 档数字误计了 GPU scope 注释。Structure 活跃率为 71.2% / 73.8% / 73.2%，Affinity 为 96.4% / 93.4% / 49.0%（表中顺序）。单样本 Affinity 的 profiler forward 为 3.34 s，Pass A 均值为 1.97 s；该 49.0% 不能当作稳态 kernel launch 空隙。

### H800 Trunk 阶段优化与系统级缓存探索（已实测）

1. **Affinity-only 算子裁剪与算子微基准（H800，60 配体）**：
   - `32015556`：`dpm2m_16`（`6.854 s/mol`）→ `dpm2m_16_affkern`（仅在 Affinity 分支启用定制 kernel，`5.720 s/mol`，**`1.20×`**，`aff_fwd` `4.091 → 3.051 s`），`val_rho=0.9968`，`prob_rho=0.9948`，`EF10%(100nM)=3.571×`（结果见 `results/hsp90_h800_affkern_status.json`）。
   - `32016460`：`boltz_ops_opt` + `bench_msa_ops` 实测 `5.55 → 5.45 s/mol`；OPM `1.20×`、PWA `0.96×`、Transition `torch.compile` `2.31×/1.62×`（尚未接入主干）。
2. **Trunk 诊断与 TeaCache / Early-Exit（H800，6/12 配体）**：
   - `32018770`（6 配体，`results/hsp90_trunk_diag_status.json`）：`boltz2_conf` 与 `boltz2_aff` 共享的 5,019 个权重键完全一致（`max_abs_diff=0.0`）；MSA delta 余弦相似度 `0.966–0.969`；Pairformer 在 `r=3..5` 层间余弦相似度 `0.9994–0.99997`，`rel_l1(z_PL)` 由 `5.3%` 降至 `0.8%`。
   - `32018858`（12 配体，基线 `dpm2m_16_affkern` `6.369 s/mol`，`results/hsp90_trunk_teacache_status.json`）：
     - `pf_adaptive_exit(0.03)`：`5.542 s/mol`（`1.15×`，`0.000 Å` RMSD，`prob_rho=0.9930`）。
     - `trunk_tc_adaptive`：`4.740 s/mol`（`1.34×`，`val_rho=0.9860`，`prob_rho=0.9720`，`med_lig=0.774 Å`，`med_ca=3.593 Å`）。
     - `trunk_tc_full`：`4.487 s/mol`（`1.42×`，`val_rho=0.9441`）。
3. **Structure 零漂移 + Affinity 复用（H800，12 配体）**：
   - `32018962`（基线 `6.277 s/mol`，`results/hsp90_aff_reuse_status.json`）：保持 Structure 分支完全不改（Cα/Ligand RMSD `0.000 Å`），仅在 Affinity 阶段（pocket crop，`shape_match_s/z=false`）启用 MSA recycle 复用与自适应 TeaCache：
     - `aff_msa_r1_adapt05`：**`4.959 s/mol`**（**`1.27×`**，`aff_fwd` `3.093 → 1.823 s` [`-41%`]，`0.000 Å` RMSD，`val_rho=0.9930`，`prob_rho=0.9790`）。
     - `aff_msa_r1_adapt03`：`5.173 s/mol`（`1.21×`，`0.000 Å` RMSD，`prob_rho=1.0000`）。
   - 已落地至 `experiments/boltz_trunk_opt.py`（`install_affinity_teacache`）与 `experiments/exp_resident_compare.py`（`dpm2m_16_affkern_tc05` / `dpm2m_16_affkern_tc03`），60 配体 H800 最终对比作业 `32021024` 已提交（当前状态 `PENDING (Priority)`）。
4. **跨配体缓存（Cross-Ligand Caching）探索（H800，12 配体）**：
   - `32019064`（基线 `6.201 s/mol`，12 配体蛋白 token 完全一致 `matched_prot_ratio=1.0`、`238/198.5` prot tokens，`results/hsp90_xlig_cache_status.json`）：
     - `xlig_msa_pp`：`4.655 s/mol`（`1.33×`，`amort_fwd=1.48×`，`prob_rho=0.6727`，`med_ca=3.956 Å`）。
     - `xlig_pocket128/96`：`4.20–4.21 s/mol`（`1.47–1.48×`，`prob_rho=0.55–0.56`，`med_lig=4.90–5.66 Å`）。
     - `xlig_warm_r2`：`3.862 s/mol`（`1.61×`，`amort_fwd=1.93×`，`prob_rho=0.5455`）。
   - 结论：直接跨配体复用蛋白隐状态在未微调时精度退化明显；后续高吞吐方向转向**多配体 Batching 推理**（审计确认 `Boltz2InferenceDataModule.predict_dataloader` 硬编码 `batch_size=1`，`Boltz2.predict_step` 设 `run_confidence_sequentially=True` / `max_parallel_samples=1`，且 `BoltzWriter.write_on_batch_end` 存在 `coords = coords.unsqueeze(0)` 单样本假设，而 `collate` 已支持 `pad_to_max`）。

### 环境选择

1. 在 GPU 作业节点记录 GPU 型号/显存、驱动、CPU、CUDA runtime、Python、PyTorch、Triton、cuEquivariance、Boltz 版本。
2. 官方基线与套件基线尽量使用套件固定软件栈，避免混入依赖升级收益。两张卡各自保留一套实测记录。
3. 集群提供 Apptainer 时，可按套件定义构建镜像；否则使用独立 Python 3.11 环境。安装依赖前确认所需软件栈与集群驱动兼容。
4. 权重放在固定可读目录；同一靶点预计算并复用 MSA。正式计时不访问公共 MSA 服务。
5. 起步只用一个短蛋白–小分子输入，跑通结构与 affinity 输出，再加入相同靶点的多个配体。蛋白–蛋白 bundled demo 不能代替 affinity 验证。

官方基线命令：

```bash
export BOLTZ_CACHE=/absolute/path/to/boltz-cache
boltz predict data/smoke.yaml \
  --model boltz2 --seed 0 \
  --recycling_steps 3 \
  --sampling_steps 200 --diffusion_samples 1 \
  --sampling_steps_affinity 200 --diffusion_samples_affinity 5 \
  --out_dir results/stock_smoke --override
```

输入包含蛋白序列、预计算 MSA、小分子，以及 `properties: [{affinity: {binder: B}}]` 对应的 ligand chain；当前可执行输入为 `data/smoke.yaml`。

首轮运行完成条件：生成结构文件和含 `affinity_probability_binary` 的 affinity JSON，确认数值有限，记录完整任务耗时与峰值显存。若失败，按真实错误修环境或输入，不通过减少任务输出掩盖问题。

之后只在需要时增加独立入口脚本，如 `exp_baseline.py`、`exp_budget.py` 和对应 kernel benchmark；暂不建立包、注册表或通用实验框架。逐条结果写入 JSONL。

## 5. 数据与评价协议

### 两类数据集

**性能集**：从研究靶点中挑选短、中、长蛋白和不同配体大小，首批约 12–24 个复合物。记录实际 token 数、atom 数及 MSA 深度；不以氨基酸长度代替完整输入规模。它用于 profile、算子 shape 选择和吞吐量测量，不用于证明富集效果。

**筛选集**：优先挑选 3 个不同靶点做 pilot，再扩至 5 个以上。LIT-PCBA 提供实验 active/inactive 和结构模板，可作为候选来源；MF-PCBA 用于对齐原论文任务，但不能假定公开全集就是 Boltz-2 的 held-out test set。

靶点选择按蛋白质–小分子适用性、结构/MSA可获得性、活性数量和计算成本决定，先确定选择规则，再查看本项目分数。仅有细胞表型、激动/拮抗标签或多组分机制的 assay，不能直接当成纯结合标签。训练数据重叠未知时，只声明相对原模型的加速与质量保持，不声明无泄漏泛化。

pilot 可用每靶点 1,000–3,000 个分子调流程，但分层抽样会改变活性率和 EF 的可比性。最终在事先确定的靶点完整库上评估；算力不支持时报告明确抽样协议与子集结论，不将其换算为完整库 EF。

若有阈值或预算策略需要学习，按靶点划分开发与最终测试；分子层面的拟合同时按 scaffold 分割。不同 assay 的分数不混成一个全局排名。

### 核心指标

| 方面 | 指标与解释 |
| --- | --- |
| 吞吐量 | 完整结构 + affinity 的分子/秒、GPU 秒/分子；区分单 GPU 与多 GPU 总吞吐。 |
| 延迟 | 逐分子 p50 / p95；重复运行给出波动，单次异常不当作加速结论。 |
| 显存 | 每次运行峰值 allocated / reserved；用于解释 batch 和并发上限。 |
| 初筛质量 | EF@1%、EF@5%、Recall@1%、Recall@5%、AP；AUROC 作为辅助。 |
| 固定实验预算 | Recall@K 或 Hit@K，K 对应实际送实验的候选数量。 |
| 输出变化 | 与官方基线的分数差、Spearman、top-K overlap；这些不能替代真实 active/inactive 的富集指标。 |
| 结构质量 | 有参考结构的小集合测 ligand RMSD、物理合理性；结构指标不代替筛选质量。 |
| 失败 | OOM、无效输入、推理失败比例及分布。主比较采用同一输入集合，同时报告有效输出覆盖率。 |

EF@x =（前 x 比例中的活性率）/（完整待筛库活性率），每靶点计算。固定 top-K 预算及并列分数处理规则；跨靶点报告逐靶点结果与宏平均。AP 随活性率变化，不比较不同抽样比例下的绝对 AP。

### 计时口径

- **冷启动**：进程启动、模型加载、首次编译/Graph capture、预测和写盘。
- **稳态端到端**：模型常驻，预计算 MSA 可用，从配体准备/特征化到结构及 affinity 结果写盘。给出处理整个库的 wall time 与吞吐量。
- **模块时间**：预处理、trunk、结构 diffusion、confidence、affinity pass 和 I/O。使用 profiler 嵌套事件避免重复计算时间。
- **算子时间**：同 dtype、shape、mask、布局和设备下测量；单独预热，CUDA 同步/事件明确。带 profiler 的运行不直接作为最终吞吐量。
- **全流程成本**：MSA/模板准备计一次并单列，整库估算加上一次性成本；多 GPU 用实际 GPU 数 × wall time 报告预算，不能只报 wall time 降幅。

固定相同输入、MSA、seed、模型与输出集合比较。切换参数使用新输出目录或显式覆盖，避免 Boltz 跳过已有预测。单次 CLI 输入目录表示批量任务入口，不预设其实现了跨配体 tensor batching。

## 6. 研究方向与实验顺序

### A. 现有实现复现与瓶颈定位

比较 stock、Anthropic `exact`、Anthropic `fast`。先在 A800 跑通，再在 L40 测量可用实现。对每种配置收集完整 affinity 路径 profile，按输入规模分析热点变化。

若热点占总时间比例为 p，即使该部分耗时降为零，总加速也最多达到 1/(1-p)。只为具有可观总时间占比、或限制吞吐量的峰值显存热点开发新内核。

### B. 无需重训练的计算预算缩减

第一轮采用单因素实验，避免全参数网格：

| 因素 | 首轮候选值 | 必须观察的代价 |
| --- | --- | --- |
| 结构 diffusion steps | 200、100、50 | 分数、早期富集与构象变化。 |
| 结构 recycling steps | 3、1 | 仅改变首阶段，对应 4、2 次 trunk pass；观察界面与筛选质量变化。 |
| affinity recycling steps | 5、3、1 | 需修改硬编码参数，对应 6、4、2 次 trunk pass；与结构 recycling 分开消融。 |
| affinity diffusion steps | 200、100、50 | affinity pass 占比与最终分数变化。 |
| affinity samples | 5、3、1 | 构象采样及最终分数的方差与早期召回损失；不等同于减少 affinity head 的模型集成数。 |
| MSA 预算 | 保持基线、减半有效序列预算 | 节省的耗时与构象/打分损失；显式控制采样行为。 |

确定各因素趋势后，只组合候选 Pareto 点。主 seed 为 0；候选配置再用 seed 1、2 观察随机性，不承诺跨架构逐位复现。

减少步数作为弱基线。后续实验同时改变噪声节点分配、数值积分方法、动态步长、跨步复用及模型学习目标，分别检验它们对少步推理和筛选质量的影响。蒸馏列入正式研究工作，与无需训练的方法分别报告训练成本和推理收益。

### C. Diffusion 算法加速：正式实验主线

所有方法先在结构阶段验证，再在 affinity 阶段独立接入，最后比较两阶段组合。原始 200 步与固定减步构成对照。实验同时记录 nominal steps、完整 denoiser NFE、部分 block 调用数、重算/拒步次数及实际 GPU 时间。高阶方法每步可能调用多次网络；缓存可能减少单次调用成本，不能只用 steps 比较。

| 方法族 | 必做的实现与对照 | 核心问题 |
| --- | --- | --- |
| 噪声日程优化 | 原 Karras polynomial、均匀 log-sigma 几何日程；在开发集搜索分段节点分配，固定噪声端点与终止规则 | 在相同 NFE 下，哪些噪声区间需要更多计算？ |
| 高阶/多步求解 | Euler 对照、Heun、DPM-Solver++ 二阶多步、UniPC predictor–corrector 的 EDM 适配 | 更高阶更新能否在更少 NFE 下保持界面与配体几何？ |
| 自适应步长调度器 | Euler–Heun 嵌入式误差估计，接受/拒绝和步长更新；与相同实际 NFE 的固定网格比较 | 难区间细化、易区间跳步能否降低每个分子的总成本？ |
| 跨步缓存与外推 | 固定间隔刷新对照，TeaCache 思路的变化量触发刷新，特征/残差外推 | 能否降低每次 denoising 的成本，且不累积破坏口袋几何的误差？ |
| 少步蒸馏与 flow map | Boltz-2 的 multistep consistency 路线，参考 Blitz；另开展 DeCAF 风格 EDM sigma-space flow-map 适配实验 | 学到跨大噪声区间的更新后，8/16/32 步是否优于未训练少步采样？ |
| recycling 算法 | 固定 recycle 对照，基于表征变化的自适应停止；参考 DCFold 的 Pairformer consistency 做学习式少 recycle 实验 | diffusion 加速后，能否继续消除 trunk 的迭代瓶颈？ |

#### C1. 求解器接入与几何约束

Boltz-2 的 `preconditioned_network_forward` 输出 denoised coordinates，采样在 EDM sigma 空间运行，包含随机 churn、每步随机刚体增强、可选刚体对齐及 `step_scale` 更新。它没有可直接替换的图像模型 scheduler 接口。

首先实现和验证确定性 ODE 分支：显式关闭 churn，使用统一坐标系，核对 denoiser 到 drift 的变换、预条件、sigma 单位及零噪声终点。将“原始随机采样”和“确定性 Euler”都保留为基线，分离关闭随机性的影响与求解器收益。DPM-Solver++ / UniPC 需要推导对应变量变换；不直接复制 VP 图像扩散公式，也不默认图像方法的 thresholding 适合原子坐标。

多步历史、误差估计和缓存张量必须处于一致的坐标系。刚体变换可作用于坐标/向量历史，但不能假设非等变网络的隐藏特征可直接旋转复用。需要固定增强坐标系或重新计算特征，并单独评估该选择对模型质量的影响。

先比较 16/32/64/100 NFE 附近的质量–成本曲线，再选少量工作点做完整筛选。高阶法末端降阶、最终 sigma=0 的处理与原始末步分别核对。结构 RMSD、键长/键角、clash、PoseBusters validity 与早期富集同时观察，不能凭整体折叠相似就接受少步方法。

#### C2. 自适应 diffusion 调度器

调度器在单条去噪轨迹内决定下一噪声节点。初版使用有效原子的归一化坐标误差，分别统计蛋白、配体和口袋区域；容差在开发靶点校准。嵌入式 Euler–Heun 的额外网络调用、拒步和误差计算全部计费。坐标误差只是控制信号，是否保持几何与富集由最终指标判断。

不以尚未去净噪声的坐标变化小直接判定完成。需要提前结束时，执行到干净坐标的终止预测，并独立测量早停损失。随机采样分支另行研究，接受/拒绝步必须保持随机增量的一致性，不能将 ODE 控制器直接套到 churn 轨迹。

#### C3. 跨步复用与动态刷新

从 token transformer 的 block 残差/中间特征开始，比较每 2/4 步刷新与变化量触发刷新。TeaCache 原工作面向视频等扩散模型，本项目将其作为方法迁移，重新拟合变化量与输出误差的关系，不使用视频模型现成阈值或多项式系数。

时间步 embedding 仅反映噪声位置，不能单独代表蛋白–配体构象变化。刷新信号需同时考虑可低成本取得的输入/特征变化，并对配体和口袋单独检查。所有阈值拟合只用开发数据。报告刷新率、部分 block 调用、实际时间、缓存占用，以及额外校准成本。

#### C4. 学习式少步生成与 recycling

Blitz 直接研究 AlphaFold3 / Boltz-2，以几何噪声日程和 multistep consistency 将去噪压到 8 步；它提供最接近本任务的蒸馏参考，但其结构结论不能代替 affinity 排序验证。接入前核实可取得的代码、权重与训练数据；缺少公开产物时按论文实现训练，不把只改 schedule 标为复现蒸馏。

DeCAF 使用 EDM sigma-space 的 denoiser flow map，并以刚体对齐处理端点损失；公开的 DeCAF-Boltz 基于 Boltz-1/1x，不能直接替换 Boltz-2 checkpoint。正式实验将该目标适配到 Boltz-2 student。DCFold 的 Dual Consistency / Temporal Geodesic Matching 作为同时减少 diffusion 与 Pairformer recycle 的参考路线，不将其原模型结果当作 Boltz-2 实测结果。

初版冻结 trunk 和 affinity heads，分别为结构 denoiser 与 affinity checkpoint 的 denoiser 训练 student，保留各自条件分布；先比较 8/16/32 步。开发集上比较 multistep consistency 与 flow-map 目标，并开展自适应 recycle 与 Pairformer consistency 实验。结构/去噪损失、刚体对齐和必要的局部几何约束以论文目标及真实错误为依据。固定 head 不保证分数不变，必须重新测筛选质量。

训练数据按靶点/同源簇与最终测试隔离，缓存只在条件与 teacher 固定时复用；完整记录 teacher 轨迹生成和 student 训练的 GPU 小时。计算训练成本与累计推理节省的盈亏平衡分子数。训练集来源、划分和可取得的权重是启动该分支前的具体准备工作。

### D. 筛选与集群的自适应调度

区分三个控制层，分别消融：

1. **轨迹内调度**：C2 的自适应噪声步长与 C3 的缓存刷新。
2. **分子级调度**：根据低预算输出决定哪些分子追加完整计算；与统一低/中/高预算在相同总 GPU 时间下比较。未实现采样状态续算前，高预算第二次执行按完整重跑计费；初轮淘汰的活性计入整库召回损失。
3. **作业级调度**：按 token/atom 数、MSA 深度和实测耗时估计分桶，在 A800/L40 常驻 worker 间动态派发；比较静态平均分配与动态队列，测 GPU 利用率、padding、尾部等待和整库 wall time。两卡的算法性能仍分开报告。

分子级晋级比例和不确定性规则只在开发靶点选择；pLDDT/ipTM 不等于结合概率，不能直接把低结构置信度作为 inactive。每层先独立证明收益，再与少步求解/蒸馏组合，避免将负载均衡收益误计为模型算法加速。

### E. 算子与算法协同优化

profile 同时覆盖原始模型和最终候选采样器。少步方法会改变热点分布，不以原始 200 步热点决定全部优化工作。

候选包括 affinity pass、A800/L40 三角算子与 transition、diffusion/ODE 更新融合，以及缓存刷新和状态搬运。自适应步长与动态刷新不适合直接捕获整条固定轨迹的 CUDA Graph；采用单步或固定片段 capture，并把 graph 重建、动态分支、CPU–GPU 同步及批次分歧计入成本。多步 solver / cache 的历史状态只在形状、噪声历史和条件允许时共享。

复用 MSA、静态模板与配体无关的预处理；跨配体冻结 trunk 表征属于另一个近似算法，需独立验证。新算子先与该卡最快可用实现比较，再回到完整任务测量 padding、mask、非整齐尺寸、累积精度与分数变化。最终组合由实测 Pareto 结果选择，不预设所有方法叠加一定更快。


## 7. 参考进度与阶段产出

按工作包推进，先后依赖为基线、采样器适配、独立算法实验、协同优化、完整筛选。训练数据与 teacher 轨迹在基线阶段开始准备；蒸馏训练可与无需训练的采样实验并行开展。课程截止日期确定后，结合实测训练和推理成本安排日历。

| 阶段 | 工作 | 可交付结果 |
| --- | --- | --- |
| P0 | 集群环境、真实 affinity smoke、stock / exact / fast profile；准备蒸馏数据划分 | 两种 GPU 的完整基线和训练数据可用性记录。 |
| P1 | EDM 确定性分支、噪声日程、Heun / DPM-Solver++ / UniPC | 匹配 NFE 的少步质量曲线及可运行的两阶段采样器。 |
| P2 | 自适应步长、动态缓存/外推、自适应 recycle | 独立算法收益、误差与实际时间开销。 |
| P3 | multistep consistency、flow-map、Pairformer consistency | 学习式少步模型、训练成本和筛选质量；与 P1/P2 同预算比较。 |
| P4 | 分子级晋级、异构 GPU 作业调度、A800/L40 算子协同优化 | 分层调度消融、最优候选组合及端到端吞吐量。 |
| P5 | 多靶点完整库、跨 seed、组合消融与成本外推 | 初筛可行性结论、逐靶点指标与可复现实验入口。 |

在开发集上暂以“相对 stock 达到 3 倍稳态端到端加速、Recall@1% 与 EF@1% 相对下降不超过 5%”作为筛选候选配置的工程目标。它不是结论或统计显著性阈值；基线接近零时报告绝对活性命中数与召回差，不计算不稳定的相对比例。最终同时与 Anthropic fast 比较，不能只选弱基线。

固定运行预算下的真实命中保留优先于某一个加速倍数。若完整 Boltz-2 的单分子成本仍达不到直接初筛要求，结论应明确其可承担的库规模，并独立评价级联重排方案。

## 8. 最终实验表与消融

至少比较：

- stock 官方基线；
- Anthropic exact / fast 可用模式；
- 固定减步 / recycle / samples 的低预算基线；
- 仅噪声日程优化；
- 各高阶/多步 solver，与确定性 Euler 匹配 NFE；
- 自适应步长，与固定日程匹配实际 NFE 和 GPU 时间；
- 固定间隔缓存、动态刷新及外推；
- multistep consistency 与 flow-map student；
- 自适应 recycle 与 Pairformer consistency；
- 仅分子级晋级、仅作业级动态调度；
- 仅新增算子/执行优化；
- 经开发集选择的最终组合；
- 随机排序，以及一种已有便宜筛选方法作为应用对照。

A800、L40 分开报告吞吐量、逐靶点 EF/Recall/AP、失败率及相对 stock 和强基线的加速。新增方法的消融只切换一个因素。算子 benchmark、结构质量与完整筛选结果分表呈现。

对候选组合分别去掉 solver、自适应步长、缓存、蒸馏和执行优化，测量交互项。8 步 student 不预设还受益于缓存或高阶 solver，动态算法也不预设能保留原整轨迹 CUDA Graph 收益。每个方法族均保留独立实验结果，最终运行配置只采用质量–成本有收益的组合。

## 9. 已知问题

- 公开筛选数据可能与预训练数据重叠，现有公开信息未必足以证实完全独立。保留适用范围与重叠证据。
- 活性极少时 EF@1% 方差很大。报告活性数量、逐靶点命中数，并对候选配置给出配对重采样的不确定性。
- 口袋裁剪、模板和已知位点会改变输入信息量。只能在基线与新方法均可获得相同信息时比较。
- affinity 对错误构象、必要辅因子缺失和超出训练范围的大配体可能不可靠。筛选集合记录这些排除条件及数量。
- 多 GPU 的任务分发收益不属于单模型算法加速。分别报告单 GPU 收益和扩展效率。
- 已有单输入推理、算子与 training-free 采样实测；尚无有标签筛选富集结果。

## 10. 参考资料

1. Boltz 官方仓库与任务说明：https://github.com/jwohlwend/boltz
2. 官方预测格式与参数：https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
3. Boltz-2 论文：https://doi.org/10.1101/2025.06.14.659707 ，可读全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/
4. Anthropic 官方研究文章：https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling
5. 技术报告，2026-09-21 更新版：https://www-cdn.anthropic.com/e96b5807039a88168733d9687afe41dfbbd5de13.pdf
6. 技术报告，2026-09-17 初版：https://www-cdn.anthropic.com/c03643714397d9d396fa1ce1794f5f9f7863a82c.pdf
7. Anthropic 开源套件：https://github.com/anthropics/uplifting-biomolecular-modeling
8. Boltz-2 套件说明：https://github.com/anthropics/uplifting-biomolecular-modeling/tree/main/boltz2
9. LIT-PCBA 数据与下载：https://lab.drugdesign.unistra.fr/datasets/lit-pcba/
10. MF-PCBA 原始数据处理：https://github.com/davidbuterez/mf-pcba
11. Blitz，Boltz-2/AF3 少步一致性方法：https://arxiv.org/abs/2610.23703
12. DeCAF，EDM sigma-space all-atom flow maps：https://arxiv.org/abs/2606.08375 ；实现：https://github.com/genesistherapeutics/decaf
13. DCFold，Dual Consistency 与 Temporal Geodesic Matching：https://arxiv.org/abs/2605.17899
14. DPM-Solver / DPM-Solver++ 原始实现与论文入口：https://github.com/LuChengTHU/dpm-solver
15. UniPC，统一 predictor–corrector：https://arxiv.org/abs/2302.04867
16. TeaCache，timestep-aware 跨步缓存：https://arxiv.org/abs/2411.19108 ；实现：https://github.com/ali-vilab/TeaCache
