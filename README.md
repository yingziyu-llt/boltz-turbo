# Boltz-Turbo (`boltz-turbo`)

面向虚拟筛选初筛场景的 **Boltz-2**（`boltz==2.2.1`）蛋白–小分子结构与亲和力（`affinity_probability_binary`）扩散推理加速研究代码与实验记录。完整研究设计、实验口径与阶段结论见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 目录结构

```text
.
├── README.md                       # 项目说明与快速上手
├── RESEARCH_PLAN.md                # 研究计划与各阶段实验记录
├── data/                           # 最小可复现输入（官方 v2.2.1 affinity 示例）
│   ├── smoke.yaml
│   └── msa/seq1.a3m
├── experiments/                    # 独立实验入口脚本
│   ├── exp_baseline.py             # 同靶点多配体基线吞吐与分阶段 profile
│   ├── exp_cuda_graph.py           # 常驻 denoiser 与 CUDA Graph 验证
│   ├── exp_resident_compare.py     # 持久化模型加载 + 交错稳态多方法公平对比（含分阶段预算解耦）
│   ├── exp_profile_hsp90.py       # HSP90 两阶段 CUDA Event 分账与独立算子追踪
│   ├── exp_anthropic_l40.py        # Anthropic boltz2_opt 套件 L40 裁剪适配
│   ├── exp_sm89.py                 # L40 (sm_89) 自定义算子组合推理入口
│   ├── exp_training_free.py        # Training-free 采样器、DPM-Solver++ 2M 与稳定性实验
│   ├── exp_vina_hsp90.py           # AutoDock Vina HSP90 180 配体对接基线与单核/多核吞吐测试
│   ├── boltz_trunk_opt.py          # Structure 零漂移 Affinity-only TeaCache 优化模块
│   ├── exp_trunk_teacache.py       # H800 Trunk-TeaCache 与自适应 Early-Exit 验证
│   ├── exp_affinity_reuse.py       # H800 Structure 零漂移 + Affinity MSA/Pairformer 复用验证
│   └── exp_cross_ligand_cache.py   # H800 跨配体蛋白特征缓存探索
├── analysis/                       # 离线指标汇总与结构漂移分析
│   ├── analyze_anthropic.py        # Anthropic 套件各模式耗时与误差汇总
│   ├── analyze_structure_drift.py  # 高置信度核心对齐与结构漂移分析
│   ├── analyze_training_free.py    # Training-free 12 组采样对比
│   ├── analyze_dpm.py              # DPM-Solver++ 2M 变体消融汇总
│   ├── analyze_dpm_confidence.py   # DPM 按 pLDDT 置信度分区 RMSD 分析
│   ├── analyze_stability.py        # 多体系/多种子稳定性与对称校正配体 RMSD 分析
│   ├── analyze_resident_compare.py # 持久化稳态公平对比汇总与 Markdown 制表
│   ├── eval_hsp90.py               # HSP90 实验 IC50/pIC50 相关性、分骨架指标与早期富集评测
│   └── eval_vina_hsp90.py          # Vina 与 Boltz-2（180 全量 / 60 分层子集）同口径对照评测
├── sm89/                           # NVIDIA L40 (Ada sm_89) 算子适配补丁与微基准
│   ├── install_sm89.py             # 写入 zz_boltz_sm89.pth 启动钩子
│   ├── sm89_startup.py             # 由 SM89_ACTIVE / SM89_FULL 触发的补丁入口
│   ├── sm89_opm.py                 # OuterProductMean 低共享内存 tile 配置
│   ├── sm89_pair.py                # Pairformer TriAttn / Transition sm_89 配置
│   ├── sm89_exact.py               # Exact LayerNorm / DiT / MSA / PWA sm_89 适配
│   └── smoke_sm89_*.py             # OPM / Pair / Exact / PWA 单算子验证脚本
├── scripts/                        # 辅助运行、环境构建与数据准备脚本
│   ├── run_smoke.sh                # 官方 stock Boltz-2 端到端 smoke 测试
│   ├── setup_anthropic.sh          # Anthropic 套件独立 venv 构建脚本
│   ├── run_sm89_compare.sh         # 旧 fast vs sm89 OPM fast 交错对比
│   ├── run_sm89_full.sh            # sm89 组合 exact/fast 端到端运行
│   └── prepare_stability.py        # 下载并生成 3PTB/1IEP/1STP 稳定性输入
├── slurm/                          # 集群 Slurm 作业脚本（含 A800/L40 常驻对比、HSP90 评测与 Vina 基线）
└── results/                        # 轻量级实验状态摘要 (*_status.json)
```

## 环境要求

- **Python**: `3.11`（Boltz 声明 `>=3.10,<3.13`）
- **Boltz**: `boltz==2.2.1`（固定版本，与 Anthropic 参考套件对齐）
- **GPU / CUDA**:
  - **集群统一主环境（`boltz2`，A800 + 全 L40 节点已验证）**：`torch==2.11.0+cu128`、`triton==3.6.0`、`cuequivariance==0.11.1`（`cuequivariance_torch==0.11.1` / `cuequivariance_ops_torch==0.11.1`），兼容集群 `575.57.08` 与 `590.44.01` 两种宿主驱动（切换前 `pip freeze` 备份于集群 `tmp/boltz2_pre_cu12_freeze.txt`）。
  - **早期 L40 独立基线栈**：PyTorch `2.13.0+cu130`、Triton `3.7.1`、cuEquivariance `0.11.1`（需 `590.x` 等支持 CUDA 13 的宿主驱动）。
  - **Anthropic 套件对照环境（`anthropic-env`，独立保留未改动）**：PyTorch `2.12.0+cu130`、Triton `3.7.0`、cuEquivariance `0.10.0`，通过 `scripts/setup_anthropic.sh` 构建（需要支持 CUDA 13 的驱动）。
  - **分子对接基线环境（`fap-docking`）**：`AutoDock Vina f458505-mod`、`RDKit 2026.03.1`、`Meeko 0.6.1`、`OpenMM / PDBFixer`（在无 GPU 节点的 `cn-short` 运行时需设 `OPENMM_PLUGIN_DIR=/dev/null` 并使用 `Reference` 平台）。
- **权重缓存**: 运行前设置 `export BOLTZ_CACHE=/path/to/boltz-cache`（包含 `boltz2_conf.ckpt` 与 `boltz2_aff.ckpt`）。

## 快速运行（均从仓库根目录执行）

### 1. 官方基线 Smoke 与吞吐/Profile

```bash
export BOLTZ_CACHE=/path/to/boltz-cache
bash scripts/run_smoke.sh

# 同靶点 6 配体吞吐基线 & 分阶段 profile
python experiments/exp_baseline.py
python experiments/exp_baseline.py --profile
```

### 2. 常驻 Denoiser、CUDA Graph 与持久化稳态对比

```bash
python experiments/exp_cuda_graph.py
python experiments/exp_cuda_graph.py --graph

# 持久化模型加载 + 交错稳态公平对比（默认 6 配体，--suite fair / decoupled / quick）
python experiments/exp_resident_compare.py --suite fair --repeats 2 --out-dir results/resident_compare_fair
python analysis/analyze_resident_compare.py --dir results/resident_compare_fair
```

HSP90 精细 profiling：在北极星提交 `sbatch slurm/hsp90_profile.slurm`。每种配置先预热，Pass A 用相同的 8 个配体做 CUDA Event 排他分账；Pass B 只追踪 `hsp90_1`，不计入吞吐。输出在 `results/hsp90_profile_<jobid>/`：`profile.json` 及各配置、阶段的 Chrome trace。CUDA Event 时间包含流内空隙；CUDA 活跃率用 Pass B 的活动区间并集计算，不能与 Pass A 的墙钟时间直接相减。

### 3. HSP90 实验亲和力基准与 AutoDock Vina 对接基线

```bash
# A800 HSP90 180 配体全量 (fair) 与 60 分层子集阶段解耦 (decoupled)
sbatch slurm/hsp90_compare_a800.slurm fair /lustre3/lhlai_pkuhpc/llt/hsp90_affinity/inputs
sbatch slurm/hsp90_compare_a800.slurm decoupled data/hsp90_strat60
python analysis/eval_hsp90.py --dir results/hsp90_a800_fair

# AutoDock Vina HSP90 180 配体对接基线 (16 CPU 核) 与同口径对照评测
sbatch slurm/vina_hsp90.slurm
python experiments/exp_vina_hsp90.py --dir results/hsp90_vina
python analysis/eval_vina_hsp90.py --dir results/hsp90_vina --boltz-dir results
```

### 4. Training-free 采样与多体系稳定性

```bash
# 单体系 DPM-Solver++ 2M (16 / 32 NFE)
python experiments/exp_training_free.py --method dpm2m --steps 16
python experiments/exp_training_free.py --method dpm2m --steps 32
python analysis/analyze_training_free.py

# 多体系 (smoke / 3PTB / 1IEP / 1STP) 稳定性数据准备与分析
python scripts/prepare_stability.py
python analysis/analyze_stability.py
```

### 5. Anthropic 套件 L40 适配与 `sm_89` 算子

```bash
# 构建独立环境（需先将 anthropic-kit 放在仓库根目录）
ROOT="$PWD" BASE="$(command -v python3)" bash scripts/setup_anthropic.sh

# 安装 sm_89 启动钩子并验证单算子
python sm89/install_sm89.py
python sm89/smoke_sm89_opm.py
python sm89/smoke_sm89_pair.py
python sm89/smoke_sm89_exact.py
python sm89/smoke_sm89_pwa.py

# 组合推理
bash scripts/run_sm89_full.sh exact fast
```

> **提示**：`slurm/*.slurm` 保留了当前集群（L40 / A800 / CPU `cn-short`）的提交配置作为参考；合作者在其他集群上使用时，请修改 `#SBATCH --chdir`、`--output`、`--account`、`--qos` 以及 `PATH` / `BOLTZ_CACHE`。

## 已完成阶段摘要

关键实测结果保存在 `results/*_status.json`，详细分析与下一步 Profiling/优化交接清单见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)：

| 阶段 | 状态文件 | 核心观察 |
| --- | --- | --- |
| P0 Smoke | `results/p0_status.json` | L40 完整结构 + affinity 跑通，`affinity_probability_binary=0.3151` |
| 基线与 Profile | `results/baseline_status.json` | L40 6 配体 CLI 批量 33.4–36.1 s/分子；diffusion 占主要计算时间 |
| 常驻 + CUDA Graph | `results/resident_status.json` | L40 结构 warm sample 1.416×（6.16 s → 4.35 s），与 eager 逐位一致；affinity 1.021× |
| Anthropic L40 适配 | `results/anthropic_status.json` | L40 适配 `exact` 与 stock 字节一致；`fast` 第二轮结构 PHASE 3.154 s（stock 10.525 s） |
| `sm_89` 单卡算子 | `results/sm89_status.json` | L40 补齐 OPM / Pair / Exact LN+DiT+PWA 路径，完整 `exact` 字节一致 |
| Training-free & DPM | `results/training_free_status.json`, `results/dpm_status.json`, `results/dpm_confidence_status.json` | L40 16/32 NFE DPM-Solver++ 2M 采样加速 ~6.3–12.2×；多体系 5 种子稳定性完成 |
| 持久化稳态对比 (A800/L40) | `results/resident_compare_a800_status.json`, `results/resident_compare_l40_status.json` | 6 配体交错稳态：`dpm2m_16` A800 **3.03×**（27.32→9.03 s/mol）、L40 **2.54×**（24.63→9.71 s/mol）；扩散缩减后 Trunk 占 83–87% |
| HSP90 真实实验基准 (A800) | `results/hsp90_a800_fair_status.json`, `results/hsp90_a800_decoupled_status.json` | 180 全量 `dpm2m_16` **3.02×**（25.02→8.29 s/mol，vs `stock_200` ρ=0.994，≤100nM EF10%=3.66×）；60 子集 `dpm2m_16_affrec1` **4.29×**（25.39→**5.91 s/mol**，突破 6.9 s/mol 目标，r=0.494/ρ=0.425/≤100nM AUROC=0.842） |
| HSP90 AutoDock Vina 基线 | `results/hsp90_vina_status.json` | `4YKR` holo 受体 180/180 零失败，16 核稳态 3.07 s/mol（单核 41.13 s/mol）；虽具晶体结构与已知口袋优势，≤100nM EF10%（2.20×）/AP（0.460）仍显著低于 Boltz-2（3.66×/0.664），且呈现强骨架互补 |
| HSP90 Trunk 优化与缓存 (H800) | `results/hsp90_h800_affkern_status.json`, `results/hsp90_trunk_teacache_status.json`, `results/hsp90_aff_reuse_status.json`, `results/hsp90_xlig_cache_status.json` | 60 配体 `dpm2m_16_affkern` 6.854→5.720 s/mol (**1.20×**, `prob_rho=0.9948`)；12 配体 `aff_msa_r1_adapt05` 6.277→**4.959 s/mol** (**1.27×**, 0.000 Å RMSD, `prob_rho=0.9790`) 已落地至 `dpm2m_16_affkern_tc05`（60 配体作业 `32021024` 排队中）；跨配体缓存达 1.33–1.61× 但精度退化，下一步转向真实 Batching |
