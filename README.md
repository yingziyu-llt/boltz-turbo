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
│   ├── exp_anthropic_l40.py        # Anthropic boltz2_opt 套件 L40 裁剪适配
│   ├── exp_sm89.py                 # L40 (sm_89) 自定义算子组合推理入口
│   └── exp_training_free.py        # Training-free 采样器、DPM-Solver++ 2M 与稳定性实验
├── analysis/                       # 离线指标汇总与结构漂移分析
│   ├── analyze_anthropic.py        # Anthropic 套件各模式耗时与误差汇总
│   ├── analyze_structure_drift.py  # 高置信度核心对齐与结构漂移分析
│   ├── analyze_training_free.py    # Training-free 12 组采样器对比
│   ├── analyze_dpm.py              # DPM-Solver++ 2M 变体消融汇总
│   ├── analyze_dpm_confidence.py   # DPM 按 pLDDT 置信度分区 RMSD 分析
│   └── analyze_stability.py        # 多体系/多种子稳定性与对称校正配体 RMSD 分析
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
├── slurm/                          # 集群 Slurm 作业脚本（按需修改路径与账号）
└── results/                        # 轻量级实验状态摘要 (*_status.json)
```

## 环境要求

- **Python**: `3.11`（Boltz 声明 `>=3.10,<3.13`）
- **Boltz**: `boltz==2.2.1`（固定版本，与 Anthropic 参考套件对齐）
- **GPU / CUDA**:
  - Stock / Training-free 实验环境（L40 已验证）：PyTorch `2.13.0+cu130`、Triton `3.7.1`、cuEquivariance `0.11.1`（需支持 CUDA 13 的驱动，如 `590.x`）。
  - Anthropic 套件对照环境（L40 已验证）：PyTorch `2.12.0+cu130`、Triton `3.7.0`、cuEquivariance `0.10.0`，通过 `scripts/setup_anthropic.sh` 在独立 `anthropic-env/` 中安装。
  - A800 节点若宿主驱动低于 CUDA 13 要求，需使用对应 CUDA 兼容版本的独立环境。
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

### 2. 常驻 Denoiser 与 CUDA Graph

```bash
python experiments/exp_cuda_graph.py
python experiments/exp_cuda_graph.py --graph
```

### 3. Training-free 采样与多体系稳定性

```bash
# 单体系 DPM-Solver++ 2M (16 / 32 NFE)
python experiments/exp_training_free.py --method dpm2m --steps 16
python experiments/exp_training_free.py --method dpm2m --steps 32
python analysis/analyze_training_free.py

# 多体系 (smoke / 3PTB / 1IEP / 1STP) 稳定性数据准备与分析
python scripts/prepare_stability.py
python analysis/analyze_stability.py
```

### 4. Anthropic 套件 L40 适配与 `sm_89` 算子

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

> **提示**：`slurm/*.slurm` 保留了当前集群（L40 / A800）的提交配置作为参考；合作者在其他集群上使用时，请修改 `#SBATCH --chdir`、`--output`、`--account`、`--qos` 以及 `PATH` / `BOLTZ_CACHE`。

## 已完成阶段摘要

关键实测结果保存在 `results/*_status.json`，详细分析见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)：

| 阶段 | 状态文件 | 核心观察（L40） |
| --- | --- | --- |
| P0 Smoke | `results/p0_status.json` | 完整结构 + affinity 跑通，`affinity_probability_binary=0.3151` |
| 基线与 Profile | `results/baseline_status.json` | 6 配体 CLI 批量约 33.4–36.1 s/分子；diffusion 占主要计算时间 |
| 常驻 + CUDA Graph | `results/resident_status.json` | 结构 warm sample 1.416×（6.16 s → 4.35 s），与 eager 逐位一致；affinity 1.021× |
| Anthropic L40 适配 | `results/anthropic_status.json` | 适配 `exact` 与 stock 字节一致；`fast` 第二轮结构 PHASE 3.154 s（stock 10.525 s） |
| `sm_89` 单卡算子 | `results/sm89_status.json` | 补齐 L40 OPM / Pair / Exact LN+DiT+PWA 路径，完整 `exact` 字节一致 |
| Training-free & DPM | `results/training_free_status.json`, `results/dpm_status.json`, `results/dpm_confidence_status.json` | 16/32 NFE DPM-Solver++ 2M 采样加速 ~6.3–12.2×；多体系稳定性已完成，待进入有标签筛选 pilot |
