# TDA Pipeline (Embeddings-Only)

Minimal reproducible pipeline for 1D persistent homology on paired embedding clouds.

最小可复现的 embeddings-only TDA 管线：对两组文本向量分别进行 1D persistent homology 分析，并输出 persistence diagram、H1 persistent entropy (PE) 与 H1 bar count 等可比较指标。

这个仓库的定位不是“完整论文工程”，而是“可公开、可复现、可演示的方法论骨架”。它强调的是方法本身的可运行性、可检查性和可复现性。

## 你会在这里看到什么

- 一个只依赖 embeddings 的最小 TDA 分析管线
- 一个不含私有语料的 toy demo，可直接跑通
- 一套适合研究展示的公开边界设计
- 一组可直接引用的方法指标：persistence diagram、H1 PE、H1 bar count

## 为什么这个仓库值得放在 GitHub 首页

- 它不是“论文全量工程”，而是公开可验证的技术骨架
- 它把私有研究材料和公开方法证据分开
- 它能展示方法设计的可运行性，而不泄露原始语料

## Quick Start

首次进入仓库，只需要两步：

```bash
pip install -r requirements.txt
python -m tda_pipeline.demo_synthetic
```

运行后默认生成：

- `out_synth/metrics_1d.json`
- `out_synth/pd_src.png`
- `out_synth/pd_tgt.png`

如果你只想在 30 秒内判断这个仓库是否可复现，就运行上面这两条命令。

## 公开边界

这个仓库默认只公开安全部分：

- 仅处理 embeddings（`np.ndarray`），不读取或要求原始文本
- 不包含语料、数据库、索引、账号配置或 API 密钥
- 不写死任何本机绝对路径；所有输入输出均通过 CLI 传入
- demo 仅使用合成数据，不触碰私有实验材料

## 环境依赖

- Python `>=3.10`
- `numpy>=1.24`
- `matplotlib>=3.7`
- `ripser>=0.6.12`

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果你已经在现有环境中工作，也可以直接：

```bash
pip install -r requirements.txt
```

## 仓库提供的能力

| 模块 | 作用 | 典型输出 |
| --- | --- | --- |
| `tda_pipeline/core.py` | 核心数据处理、persistent homology 与指标函数 | diagram arrays、PE、bar count |
| `tda_pipeline/run_1d.py` | 对单个 `.npz` 配对样本运行一次 1D 分析 | `metrics_1d.json`、PD 图 |
| `tda_pipeline/sweep_1d.py` | 做样本量或阈值 sweep 稳定性检查 | `sweep_1d.json`、曲线图 |
| `tda_pipeline/demo_synthetic.py` | 生成合成 embedding 并跑通最小示例 | `out_synth/` 输出目录 |
| `examples/synthetic_demo.py` | 示例级脚本入口，便于快速演示 | demo 结果文件 |

## 目录结构

```text
tda_pipeline_public/
├─ tda_pipeline/
│  ├─ core.py
│  ├─ run_1d.py
│  ├─ sweep_1d.py
│  └─ demo_synthetic.py
├─ examples/
│  └─ synthetic_demo.py
├─ README.md
├─ LICENSE
├─ requirements.txt
└─ .gitignore
```

## 输入格式

使用 `.npz` 文件保存两组向量，支持以下任意键名组合：

- `src` / `tgt`
- `src_emb` / `tgt_emb`
- `src_embeddings` / `tgt_embeddings`
- `source` / `target`
- `pairs`（shape 为 `[2, n, d]`）

## 方法论最小定义

这套管线固定做三件事：

1. 预处理：对两组 embeddings 可选做 L2 normalization 与 PCA 降维
2. 持久同调：分别运行 `ripser(maxdim=1)` 计算 H0/H1 persistence diagram
3. 汇总指标：默认输出 H1 persistent entropy (PE) 与 H1 bar count

为什么重点看 H1：

- H0 更接近基础连通过程，适合 sanity check
- H1 更接近局部到整体的组织复杂度，更适合做主证据链
- PE 反映的是 H1 寿命分布，而不是简单的“环数量”

## 最简示例

直接运行合成示例：

```bash
python -m tda_pipeline.demo_synthetic
```

默认输出到 `out_synth/`：

- `metrics_1d.json`
- `pd_src.png`
- `pd_tgt.png`

这就是仓库里的 toy demo。它不依赖任何私有语料，能够直接跑通一条最小 TDA 分析链。

如果你需要展示这个仓库，最值得截图的通常是：

- GitHub 首页上的 README 说明
- `metrics_1d.json` 中的核心指标
- `pd_src.png` / `pd_tgt.png` 这样的 persistence diagram 输出

## 单次 1D 指标

```bash
python -m tda_pipeline.run_1d --npz path/to/pair.npz --outdir out --sample-size 500 --pca-dim 64
```

默认输出：

- `metrics_1d.json`
- `pd_src.png`
- `pd_tgt.png`

常用选项：

- `--no-normalize`：关闭 L2 normalization
- `--pca-dim 0`：关闭 PCA
- `--thresh <float>`：设置 ripser 的 filtration 上限
- `--no-plots`：只保留 JSON，不输出图件

## 样本量 Sweep

```bash
python -m tda_pipeline.sweep_1d --npz path/to/pair.npz --outdir out --sizes 250,500,1000
```

可选阈值 sweep：

```bash
python -m tda_pipeline.sweep_1d --npz path/to/pair.npz --outdir out --sizes 250,500,1000 --thresh-grid 0.2,0.3,0.4 --fixed-size 500
```

默认输出：

- `sweep_1d.json`
- `betti1_curve_size_*.png`

## 适用场景

- 数字人文或翻译研究中的方法展示
- embeddings 几何结构比较的教学或演示
- 公开研究代码的复现展示
- 不方便公开原始文本时的可复现方法证明

## 隐私与开源注意事项

不要上传：

- 任意包含原文/译文的 JSONL、CSV、数据库或索引
- 写入本机路径的元数据文件
- `.env`、密钥、token、账号配置
- 能回溯到具体样本的逐条记录

推荐上传：

- 公开代码
- 合成数据 demo
- 聚合后的 JSON 指标或示意图
- 与方法相关的简要说明文档
