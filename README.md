# 罪名预测复现与层次分类提效

本仓库只保留“罪名预测”部分，不包含法条推荐。

当前实现围绕论文第 4 章做了重新整理，目标是：
- 使用 `110 类单标签` 任务复现论文中的层次分类实验口径
- 保留两组同骨干对照实验：`BERT-FC flat vs hierarchical`、`BERT-RCNN flat vs hierarchical`
- 支持服务器 `CUDA` 训练、批量推理导出、结果表汇总和可视化分析数据导出

## 1. 任务口径

- 仅保留单罪名样本
- 粗类固定为 3 个高频大类：
  - `侵犯财产罪`
  - `侵犯公民人身权利、民主权利罪`
  - `危害公共安全罪`
- 细类默认按 `top_k_labels=110` 选择；如果开启稳定重切分且原始支持度不足，实际保留标签数可能少于 `110`
- 默认抽样 `50000` 条，输出训练/验证/测试集和分析 CSV

## 2. 目录说明

- `charge_prediction/`: 公共模块
- `scripts/prepare_data.py`: 构建论文口径 `110` 类数据
- `scripts/train_deep_models.py`: 训练平层 `BERT-FC` / `BERT-RCNN`
- `scripts/train_deep_hierarchical.py`: 训练两阶段层次模型，并在验证集上调优回退门控
- `scripts/predict.py`: 单条或批量推理，支持深度平层/层次模型
- `scripts/make_results_table.py`: 生成主结果表、对照表和中间步骤表
- `scripts/run_pipeline.py`: 一键执行全流程

## 3. 环境安装

建议在服务器上执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你使用 CUDA 服务器，建议先确认 `torch` 已安装为对应的 CUDA 版本。

### 3.1 Windows 训练机直接复制版

如果你已经把仓库、数据和本地 BERT 同步到 Windows 机器，并且目录与下面一致，可以直接复制本节命令到 `cmd` 中执行：

- 项目目录：`D:\project\charge-prediction-paper-repro`
- 本地 BERT：`D:\project\charge-prediction-paper-repro\chinese-bert-wwm-ext`
- 原始数据：`D:\project\2018数据集\2018数据集`
- 已验证可用的 conda 环境：`D:\conda_envs\caoyao-resnet`

先进入环境和项目目录：

```bat
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet
cd /d D:\project\charge-prediction-paper-repro
set BERT_DIR=D:\project\charge-prediction-paper-repro\chinese-bert-wwm-ext
```

检查 GPU 和 PyTorch：

```bat
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

最小烟雾测试：确认模型、CUDA、数据集都能跑通。这个命令只跑很少样本和 `1` 个 epoch，适合先验环境。

```bat
python scripts\train_deep_models.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_smoke\deep_models ^
  --models fc ^
  --device cuda ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 1 ^
  --train-batch-size 2 ^
  --eval-batch-size 2 ^
  --gradient-accumulation-steps 1 ^
  --selection-metric accuracy ^
  --max-train-samples 64 ^
  --max-valid-samples 32 ^
  --max-test-samples 32
```

如果你要先正式跑一个最稳的版本，建议从 `BERT-FC` 开始：

```bat
python scripts\train_deep_models.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_paper\deep_models ^
  --models fc ^
  --device cuda ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 2 ^
  --eval-batch-size 4 ^
  --gradient-accumulation-steps 4 ^
  --selection-metric accuracy
```

如果你要同时跑 `BERT-FC` 和 `BERT-RCNN` 两个平层模型，用下面这条：

```bat
python scripts\train_deep_models.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_paper\deep_models ^
  --models fc rcnn ^
  --device cuda ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 2 ^
  --eval-batch-size 4 ^
  --gradient-accumulation-steps 4 ^
  --selection-metric accuracy
```

跑 `BERT-FC` 的层次模型：

```bat
python scripts\train_deep_hierarchical.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_paper\deep_hierarchical_fc ^
  --device cuda ^
  --fine-model-type fc ^
  --coarse-model-type fc ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 2 ^
  --eval-batch-size 4 ^
  --gradient-accumulation-steps 4 ^
  --fallback-to-flat ^
  --fine-checkpoint outputs_paper\deep_models\fc\best_fc.pt
```

跑 `BERT-RCNN` 的层次模型：

```bat
python scripts\train_deep_hierarchical.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_paper\deep_hierarchical_rcnn ^
  --device cuda ^
  --fine-model-type rcnn ^
  --coarse-model-type rcnn ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 1 ^
  --eval-batch-size 2 ^
  --gradient-accumulation-steps 8 ^
  --fallback-to-flat ^
  --fine-checkpoint outputs_paper\deep_models\rcnn\best_rcnn.pt
```

如果你想从原始数据开始重建 `110` 类处理数据：

```bat
python scripts\prepare_data.py ^
  --data-dir ..\2018数据集\2018数据集 ^
  --output-dir data\processed_110_paper ^
  --target-size 50000 ^
  --seed 42
```

如果处理后数据已经在 `data\processed_110_paper`，可以直接一键跑完整流程并跳过数据准备：

```bat
python scripts\run_pipeline.py ^
  --skip-prepare ^
  --processed-dir data\processed_110_paper ^
  --output-dir outputs_paper ^
  --device cuda ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 2 ^
  --eval-batch-size 4 ^
  --gradient-accumulation-steps 4 ^
  --fallback-to-flat
```

导出主结果表：

```bat
python scripts\make_results_table.py ^
  --output-dir outputs_paper ^
  --save-path outputs_paper\results_table.csv
```

单条推理：

```bat
python scripts\predict.py ^
  --artifact-path outputs_paper\deep_hierarchical_fc\model_bundle.json ^
  --device cuda ^
  --text "被告人张某深夜持刀抢劫路人手机和现金。"
```

批量推理：

```bat
python scripts\predict.py ^
  --artifact-path outputs_paper\deep_hierarchical_fc\model_bundle.json ^
  --device cuda ^
  --input-file demo.jsonl ^
  --output-file outputs_paper\predictions.csv
```

显存不够时，优先这样调：

- `BERT-FC`: 把 `--train-batch-size` 降到 `1`
- `BERT-RCNN`: 把 `--train-batch-size` 降到 `1`，`--eval-batch-size` 降到 `2`
- 如果后台有游戏、浏览器或桌面程序占显存，先关掉再训练

### 3.2 Windows 4060 Ti 优化版推荐入口

如果你要在 `RTX 4060 Ti 8GB` 上优先追求更高的 `macro F1`，推荐直接使用仓库根目录下的 `.cmd` 脚本，而不是手动拼接长命令。

先做平层 `fc` 的环境与小样本烟雾测试：

```bat
run_smoke_flat_fc.cmd
```

平层 `fc` 优化训练：

```bat
run_train_flat_fc_optimized.cmd
```

平层 `rcnn` 优化训练：

```bat
run_train_flat_rcnn_optimized.cmd
```

层次 `fc` 训练前的检查：

```bat
run_smoke_hier_fc.cmd
```

层次 `fc` 优化训练：

```bat
run_train_hier_fc_optimized.cmd
```

层次 `rcnn` 优化训练：

```bat
run_train_hier_rcnn_optimized.cmd
```

优化版脚本默认启用的核心策略：

- `optimize_profile = windows_4060ti_best`
- `loss = weighted_ce`
- `sampler = weighted`
- `label_smoothing = 0.05`
- `selection_metric = f1_macro`
- tokenizer cache 写入 `.cache/tokenized/`

优化版训练结束后，除了原有 `metrics.json` / `model_bundle.json` 外，还会额外产出：

- `per_class_metrics.csv`: 每个罪名类别的 precision / recall / F1
- `head_tail_metrics.json`: 头部/中部/尾部类别表现汇总
- `routing_diagnostics.json`: 层次模型的路由诊断结果

4060 Ti 8GB 的推荐顺序：

1. `run_train_flat_fc_optimized.cmd`
2. `run_train_hier_fc_optimized.cmd`
3. `run_train_flat_rcnn_optimized.cmd`
4. `run_train_hier_rcnn_optimized.cmd`

如果你只想要当前最稳的主模型，优先做完前两步即可。

## 4. 数据准备

仓库不直接包含原始 CAIL 数据。请把原始数据放到与仓库同级的目录：

```text
../2018数据集/2018数据集/data_train.json
../2018数据集/2018数据集/data_valid.json
../2018数据集/2018数据集/data_test.json
```

然后执行：

```bash
python scripts/prepare_data.py \
  --data-dir ../2018数据集/2018数据集 \
  --output-dir data/processed_110_paper \
  --target-size 50000 \
  --seed 42
```

默认会使用更稳的 `rebuild_stable` 策略：先在三大类内筛标签，再从全量样本池重建分层切分，避免验证集/测试集缺失大量标签。默认 `--min-label-support 3`，因此如果支持度不足，最终标签数可能少于 `110`。

如果你想保留旧版“按官方原始 split 过滤”的行为，可以显式指定：

```bash
python scripts/prepare_data.py \
  --data-dir ../2018数据集/2018数据集 \
  --output-dir data/processed_110_paper \
  --split-strategy official_filtered \
  --min-label-support 1 \
  --target-size 50000 \
  --seed 42
```

输出包括：
- `train_50k.jsonl`
- `valid_50k.jsonl`
- `test_50k.jsonl`
- `label2id.json`
- `coarse_label2id.json`
- `accusation_to_category.json`
- `dataset_stats.json`
- `analysis/processed_50k_table.csv`
- `analysis/fine_label_distribution.csv`
- `analysis/coarse_label_distribution.csv`
- `analysis/text_length_summary.csv`

## 5. 平层训练

```bash
python scripts/train_deep_models.py \
  --data-dir data/processed_110_paper \
  --output-dir outputs_paper/deep_models \
  --models fc rcnn \
  --device cuda \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --selection-metric accuracy
```

## 6. 层次训练

推荐每个骨干分别跑一次：

```bash
python scripts/train_deep_hierarchical.py \
  --data-dir data/processed_110_paper \
  --output-dir outputs_paper/deep_hierarchical_fc \
  --device cuda \
  --fine-model-type fc \
  --coarse-model-type fc \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4 \
  --fallback-to-flat
```

```bash
python scripts/train_deep_hierarchical.py \
  --data-dir data/processed_110_paper \
  --output-dir outputs_paper/deep_hierarchical_rcnn \
  --device cuda \
  --fine-model-type rcnn \
  --coarse-model-type rcnn \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4 \
  --fallback-to-flat
```

层次模型流程：
- 第 1 阶段：粗类三分类
- 第 2 阶段：按预测粗类进入对应细分类器
- 门控回退：如果粗类置信度或 margin 不足，则回退到平层预测，避免层次模型拖低基线

## 7. 法条推荐与一键跑全流程

先构建 183 类多标签法条数据：

```bash
python scripts/prepare_law_data.py \
  --data-dir ../2018数据集/2018数据集 \
  --output-dir data/processed_law \
  --top-k-articles 183
```

训练平层法条推荐模型：

```bash
python scripts/train_law_deep_models.py \
  --data-dir data/processed_law \
  --output-dir outputs_paper/law_deep \
  --models fc rcnn \
  --device cuda \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4
```

基于验证集划分易混淆法条并做层次融合：

```bash
python scripts/train_law_hierarchical.py \
  --law-deep-dir outputs_paper/law_deep \
  --output-dir outputs_paper/law_hierarchical \
  --models fc rcnn \
  --base-model fc
```

罪名预测 + 法条推荐完整流程可加 `--include-law`：

```bash
python scripts/run_pipeline.py \
  --data-dir ../2018数据集/2018数据集 \
  --processed-dir data/processed_110_paper \
  --law-processed-dir data/processed_law \
  --output-dir outputs_paper \
  --device cuda \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4 \
  --fallback-to-flat \
  --include-law
```

## 8. 结果表导出

```bash
python scripts/make_results_table.py \
  --output-dir outputs_paper \
  --save-path outputs_paper/results_table.csv
```

输出包括：
- `results_table.csv/.md`: 主结果表
- `results_table_contrast.csv/.md`: 同骨干 `flat vs hier` 增益表
- `results_table_intermediate.csv/.md`: 层次中间步骤指标表

主表指标列为 `accuracy`、`recall_macro`、`recall_micro`、`f1_score`。最终展示和 ROC/AUC 曲线：

```bash
python scripts/show_final_results.py \
  --output-dir outputs_paper \
  --export-dir outputs_paper/final_report
```

该命令会生成 `final_summary.md`、`requirements_check.md`、`auc_summary.csv` 和 `roc_auc/*.png`。

## 9. 推理

单条推理：

```bash
python scripts/predict.py \
  --artifact-path outputs_paper/deep_hierarchical_fc/model_bundle.json \
  --device cuda \
  --text "被告人张某深夜持刀抢劫路人手机和现金。"
```

批量推理：

```bash
python scripts/predict.py \
  --artifact-path outputs_paper/deep_hierarchical_fc/model_bundle.json \
  --device cuda \
  --input-file demo.jsonl \
  --output-file outputs_paper/predictions.csv
```

`demo.jsonl` 中每行可为：

```json
{"text": "被告人张某深夜持刀抢劫路人手机和现金。"}
```

## 10. GitHub 公开仓库说明

为避免公开仓库包含大体积数据和模型，以下内容默认不纳入版本控制：
- `data/`
- `outputs_paper/`
- 训练权重 `*.pt`
- 中间缓存和日志

仓库只保留代码、说明文档和轻量结果脚本。原始数据请按上面的目录自行准备。
