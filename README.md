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
- 细类固定为 `110` 个
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

## 7. 一键跑全流程

```bash
python scripts/run_pipeline.py \
  --data-dir ../2018数据集/2018数据集 \
  --processed-dir data/processed_110_paper \
  --output-dir outputs_paper \
  --device cuda \
  --pretrained-model hfl/chinese-bert-wwm-ext \
  --epochs 4 \
  --fallback-to-flat
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
- `results_table_intermediate.csv/.md`: 层次中间步骤准确率表

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
