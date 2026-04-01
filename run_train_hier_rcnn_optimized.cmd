@echo off
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet || exit /b 1
cd /d "%~dp0" || exit /b 1
set "BERT_DIR=%cd%\chinese-bert-wwm-ext"
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
  --fine-checkpoint outputs_paper\deep_models\rcnn\best_rcnn.pt ^
  --optimize-profile windows_4060ti_best
