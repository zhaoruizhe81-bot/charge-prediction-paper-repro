@echo off
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet || exit /b 1
cd /d "%~dp0" || exit /b 1
set "BERT_DIR=%cd%\chinese-bert-wwm-ext"
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
  --fine-checkpoint outputs_paper\deep_models\fc\best_fc.pt ^
  --optimize-profile windows_4060ti_best
