@echo off
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet || exit /b 1
cd /d "%~dp0" || exit /b 1
python scripts\smoke_test_flat.py || exit /b 1
set "BERT_DIR=%cd%\chinese-bert-wwm-ext"
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
  --max-train-samples 32 ^
  --max-valid-samples 16 ^
  --max-test-samples 16 ^
  --optimize-profile windows_4060ti_best
