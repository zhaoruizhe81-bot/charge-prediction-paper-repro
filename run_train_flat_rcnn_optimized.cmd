@echo off
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet || exit /b 1
cd /d "%~dp0" || exit /b 1
set "BERT_DIR=%cd%\chinese-bert-wwm-ext"
python scripts\train_deep_models.py ^
  --data-dir data\processed_110_paper ^
  --output-dir outputs_paper\deep_models ^
  --models rcnn ^
  --device cuda ^
  --pretrained-model "%BERT_DIR%" ^
  --epochs 4 ^
  --train-batch-size 1 ^
  --eval-batch-size 2 ^
  --gradient-accumulation-steps 8 ^
  --optimize-profile windows_4060ti_best
