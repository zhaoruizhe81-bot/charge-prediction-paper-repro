@echo off
call D:\software\miniconda\condabin\conda.bat activate D:\conda_envs\caoyao-resnet || exit /b 1
cd /d "%~dp0" || exit /b 1
python scripts\smoke_test_hier_fc.py
