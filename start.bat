@echo off
python src\main\main.py %* > log/output.txt
pause
call "clear-cache.sh"