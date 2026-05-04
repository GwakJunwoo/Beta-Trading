@echo off
REM Beta Trading 4-category daily snapshot (scheduled task wrapper)
REM Runs at 09:00. Output -> data\factor_trading\daily_snapshots_all\
cd /d "C:\Users\infomax\Beta Trading"

set LOGDIR=data\factor_trading\daily_snapshots_all
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set LOGFILE=%LOGDIR%\_run.log

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo [%date% %time%] daily_snapshot_all start >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

set PYTHONIOENCODING=utf-8
"C:\Users\infomax\anaconda3\python.exe" "factor_trading\scripts\daily_snapshot_all.py" 1>> "%LOGFILE%" 2>&1
set RC=%ERRORLEVEL%

echo. >> "%LOGFILE%"
echo [%date% %time%] daily_snapshot_all done (exit=%RC%) >> "%LOGFILE%"
exit /b %RC%
