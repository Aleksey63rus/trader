@echo off
chcp 65001 > nul
title INVESTOR Terminal — API Server :8002

echo.
echo  ╔══════════════════════════════════════╗
echo  ║      INVESTOR Terminal  v1.0         ║
echo  ║      API Server — порт 8002          ║
echo  ╚══════════════════════════════════════╝
echo.

:: Освобождаем порт 8002
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8002 " ^| findstr LISTENING') do (
    echo  Останавливаю старый процесс PID %%a...
    taskkill /F /PID %%a > nul 2>&1
    timeout /t 2 /nobreak > nul
)

echo  Запуск API сервера на порту 8002...
echo  Дашборд: file:///c:/investor/trader/frontend/index.html
echo  API Docs: http://localhost:8002/docs
echo  Данные:   c:\investor\data\ (151 CSV файлов)
echo.

cd /d c:\investor\trader
python -m uvicorn api.main:app --host 0.0.0.0 --port 8002

pause
