@echo off
chcp 65001 > nul
title 拼图还原 · 局域网网页版
cd /d "%~dp0.."
python web\server.py --port 8000
echo.
echo 服务器已停止，按任意键关闭窗口
pause > nul
