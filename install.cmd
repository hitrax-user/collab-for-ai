@echo off
rem Colab LLM setup: double-click me. Checks and installs WSL 2 + Ubuntu, then the project tools.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if %errorlevel% equ 1 pause
