@echo off
echo ============================================
echo  Animals-10 Klasifikator - Pokretanje
echo ============================================
echo.
echo Proveri da li su .pt fajlovi u models\ folderu...
echo.

cd /d "%~dp0"


if not exist "models\" mkdir models
if not exist "results\" mkdir results

docker-compose up --build

pause
