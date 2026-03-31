@echo off
cd /d "d:\Mestrado\Simulações\TAG-Authentication"
echo.
echo ============================================================
echo Ativando Virtual Environment e Instalando Dependencies
echo ============================================================
echo.

REM Ativar venv
call venv_nn\Scripts\activate.bat

REM Instalar requirements
echo Instalando packages... Isso pode levar 10-15 minutos
py -m pip install -r requirements.txt

echo.
echo ============================================================
echo Instalacao Concluida!
echo ============================================================
echo.
pause
