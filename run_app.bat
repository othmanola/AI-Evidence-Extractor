@echo off
setlocal
set "AEE_APP_DIR=%~dp0"
set "AEE_DATA_DIR=%LOCALAPPDATA%\AI Evidence Extractor"
if not exist "%AEE_DATA_DIR%" mkdir "%AEE_DATA_DIR%"
set "AEE_LOG=%AEE_DATA_DIR%\startup.log"
set "AEE_RUNTIME_DIR=%AEE_APP_DIR%runtime"
set "AEE_PYTHON=%AEE_RUNTIME_DIR%\python.exe"
set "AEE_VENV=%AEE_DATA_DIR%\runtimeenv"
set "AEE_VENV_PYTHON=%AEE_VENV%\Scripts\python.exe"
set "AEE_VENV_PYW=%AEE_VENV%\Scripts\pythonw.exe"
set "AEE_SETUP_MARKER=%AEE_DATA_DIR%\setup_complete.txt"
echo ==== AI Evidence Extractor launch %DATE% %TIME% ====>>"%AEE_LOG%"

if not exist "%AEE_PYTHON%" (
  echo Private Python runtime is missing. Reinstall the app using Setup.exe.
  echo Runtime missing at %AEE_PYTHON%>>"%AEE_LOG%"
  pause
  exit /b 1
)
pushd "%AEE_RUNTIME_DIR%" || goto :error

findstr /x /c:"desktop-v1" "%AEE_SETUP_MARKER%" >nul 2>nul
if errorlevel 1 (
  echo Preparing the app environment. This can take several minutes and requires internet access.
  if not exist "%AEE_VENV_PYTHON%" "%AEE_PYTHON%" -m venv "%AEE_VENV%" >>"%AEE_LOG%" 2>&1 || goto :error
  "%AEE_VENV_PYTHON%" -m pip install --upgrade pip >>"%AEE_LOG%" 2>&1 || goto :error
  "%AEE_VENV_PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu >>"%AEE_LOG%" 2>&1 || goto :error
  "%AEE_VENV_PYTHON%" -m pip install -r "%AEE_APP_DIR%requirements.txt" >>"%AEE_LOG%" 2>&1 || goto :error
  "%AEE_VENV_PYTHON%" -u "%AEE_APP_DIR%download_models.py" >>"%AEE_LOG%" 2>&1 || goto :error
  >"%AEE_SETUP_MARKER%" echo desktop-v1
)

echo Starting app... (log: %AEE_LOG%)
if not exist "%AEE_VENV_PYW%" goto :error
"%AEE_VENV_PYW%" "%AEE_APP_DIR%desktop_launcher.py"
if errorlevel 1 goto :error
popd
exit /b 0

:error
echo.
echo Startup failed. Details are in:
echo %AEE_LOG%
echo.
type "%AEE_LOG%"
pause
exit /b 1
