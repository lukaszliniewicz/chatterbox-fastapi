@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PARENT_DIR=%PROJECT_DIR%..\"
for %%I in ("%PARENT_DIR%") do set "PARENT_DIR=%%~fI"

set "PIXI_EXE=%PARENT_DIR%\bin\pixi.exe"
set "CUSTOM_PIXI=0"
set "PASS_ARGS="
set "BACKEND_MODE=cuda"

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--pixi-path" (
    if "%~2"=="" (
        echo Missing value for --pixi-path.
        exit /b 1
    )
    for %%I in ("%~2") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    shift
    goto parse_args
)

set "ARG1=%~1"
if /I "!ARG1:~0,12!"=="--pixi-path=" (
    set "PIXI_VALUE=!ARG1:~12!"
    for %%I in ("!PIXI_VALUE!") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    goto parse_args
)

if /I "%~1"=="--backend" (
    if /I "%~2"=="cpu" set "BACKEND_MODE=cpu"
    if /I "%~2"=="cuda" set "BACKEND_MODE=cuda"
    shift
    shift
    goto parse_args
)

if /I "%~1"=="--backend=cpu" (
    set "BACKEND_MODE=cpu"
    shift
    goto parse_args
)

if /I "%~1"=="--backend=cuda" (
    set "BACKEND_MODE=cuda"
    shift
    goto parse_args
)

if /I "%~1"=="cpu" (
    set "BACKEND_MODE=cpu"
    shift
    goto parse_args
)

if /I "%~1"=="cuda" (
    set "BACKEND_MODE=cuda"
    shift
    goto parse_args
)

if /I "%~1"=="gpu" (
    set "BACKEND_MODE=cuda"
    shift
    goto parse_args
)

:: Forward other args
set "PASS_ARGS=!PASS_ARGS! %1"
shift
goto parse_args

:args_done

if "%CUSTOM_PIXI%"=="1" (
    if not exist "%PIXI_EXE%" (
        echo Provided --pixi-path does not exist: "%PIXI_EXE%"
        exit /b 1
    )
)

:: Download pixi if missing (skip when --pixi-path is provided)
if "%CUSTOM_PIXI%"=="0" if not exist "%PIXI_EXE%" (
    echo Downloading pixi...
    if not exist "%PARENT_DIR%\bin" mkdir "%PARENT_DIR%\bin"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/download/v0.68.1/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%PIXI_EXE%'"
    if errorlevel 1 (
        echo Failed to download pixi.
        exit /b 1
    )
)

:: Set unified portable models, caches, and tmp directories
set "PIXI_CACHE_DIR=%PARENT_DIR%\.pixi-cache"
set "RATTLER_CACHE_DIR=%PIXI_CACHE_DIR%\rattler"
set "PIP_CACHE_DIR=%PARENT_DIR%\.pip-cache"
set "TMP=%PARENT_DIR%\.tmp"
set "TEMP=%PARENT_DIR%\.tmp"

if not exist "%PIXI_CACHE_DIR%" mkdir "%PIXI_CACHE_DIR%"
if not exist "%RATTLER_CACHE_DIR%" mkdir "%RATTLER_CACHE_DIR%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

:: Ensure default environment is initialized via pixi
if not exist "%PROJECT_DIR%\.pixi\envs\default\python.exe" (
    echo Initializing pixi environment...
    cd /d "%PROJECT_DIR%"
    "%PIXI_EXE%" install
    if errorlevel 1 (
        echo pixi install failed.
        exit /b 1
    )
)

:: Check if nvidia-smi exists for GPU detection
if "%BACKEND_MODE%"=="cuda" (
    where nvidia-smi >nul 2>nul
    if !ERRORLEVEL! neq 0 (
        echo nvidia-smi not detected. Defaulting to CPU mode.
        set "BACKEND_MODE=cpu"
    )
)

:: Verify PyTorch CUDA support in Pixi env
set "NEED_TORCH_INSTALL=0"
if "%BACKEND_MODE%"=="cuda" (
    "%PIXI_EXE%" run python -c "import torch; exit(0 if torch.cuda.is_available() and '2.8.0' in torch.__version__ else 1)" >nul 2>nul
    if !ERRORLEVEL! neq 0 set "NEED_TORCH_INSTALL=1"
) else (
    "%PIXI_EXE%" run python -c "import torch; exit(0 if '2.8.0' in torch.__version__ else 1)" >nul 2>nul
    if !ERRORLEVEL! neq 0 set "NEED_TORCH_INSTALL=1"
)

if "%NEED_TORCH_INSTALL%"=="1" (
    if "%BACKEND_MODE%"=="cuda" (
        echo Installing PyTorch 2.8.0 CUDA 12.8 in Pixi env...
        "%PIXI_EXE%" run pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    ) else (
        echo Installing PyTorch 2.8.0 CPU-only in Pixi env...
        "%PIXI_EXE%" run pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    )
    if errorlevel 1 (
        echo PyTorch installation failed.
        exit /b 1
    )
) else (
    echo PyTorch is already up to date ^(version 2.8.0 CUDA=%BACKEND_MODE%^).
)

:: Verify Chatterbox-TTS package installation in Pixi env
"%PIXI_EXE%" run python -c "import chatterbox" >nul 2>nul
if !ERRORLEVEL! neq 0 (
    echo Chatterbox package not found. Preparing chatterbox source...
    if not exist chatterbox_src (
        echo Cloning resemble-ai/chatterbox...
        git clone https://github.com/resemble-ai/chatterbox.git chatterbox_src
        if errorlevel 1 (
            echo Failed to clone Chatterbox repository.
            exit /b 1
        )
    )
    echo Installing Chatterbox wrapper package...
    cd /d "%PROJECT_DIR%\chraph_src" 2>nul || cd /d "%PROJECT_DIR%\chatterbox_src"
    "%PIXI_EXE%" run pip install --no-dependencies .
    if errorlevel 1 (
        echo Failed to install Chatterbox package.
        exit /b 1
    )
    cd /d "%PROJECT_DIR%"
)

:: Start server with passed parameters
echo Starting Chatterbox API Server...
"%PIXI_EXE%" run python run.py --backend !BACKEND_MODE! !PASS_ARGS!
pause
