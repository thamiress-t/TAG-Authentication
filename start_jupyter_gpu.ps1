# start_jupyter_gpu.ps1
# PowerShell script to start Jupyter with TensorFlow GPU support in WSL
#
# Usage (from Windows PowerShell):
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#   .\start_jupyter_gpu.ps1
#

Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Jupyter with TensorFlow GPU Support (WSL)            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Path to project
$projectPath = "C:\Users\thami\OneDrive\Documents\TAG-Authentication"
$venvPath = "\\wsl$\Ubuntu\home\thami\venv_tf_gpu"

if (-not (Test-Path $venvPath)) {
    Write-Host "❌ Error: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "   Note: WSL Ubuntu must be running" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Found venv at: $venvPath" -ForegroundColor Green
Write-Host ""

# Check if wsl command is available
if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: WSL not found. Make sure Windows Subsystem for Linux is installed." -ForegroundColor Red
    exit 1
}

Write-Host "🔄 Activating virtual environment and launching Jupyter..." -ForegroundColor Yellow
Write-Host ""

# Ask which Jupyter to use
Write-Host "Choose Jupyter interface:" -ForegroundColor Cyan
Write-Host "  1) Jupyter Lab (recommended, modern)"
Write-Host "  2) Jupyter Notebook (classic)"
Write-Host ""

$choice = Read-Host "Enter choice (1 or 2)"

$command = ""
switch ($choice) {
    "1" {
        $command = "source /home/thami/venv_tf_gpu/bin/activate && jupyter lab"
        Write-Host "🚀 Starting Jupyter Lab..." -ForegroundColor Yellow
    }
    "2" {
        $command = "source /home/thami/venv_tf_gpu/bin/activate && jupyter notebook"
        Write-Host "🚀 Starting Jupyter Notebook..." -ForegroundColor Yellow
    }
    default {
        Write-Host "❌ Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Change to project directory and run in WSL
cd "$projectPath"
wsl -d Ubuntu bash -c "cd /mnt/c/Users/thami/OneDrive/Documents/TAG-Authentication && $command"
