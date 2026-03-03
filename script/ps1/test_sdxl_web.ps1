# LaRE SDXL 升级测试脚本
$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== LaRE SDXL Backbone Test ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will test the upgraded SDXL LaRE extractor on your web demo."
Write-Host "First launch will download SDXL models (~6GB), please be patient..."
Write-Host ""

# Set environment variable for SDXL
$env:LARE_MODEL_TYPE = "sdxl"

Write-Host "[1/2] Starting Flask server with SDXL backbone..." -ForegroundColor Yellow
Write-Host "Please wait for model loading (may take 2-3 minutes first time)..."
Write-Host ""

# Start server
python web/flask/app.py
