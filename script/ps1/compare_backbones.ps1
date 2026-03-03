# 快速对比测试：SD 1.5 vs SDXL
# 用于评估不同骨干网络的检测性能

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== LaRE Multi-Backbone Comparison Test ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will help you compare detection accuracy between:"
Write-Host "  - SD 1.5 (Original, 512x512)"
Write-Host "  - SDXL   (Upgraded, 1024x1024)"
Write-Host ""

# Prepare test images (user should provide)
$testDir = "web/test_images"
if (-not (Test-Path $testDir)) {
    New-Item -ItemType Directory -Path $testDir | Out-Null
    Write-Host "Created $testDir folder. Please add test images there." -ForegroundColor Yellow
    Write-Host "Recommended: Mix of SD1.5, SDXL, Midjourney, Real photos"
    Write-Host ""
    Read-Host "Press Enter after adding test images..."
}

$models = @("sd15", "sdxl")

foreach ($model in $models) {
    Write-Host ""
    Write-Host "Testing with $model backbone..." -ForegroundColor Green
    $env:LARE_MODEL_TYPE = $model
    
    Write-Host "Starting server... (Press Ctrl+C after testing)"
    python web/flask/app.py
    
    Write-Host ""
    Write-Host "Test completed for $model" -ForegroundColor Cyan
    Write-Host "Please note down your observations before continuing."
    Read-Host "Press Enter to test next model..."
}

Write-Host ""
Write-Host "All tests completed!" -ForegroundColor Green
