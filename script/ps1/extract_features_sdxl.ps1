# 用 SDXL 重新提取训练集特征
# 这会生成新的特征文件用于重新训练

param(
    [string]$DataSplit = "train",  # train, val, test
    [string]$Annotation = "annotation/train_sdv5.txt"
)

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== SDXL Feature Re-extraction ===" -ForegroundColor Cyan
Write-Host "Target: $DataSplit"
Write-Host "Annotation: $Annotation"
Write-Host ""

# 设置输出目录（避免覆盖原有 SD1.5 特征）
$outputDir = "features/sdv5_${DataSplit}_sdxl"
Write-Host "Output: $outputDir" -ForegroundColor Yellow
Write-Host ""

# 确认操作
Write-Host "This will extract features using SDXL backbone (1024x1024)." -ForegroundColor Yellow
Write-Host "Estimated time: ~1-2 hours for 3000 images (depends on GPU)." -ForegroundColor Yellow
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne "y") { exit }

# 运行特征提取（2_特征提取.py 支持 SDXL）
python 模型训练/2_特征提取.py `
    --annotation_file $Annotation `
    --output_path $outputDir `
    --model_type sdxl `
    --extract_batch_size 4 `
    --extract_workers 4

Write-Host ""
Write-Host "Feature extraction completed!" -ForegroundColor Green
Write-Host "Next: Update training script to use $outputDir"
