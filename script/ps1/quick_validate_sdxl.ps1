# 快速验证 SDXL 效果
# 只提取一小部分数据的特征，快速训练看效果

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Quick SDXL Validation Experiment ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will:"
Write-Host "  1. Extract 500 train + 100 val samples with SDXL"
Write-Host "  2. Train a small model (5 epochs)"
Write-Host "  3. Compare with SD1.5 baseline"
Write-Host ""
Write-Host "Time: ~30 minutes"
Write-Host ""

$confirm = Read-Host "Start experiment? (y/n)"
if ($confirm -ne "y") { exit }

# 创建小型数据集
Write-Host "[1/4] Creating small dataset..." -ForegroundColor Yellow
$trainLines = Get-Content "annotation/train_sdv5.txt" | Select-Object -First 500
$valLines = Get-Content "annotation/val_sdv5.txt" | Select-Object -First 100

$trainLines | Out-File "annotation/train_sdxl_quick.txt" -Encoding UTF8
$valLines | Out-File "annotation/val_sdxl_quick.txt" -Encoding UTF8

# 提取 SDXL 特征
Write-Host "[2/4] Extracting SDXL features (this may take 20-30 min)..." -ForegroundColor Yellow
python 模型训练/2_特征提取.py `
    --annotation_file "annotation/train_sdxl_quick.txt" `
    --output_path "features/quick_train_sdxl" `
    --model_type sdxl `
    --extract_batch_size 8

python 模型训练/2_特征提取.py `
    --annotation_file "annotation/val_sdxl_quick.txt" `
    --output_path "features/quick_val_sdxl" `
    --model_type sdxl `
    --extract_batch_size 8

# 训练
Write-Host "[3/4] Training quick model..." -ForegroundColor Yellow
python 模型训练/3_模型训练.py `
    --train_feature_path "features/quick_train_sdxl" `
    --val_feature_path "features/quick_val_sdxl" `
    --annotation_train "annotation/train_sdxl_quick.txt" `
    --annotation_val "annotation/val_sdxl_quick.txt" `
    --output_dir "模型训练/output/quick_sdxl_test" `
    --batch_size 32 `
    --epochs 5 `
    --clip_type RN50x64

# 对比测试
Write-Host "[4/4] Comparing with SD1.5 baseline..." -ForegroundColor Yellow
Write-Host ""
Write-Host "SDXL Model: 模型训练/output/quick_sdxl_test/Val_best.pth"
Write-Host "SD1.5 Model: 模型训练/output/Expsdv5_wmap_v7_Log_v01221725/Val_best.pth"
Write-Host ""
Write-Host "Next: Test both models on new AI images and compare accuracy."

Write-Host ""
Write-Host "Experiment completed!" -ForegroundColor Green
