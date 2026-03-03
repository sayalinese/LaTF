# 完整 SDXL 训练流程
# 在快速实验验证有效后执行

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Full SDXL Training Pipeline ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Prerequisites:"
Write-Host "  - Quick validation showed improvement"
Write-Host "  - ~12GB GPU VRAM available"
Write-Host "  - 4-6 hours training time"
Write-Host ""

$confirm = Read-Host "Continue with full training? (y/n)"
if ($confirm -ne "y") { exit }

# 配置
$TRAIN_ANNO = "annotation/train_sdv5.txt"
$VAL_ANNO = "annotation/val_sdv5.txt"
$TRAIN_FEAT = "features/sdv5_train_sdxl"
$VAL_FEAT = "features/sdv5_val_sdxl"
$OUTPUT = "模型训练/output/Expsdv5_wmap_sdxl_full"

Write-Host ""
Write-Host "[Phase 1/2] Feature Extraction" -ForegroundColor Cyan
Write-Host "============================================"

# 训练集特征
if (-not (Test-Path $TRAIN_FEAT)) {
    Write-Host "Extracting training features (~2-3 hours)..." -ForegroundColor Yellow
    python 模型训练/2_特征提取.py `
        --annotation_file $TRAIN_ANNO `
        --output_path $TRAIN_FEAT `
        --model_type sdxl `
        --extract_batch_size 4 `
        --extract_workers 4
} else {
    Write-Host "Training features already exist, skipping..." -ForegroundColor Green
}

# 验证集特征
if (-not (Test-Path $VAL_FEAT)) {
    Write-Host "Extracting validation features (~20 min)..." -ForegroundColor Yellow
    python 模型训练/2_特征提取.py `
        --annotation_file $VAL_ANNO `
        --output_path $VAL_FEAT `
        --model_type sdxl `
        --extract_batch_size 4 `
        --extract_workers 4
} else {
    Write-Host "Validation features already exist, skipping..." -ForegroundColor Green
}

Write-Host ""
Write-Host "[Phase 2/2] Model Training" -ForegroundColor Cyan
Write-Host "============================================"
Write-Host "Training with SDXL features..." -ForegroundColor Yellow

python 模型训练/3_模型训练.py `
    --train_feature_path $TRAIN_FEAT `
    --val_feature_path $VAL_FEAT `
    --annotation_train $TRAIN_ANNO `
    --annotation_val $VAL_ANNO `
    --output_dir $OUTPUT `
    --batch_size 32 `
    --epochs 20 `
    --lr 0.0001 `
    --clip_type RN50x64 `
    --num_classes 2

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Training Completed!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""
Write-Host "Best model: $OUTPUT/Val_best.pth"
Write-Host ""
Write-Host "To use in web demo:"
Write-Host "  1. Update app.py checkpoint path to: $OUTPUT"
Write-Host "  2. Set LARE_MODEL_TYPE=sdxl in web/.env"
Write-Host "  3. Restart: python web/flask/app.py"
