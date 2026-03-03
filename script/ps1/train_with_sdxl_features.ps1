# SDXL 特征训练配置
# 基于原有训练脚本，使用 SDXL 提取的特征

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== LaRE Training with SDXL Features ===" -ForegroundColor Cyan

# 配置
$TRAIN_FEATURES = "features/sdv5_train_sdxl"  # SDXL 提取的特征
$VAL_FEATURES = "features/sdv5_val_sdxl"
$OUTPUT_DIR = "模型训练/output/Expsdv5_wmap_v7_sdxl"
$BATCH_SIZE = 32
$EPOCHS = 20
$LR = 0.0001

Write-Host "Train Features: $TRAIN_FEATURES"
Write-Host "Val Features: $VAL_FEATURES"
Write-Host "Output: $OUTPUT_DIR"
Write-Host "Epochs: $EPOCHS"
Write-Host ""

# 检查特征是否存在
if (-not (Test-Path $TRAIN_FEATURES)) {
    Write-Host "Error: SDXL features not found at $TRAIN_FEATURES" -ForegroundColor Red
    Write-Host "Please run: .\script\ps1\extract_features_sdxl.ps1 first" 
    exit 1
}

Write-Host "Starting training..." -ForegroundColor Green
Write-Host ""

# 运行训练
python 模型训练/3_模型训练.py `
    --train_feature_path $TRAIN_FEATURES `
    --val_feature_path $VAL_FEATURES `
    --output_dir $OUTPUT_DIR `
    --batch_size $BATCH_SIZE `
    --epochs $EPOCHS `
    --lr $LR `
    --clip_type RN50x64 `
    --num_classes 2

Write-Host ""
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "Best model saved at: $OUTPUT_DIR/Val_best.pth"
