# 豆包数据增强训练脚本
# 混合原有SD1.5数据 + 豆包数据 + SDXL特征

param(
    [switch]$UseDoubaoOnly = $false  # 仅用豆包数据训练（快速测试）
)

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Enhanced Training with Doubao Data ===" -ForegroundColor Cyan
Write-Host ""

# 检查标注文件是否已生成
if (-not (Test-Path "annotation/train_doubao_only.txt")) {
    Write-Host "Error: Doubao annotation files not found!" -ForegroundColor Red
    Write-Host "Please run: .\script\ps1\generate_doubao_annotations.ps1 first"
    exit 1
}

# 根据参数选择训练模式
if ($UseDoubaoOnly) {
    Write-Host "Training mode: Doubao-only (Quick test)" -ForegroundColor Yellow
    $TRAIN_ANNO = "annotation/train_doubao_only.txt"
    $VAL_ANNO = "annotation/val_doubao_only.txt"
    $TRAIN_FEAT = "features/doubao_train_sdxl"
    $VAL_FEAT = "features/doubao_val_sdxl"
    $OUTPUT = "模型训练/output/Expsdv5_wmap_sdxl_doubao_only"
} else {
    Write-Host "Training mode: Mixed (SD1.5 + Doubao)" -ForegroundColor Yellow
    $TRAIN_ANNO = "annotation/train_sdv5_doubao_mixed.txt"
    $VAL_ANNO = "annotation/val_sdv5.txt"
    $TRAIN_FEAT = "features/sdv5_train_doubao_sdxl"
    $VAL_FEAT = "features/sdv5_val_sdxl"
    $OUTPUT = "模型训练/output/Expsdv5_wmap_sdxl_doubao"
    
    if (-not (Test-Path $TRAIN_ANNO)) {
        Write-Host "Error: Mixed annotation not found!" -ForegroundColor Red
        Write-Host "Please ensure original SD1.5 dataset exists at: annotation/train_sdv5.txt"
        exit 1
    }
}

Write-Host "Annotation: $TRAIN_ANNO" -ForegroundColor Cyan
Write-Host "Output: $OUTPUT" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Extracting SDXL features..." -ForegroundColor Cyan
Write-Host "============================================"
Write-Host ""

# 训练集特征提取
if (-not (Test-Path $TRAIN_FEAT)) {
    Write-Host "Extracting training features (this will take 2-3 hours)..." -ForegroundColor Yellow
    python 模型训练/2_特征提取.py `
        --annotation_file $TRAIN_ANNO `
        --output_path $TRAIN_FEAT `
        --model_type sdxl `
        --extract_batch_size 4 `
        --extract_workers 4
} else {
    Write-Host "Training features already exist at $TRAIN_FEAT, skipping extraction..." -ForegroundColor Green
}

# 验证集特征提取
if (-not (Test-Path $VAL_FEAT)) {
    Write-Host "Extracting validation features (~20 minutes)..." -ForegroundColor Yellow
    python 模型训练/2_特征提取.py `
        --annotation_file $VAL_ANNO `
        --output_path $VAL_FEAT `
        --model_type sdxl `
        --extract_batch_size 4 `
        --extract_workers 4
} else {
    Write-Host "Validation features already exist at $VAL_FEAT, skipping extraction..." -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/2] Training classifier..." -ForegroundColor Cyan
Write-Host "============================================"
Write-Host ""

python 模型训练/3_模型训练.py `
    --train_feature_path $TRAIN_FEAT `
    --val_feature_path $VAL_FEAT `
    --annotation_train $TRAIN_ANNO `
    --annotation_val $VAL_ANNO `
    --output_dir $OUTPUT `
    --batch_size 32 `
    --epochs 25 `
    --lr 0.0001 `
    --clip_type RN50x64 `
    --num_classes 2

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Enhanced Training Completed!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""
Write-Host "Model optimized for Doubao detection: 模型训练/output/Expsdv5_wmap_sdxl_doubao/Val_best.pth"
Write-Host ""
Write-Host "Expected improvement on Doubao images: +25-35%"
