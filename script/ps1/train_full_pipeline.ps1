# ============================================================
#  LaRE 全流程训练脚本 (分类 + 定位 双管道)
#  用法: powershell -ExecutionPolicy Bypass -File script\ps1\train_full_pipeline.ps1
#  注意: 执行前确认 .env 中的采样参数已调好
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
Write-Host "`n====== LaRE Full Training Pipeline ======" -ForegroundColor Cyan
Write-Host "工作目录: $(Get-Location)" -ForegroundColor Gray
Write-Host "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# ─── Phase 0: 环境检查 ─────────────────────────────────────
Write-Host "`n[Phase 0] 环境检查..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
if ($LASTEXITCODE -ne 0) { Write-Host "环境检查失败!" -ForegroundColor Red; exit 1 }

# ─── Phase 1: 生成标注文件 ───────────────────────────────────
Write-Host "`n[Phase 1] 生成标注文件..." -ForegroundColor Yellow

Write-Host "  1a. 分类标注 (train_v2 / val_v2 / test_v2)..."
python script/gen_annotations_v2.py
if ($LASTEXITCODE -ne 0) { Write-Host "分类标注生成失败!" -ForegroundColor Red; exit 1 }

Write-Host "  1b. 定位标注 (train_seg / val_seg)..."
python script/gen_mask_annotations.py
if ($LASTEXITCODE -ne 0) { Write-Host "定位标注生成失败!" -ForegroundColor Red; exit 1 }

# ─── Phase 2: 构建特征提取列表 + 提取 SSFR 特征 ──────────────
Write-Host "`n[Phase 2] 特征提取..." -ForegroundColor Yellow

Write-Host "  2a. 合并去重提取列表..."
python script/3_build_extract_list.py `
    --train annotation/train_v2.txt `
    --val   annotation/val_v2.txt `
    --out   annotation/extract_v2.txt
if ($LASTEXITCODE -ne 0) { Write-Host "提取列表构建失败!" -ForegroundColor Red; exit 1 }

# 也把 test_v2 和定位标注加入提取列表 (确保所有样本都有特征)
python script/3_build_extract_list.py `
    --train annotation/extract_v2.txt `
    --val   annotation/test_v2.txt `
    --out   annotation/extract_v2.txt
if ($LASTEXITCODE -ne 0) { Write-Host "提取列表合并test失败!" -ForegroundColor Red; exit 1 }

python script/3_build_extract_list.py `
    --train annotation/extract_v2.txt `
    --val   annotation/train_seg.txt `
    --out   annotation/extract_v2.txt
if ($LASTEXITCODE -ne 0) { Write-Host "提取列表合并seg失败!" -ForegroundColor Red; exit 1 }

python script/3_build_extract_list.py `
    --train annotation/extract_v2.txt `
    --val   annotation/val_seg.txt `
    --out   annotation/extract_v2.txt
if ($LASTEXITCODE -ne 0) { Write-Host "提取列表合并seg_val失败!" -ForegroundColor Red; exit 1 }

Write-Host "  2b. 提取 SSFR 特征 (7ch, 32x32)..."
python script/2_extract_features.py `
    --input_path  annotation/extract_v2.txt `
    --output_path dift.pt `
    --extractor_type ssfr `
    --bf16
if ($LASTEXITCODE -ne 0) { Write-Host "SSFR 特征提取失败!" -ForegroundColor Red; exit 1 }

# ─── Phase 3: 训练分类模型 ──────────────────────────────────
Write-Host "`n[Phase 3] 训练分类模型 (LaREDeepFakeV11)..." -ForegroundColor Yellow
python script/5_train_model_v11.py
if ($LASTEXITCODE -ne 0) { Write-Host "分类训练失败!" -ForegroundColor Red; exit 1 }

# ─── Phase 4: 训练定位模型 (SegFormer) ──────────────────────
Write-Host "`n[Phase 4] 训练定位模型 (SegFormer-B2)..." -ForegroundColor Yellow

Write-Host "  4a. RGB 3ch 基线..."
python script/train_segformer.py `
    --train_file annotation/train_seg.txt `
    --val_file   annotation/val_seg.txt `
    --out_dir    outputs/segformer_rgb `
    --batch_size 12 `
    --epochs     40 `
    --patience   10
if ($LASTEXITCODE -ne 0) { Write-Host "SegFormer RGB 训练失败!" -ForegroundColor Red; exit 1 }

# ─── Phase 5: 评估 ──────────────────────────────────────────
Write-Host "`n[Phase 5] 模型评估..." -ForegroundColor Yellow

Write-Host "  5a. 分类测试集评估..."
python script/4_test_model.py `
    --dir  data `
    --model outputs/v14_multiscale/best.pth
if ($LASTEXITCODE -ne 0) { Write-Host "分类评估失败 (非致命)" -ForegroundColor DarkYellow }

Write-Host "  5b. 定位验证集评估..."
python test/evaluate_segformer.py `
    --model    outputs/segformer_rgb/best.pth `
    --ann_file annotation/val_seg.txt `
    --batch_size 12
if ($LASTEXITCODE -ne 0) { Write-Host "定位评估失败 (非致命)" -ForegroundColor DarkYellow }

Write-Host "  5c. 定位独立评估集..."
python test/evaluate_segformer.py `
    --model    outputs/segformer_rgb/best.pth `
    --ann_file annotation/eval_change.txt `
    --batch_size 12
if ($LASTEXITCODE -ne 0) { Write-Host "定位独立评估失败 (非致命)" -ForegroundColor DarkYellow }

# ─── 完成 ────────────────────────────────────────────────────
Write-Host "`n====== 全流程完成 ======" -ForegroundColor Green
Write-Host "分类模型: outputs/v14_multiscale/best.pth"
Write-Host "定位模型: outputs/segformer_rgb/best.pth"
Write-Host "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
