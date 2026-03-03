# 生成豆包数据集的注释文件
# 扫描目录并创建训练/验证集标注

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Generate Doubao Annotations ===" -ForegroundColor Cyan
Write-Host ""

$baseDir = "GenImage_Dataset"
$annoDir = "annotation"

# 确保注释目录存在
if (-not (Test-Path $annoDir)) {
    New-Item -ItemType Directory -Path $annoDir -Force | Out-Null
}

# 收集所有图片路径和标签
$allImages = @()

# AI生成图片（标签=1）
$aiDirs = @("doubao_ai", "midjourney", "dalle3", "flux")
foreach ($dir in $aiDirs) {
    $fullPath = Join-Path $baseDir $dir
    if (Test-Path $fullPath) {
        Get-ChildItem $fullPath -Include *.jpg,*.jpeg,*.png,*.webp -Recurse | ForEach-Object {
            $allImages += [PSCustomObject]@{
                Path = $_.FullName -replace '\\', '/'
                Label = 1
                Source = $dir
            }
        }
    }
}

# 真实照片（标签=0）
$realDir = Join-Path $baseDir "doubao_real"
if (Test-Path $realDir) {
    Get-ChildItem $realDir -Include *.jpg,*.jpeg,*.png,*.webp -Recurse | ForEach-Object {
        $allImages += [PSCustomObject]@{
            Path = $_.FullName -replace '\\', '/'
            Label = 0
            Source = "real"
        }
    }
}

Write-Host "Found images:" -ForegroundColor Yellow
$grouped = $allImages | Group-Object Source
foreach ($group in $grouped) {
    Write-Host "  $($group.Name): $($group.Count) images" -ForegroundColor Green
}
Write-Host "  Total: $($allImages.Count) images" -ForegroundColor Cyan
Write-Host ""

if ($allImages.Count -eq 0) {
    Write-Host "Error: No images found in $baseDir" -ForegroundColor Red
    Write-Host "Please run .\script\ps1\setup_doubao_dataset.ps1 first and add images."
    exit 1
}

# 随机打乱
$allImages = $allImages | Get-Random -Count $allImages.Count

# 划分训练集和验证集（80/20）
$splitIdx = [math]::Floor($allImages.Count * 0.8)
$trainImages = $allImages[0..($splitIdx-1)]
$valImages = $allImages[$splitIdx..($allImages.Count-1)]

Write-Host "Dataset split:" -ForegroundColor Yellow
Write-Host "  Training: $($trainImages.Count) images (80%)" -ForegroundColor Green
Write-Host "  Validation: $($valImages.Count) images (20%)" -ForegroundColor Green
Write-Host ""

# 生成注释文件
$trainAnno = "annotation/train_doubao_only.txt"
$valAnno = "annotation/val_doubao_only.txt"
$mixedAnno = "annotation/train_sdv5_doubao_mixed.txt"

# 纯豆包数据集
$trainImages | ForEach-Object { "$($_.Path) $($_.Label)" } | Out-File $trainAnno -Encoding UTF8
$valImages | ForEach-Object { "$($_.Path) $($_.Label)" } | Out-File $valAnno -Encoding UTF8

Write-Host "Created annotation files:" -ForegroundColor Green
Write-Host "  ✓ $trainAnno" -ForegroundColor Green
Write-Host "  ✓ $valAnno" -ForegroundColor Green

# 如果原有SD数据集存在，创建混合版本
$originalTrain = "annotation/train_sdv5.txt"
if (Test-Path $originalTrain) {
    Write-Host ""
    Write-Host "Merging with original SD1.5 dataset..." -ForegroundColor Yellow
    
    $sdLines = Get-Content $originalTrain
    $doubaoLines = $trainImages | ForEach-Object { "$($_.Path) $($_.Label)" }
    
    # 合并并打乱
    $mixedLines = $sdLines + $doubaoLines | Get-Random -Count ($sdLines.Count + $doubaoLines.Count)
    $mixedLines | Out-File $mixedAnno -Encoding UTF8
    
    Write-Host "  ✓ $mixedAnno (SD1.5: $($sdLines.Count) + Doubao: $($doubaoLines.Count) = $($mixedLines.Count))" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Note: Original SD1.5 dataset not found at $originalTrain" -ForegroundColor Yellow
    Write-Host "Using Doubao-only dataset."
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Annotation files ready!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""
Write-Host "Training options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 1: Train on Doubao-only (Quick test)"
Write-Host "  .\train_with_doubao.ps1 -UseDoubaoOnly" -ForegroundColor Yellow
Write-Host ""
Write-Host "Option 2: Train on Mixed dataset (Recommended)"
Write-Host "  .\train_with_doubao.ps1" -ForegroundColor Yellow
Write-Host ""

# 显示数据分布统计
Write-Host "Dataset statistics:" -ForegroundColor Cyan
Write-Host ""
$trainStats = $trainImages | Group-Object Label
$valStats = $valImages | Group-Object Label
Write-Host "Training set:"
foreach ($stat in $trainStats) {
    $labelName = if ($stat.Name -eq "1") { "AI Generated" } else { "Real Photo" }
    Write-Host "  $labelName : $($stat.Count) ($([math]::Round($stat.Count/$trainImages.Count*100, 1))%)"
}
Write-Host ""
Write-Host "Validation set:"
foreach ($stat in $valStats) {
    $labelName = if ($stat.Name -eq "1") { "AI Generated" } else { "Real Photo" }
    Write-Host "  $labelName : $($stat.Count) ($([math]::Round($stat.Count/$valImages.Count*100, 1))%)"
}
Write-Host ""

# 检查数据平衡性
$aiRatio = ($trainImages | Where-Object { $_.Label -eq 1 }).Count / $trainImages.Count
if ($aiRatio -lt 0.3 -or $aiRatio -gt 0.7) {
    Write-Host "Warning: Dataset is imbalanced (AI: $([math]::Round($aiRatio*100, 1))%)" -ForegroundColor Yellow
    Write-Host "Consider adding more " -NoNewline
    if ($aiRatio -lt 0.5) {
        Write-Host "AI images" -ForegroundColor Yellow -NoNewline
    } else {
        Write-Host "real photos" -ForegroundColor Yellow -NoNewline
    }
    Write-Host " for better training."
    Write-Host ""
}
