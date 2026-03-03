# 豆包数据集规范化重命名脚本
# 统一命名格式：doubao_00001.jpg, real_00001.jpg

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Doubao Dataset Rename Utility ===" -ForegroundColor Cyan
Write-Host ""

# 重命名函数
function Rename-Images {
    param(
        [string]$Path,
        [string]$Prefix
    )
    
    if (-not (Test-Path $Path)) {
        Write-Host "Warning: $Path not found, skipping..." -ForegroundColor Yellow
        return
    }
    
    # 获取所有图片文件（支持多种扩展名）
    $files = @()
    $files += Get-ChildItem $Path -Filter *.jpg -File -ErrorAction SilentlyContinue
    $files += Get-ChildItem $Path -Filter *.jpeg -File -ErrorAction SilentlyContinue
    $files += Get-ChildItem $Path -Filter *.png -File -ErrorAction SilentlyContinue
    $files += Get-ChildItem $Path -Filter *.webp -File -ErrorAction SilentlyContinue
    $files = $files | Sort-Object Name
    
    if ($files.Count -eq 0) {
        Write-Host "No images found in $Path" -ForegroundColor Yellow
        return
    }
    
    Write-Host "Processing: $Path" -ForegroundColor Yellow
    Write-Host "  Found: $($files.Count) images" -ForegroundColor Green
    
    $counter = 1
    foreach ($file in $files) {
        $newName = "{0}_{1:D5}{2}" -f $Prefix, $counter, $file.Extension
        $newPath = Join-Path $file.DirectoryName $newName
        
        # 避免重复重命名
        if ($file.FullName -ne $newPath) {
            try {
                Rename-Item -Path $file.FullName -NewName $newName -Force
                Write-Host "  ✓ $($file.Name) -> $newName" -ForegroundColor Gray
            } catch {
                Write-Host "  ✗ Failed to rename $($file.Name): $_" -ForegroundColor Red
            }
        }
        
        $counter++
    }
    
    Write-Host "  Completed: $($files.Count) files renamed" -ForegroundColor Green
    Write-Host ""
}

Write-Host "This will rename all images in doubao_ai/ and doubao_real/ to standard format."
Write-Host ""
Write-Host "Example:"
Write-Host "  doubao_ai/1.png -> doubao_ai/doubao_00001.png"
Write-Host "  doubao_real/苹果.jpg -> doubao_real/real_00001.jpg"
Write-Host ""

$confirm = Read-Host "Proceed with rename? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Starting rename..." -ForegroundColor Cyan
Write-Host ""

# 重命名 doubao_ai
Rename-Images -Path "GenImage_Dataset/doubao_ai" -Prefix "doubao"

# 重命名 doubao_real
Rename-Images -Path "GenImage_Dataset/doubao_real" -Prefix "real"

# 重命名其他可选目录
Rename-Images -Path "GenImage_Dataset/midjourney" -Prefix "mj"
Rename-Images -Path "GenImage_Dataset/dalle3" -Prefix "dalle"
Rename-Images -Path "GenImage_Dataset/flux" -Prefix "flux"

Write-Host "============================================" -ForegroundColor Green
Write-Host "Rename completed!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""

# 显示统计
$aiFiles = @()
$aiFiles += Get-ChildItem "GenImage_Dataset/doubao_ai" -Filter doubao_*.jpg -ErrorAction SilentlyContinue
$aiFiles += Get-ChildItem "GenImage_Dataset/doubao_ai" -Filter doubao_*.png -ErrorAction SilentlyContinue
$aiCount = $aiFiles.Count

$realFiles = @()
$realFiles += Get-ChildItem "GenImage_Dataset/doubao_real" -Filter real_*.jpg -ErrorAction SilentlyContinue
$realFiles += Get-ChildItem "GenImage_Dataset/doubao_real" -Filter real_*.png -ErrorAction SilentlyContinue
$realCount = $realFiles.Count

Write-Host "Final statistics:" -ForegroundColor Cyan
Write-Host "  Doubao AI images: $aiCount" -ForegroundColor Green
Write-Host "  Real photos: $realCount" -ForegroundColor Green
Write-Host "  Total: $($aiCount + $realCount)" -ForegroundColor Cyan
Write-Host ""

if ($aiCount -ge 100) {
    Write-Host "✓ Dataset ready for training!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: .\script\ps1\generate_doubao_annotations.ps1"
} else {
    Write-Host "Note: At least 100 AI images recommended (current: $aiCount)" -ForegroundColor Yellow
    Write-Host "Consider adding more images for better results."
}
