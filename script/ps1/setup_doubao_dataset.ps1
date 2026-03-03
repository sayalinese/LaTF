# 豆包数据集初始化脚本
# 创建标准目录结构并生成注释文件

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Doubao Dataset Setup ===" -ForegroundColor Cyan
Write-Host ""

$baseDir = "GenImage_Dataset"

# 创建目录结构
$dirs = @(
    "$baseDir/doubao_ai",
    "$baseDir/doubao_real",
    "$baseDir/midjourney",
    "$baseDir/dalle3",
    "$baseDir/flux"
)

Write-Host "Creating directory structure..." -ForegroundColor Yellow
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✓ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  - Exists: $dir" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Directory structure created!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Add Doubao AI images to: $baseDir/doubao_ai/"
Write-Host "     Recommended: 300-500 images"
Write-Host ""
Write-Host "  2. (Optional) Add real photos to: $baseDir/doubao_real/"
Write-Host "     Recommended: 100-200 images for balance"
Write-Host ""
Write-Host "  3. (Optional) Add other AI images to: midjourney/, dalle3/, flux/"
Write-Host "     Each: 100-300 images"
Write-Host ""
Write-Host "After adding images, run: .\script\ps1\generate_doubao_annotations.ps1"
Write-Host ""

# 创建README
$readmeContent = @"
# 豆包数据集目录说明

## 目录结构

### doubao_ai/
存放豆包生成的AI图片（标签=1，AI生成）
- 推荐数量：300-500张
- 命名格式：doubao_00001.jpg, doubao_00002.png, ...
- 要求：多样化主题，分辨率 ≥512x512

### doubao_real/
存放真实照片作为对照（标签=0，真实照片）
- 推荐数量：100-200张
- 命名格式：real_00001.jpg, ...
- 用途：平衡训练集，可选

### midjourney/
存放Midjourney生成的图片（标签=1）
- 推荐数量：100-300张
- 用途：增强对其他AI工具的检测能力

### dalle3/
存放DALL-E 3生成的图片（标签=1）
- 推荐数量：100-300张

### flux/
存放Flux生成的图片（标签=1）
- 推荐数量：100-300张

## 数据收集建议

### 豆包图片来源
1. 豆包官网/APP直接生成
2. 社交媒体（小红书、微博）搜索"豆包AI"
3. AI生成图片论坛/社区

### 真实照片来源
1. COCO数据集
2. ImageNet真实图片
3. 自己拍摄的照片
4. Unsplash等免费图库

### 主题多样性
建议包含以下类别（每类50-100张）：
- 人物肖像
- 自然风景
- 城市建筑
- 动物
- 静物/产品
- 艺术画作风格

## 数据质量要求

### 必须满足
✓ 清晰度高（无模糊）
✓ 完整图片（无裁剪边缘）
✓ 格式正确（JPG/PNG）

### 避免
✗ 带水印的图片
✗ 拼图/多图组合
✗ 过度编辑的图片
✗ 重复图片

## 下一步

数据收集完成后，运行：
```powershell
.\script\ps1\generate_doubao_annotations.ps1
```

这将自动生成训练所需的注释文件。
"@

$readmeContent | Out-File "$baseDir/README.md" -Encoding UTF8
Write-Host "Created README: $baseDir/README.md" -ForegroundColor Green
Write-Host ""
Write-Host "Please read the README for detailed instructions."

# Optional: generate annotations immediately if user wants.
$genScript = Join-Path $PSScriptRoot "generate_doubao_annotations.ps1"
if (Test-Path $genScript) {
    Write-Host "" 
    $doGen = Read-Host "Run annotation generation now? (y/n)"
    if ($doGen -eq "y") {
        & $genScript
    }
}
