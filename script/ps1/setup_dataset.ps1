# LaRE Project - Dataset Setup Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "LaRE Dataset Setup" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $projectRoot
$zipPath = Join-Path $projectRoot "stable_diffusion_v_1_5\imagenet_ai_0424_sdv5.zip"
$extractPath = Join-Path $projectRoot "GenImage_Dataset"

# Step 1: Check dataset
Write-Host "[Step 1] Checking dataset archive..." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

if (-not (Test-Path $zipPath))
{
    Write-Host "ERROR: Archive not found: $zipPath" -ForegroundColor Red
    exit 1
}

Write-Host "OK: Found archive file" -ForegroundColor Green

# Check split files
$z01Path = Join-Path $projectRoot "stable_diffusion_v_1_5\imagenet_ai_0424_sdv5.z01"
if (-not (Test-Path $z01Path))
{
    Write-Host "ERROR: Split files incomplete" -ForegroundColor Red
    Write-Host "Need: .zip, .z01-.z30" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK: Found split files (.z01-.z30)" -ForegroundColor Green
Write-Host ""
Write-Host "This is a multi-volume archive, requires 7-Zip" -ForegroundColor Yellow
Write-Host ""

# Check if already extracted
if (Test-Path $extractPath)
{
    Write-Host "WARNING: Target directory exists: $extractPath" -ForegroundColor Yellow
    $response = Read-Host "Re-extract? (y/N)"
    if (($response -eq 'y') -or ($response -eq 'Y'))
    {
        Write-Host "Removing old directory..." -ForegroundColor Yellow
        Remove-Item $extractPath -Recurse -Force
    }
    else
    {
        Write-Host "Skipping extraction" -ForegroundColor Gray
        $skipExtract = $true
    }
}

# Extract
if ((-not (Test-Path $extractPath)) -and (-not $skipExtract))
{
    # Find 7-Zip
    $sevenZipPaths = @(
        "C:\Program Files\7-Zip\7z.exe",
        "C:\Program Files (x86)\7-Zip\7z.exe",
        "$env:ProgramFiles\7-Zip\7z.exe"
    )
    
    $sevenZip = $null
    foreach ($path in $sevenZipPaths)
    {
        if (Test-Path $path)
        {
            $sevenZip = $path
            break
        }
    }
    
    if ($sevenZip)
    {
        Write-Host "OK: Found 7-Zip: $sevenZip" -ForegroundColor Green
        Write-Host ""
        Write-Host "Extracting..." -ForegroundColor Cyan
        Write-Host "This may take 10-30 minutes, please wait" -ForegroundColor Yellow
        Write-Host ""
        
        try
        {
            & $sevenZip x $zipPath "-o$extractPath" -y
            Write-Host ""
            Write-Host "OK: Extraction complete!" -ForegroundColor Green
        }
        catch
        {
            Write-Host "ERROR: Extraction failed: $_" -ForegroundColor Red
            Write-Host "Please extract manually with 7-Zip" -ForegroundColor Yellow
        }
    }
    else
    {
        Write-Host "ERROR: 7-Zip not found" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please install 7-Zip:" -ForegroundColor Yellow
        Write-Host "  1. Download: https://www.7-zip.org/" -ForegroundColor Cyan
        Write-Host "  2. Install and re-run this script" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Or extract manually:" -ForegroundColor Yellow
        Write-Host "  1. Right-click imagenet_ai_0424_sdv5.zip" -ForegroundColor Cyan
        Write-Host "  2. 7-Zip -> Extract to..." -ForegroundColor Cyan
        Write-Host "  3. Target: GenImage_Dataset" -ForegroundColor Cyan
        exit 1
    }
}

# Step 2: Create directories
Write-Host ""
Write-Host "[Step 2] Creating directory structure..." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

$dirs = @(
    "annotation",
    "features",
    "features\sdv5_train",
    "features\sdv5_val",
    "outputs",
    "outputs\saved_models"
)

foreach ($dir in $dirs)
{
    $fullPath = Join-Path $projectRoot $dir
    if (-not (Test-Path $fullPath))
    {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "OK: Created $dir" -ForegroundColor Green
    }
    else
    {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

# Step 3: Check extraction result
Write-Host ""
Write-Host "[Step 3] Checking dataset structure..." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray

if (Test-Path $extractPath)
{
    Write-Host "Dataset directory: $extractPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Contents:" -ForegroundColor White
    
    Get-ChildItem $extractPath | ForEach-Object {
        if ($_.PSIsContainer)
        {
            Write-Host "  [DIR] $($_.Name)" -ForegroundColor Cyan
            
            $trainPath = Join-Path $_.FullName "train"
            if (Test-Path $trainPath)
            {
                Write-Host "     [DIR] train" -ForegroundColor Green
                
                $aiPath = Join-Path $trainPath "ai"
                $naturePath = Join-Path $trainPath "nature"
                
                if (Test-Path $aiPath)
                {
                    $aiCount = (Get-ChildItem $aiPath -File).Count
                    Write-Host "        [DIR] ai ($aiCount files)" -ForegroundColor Green
                }
                if (Test-Path $naturePath)
                {
                    $natureCount = (Get-ChildItem $naturePath -File).Count
                    Write-Host "        [DIR] nature ($natureCount files)" -ForegroundColor Green
                }
            }
        }
    }
}
else
{
    Write-Host "ERROR: Dataset directory not found" -ForegroundColor Red
}

# Step 4: Next steps
Write-Host ""
Write-Host "[Step 4] Next steps..." -ForegroundColor Yellow
Write-Host "------------------------------------------------------------" -ForegroundColor Gray
Write-Host ""
Write-Host "Setup complete! Next:" -ForegroundColor Green
Write-Host ""
Write-Host "1. Generate annotations:" -ForegroundColor White
Write-Host "   python 模型测试/1_标注生成.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Extract features (requires GPU):" -ForegroundColor White
Write-Host "   python 模型测试/2_特征提取.py ..." -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Setup script completed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
