<#
创建小规模训练集：1万 SD1.5 AI + 1万真实 + 豆包数据

关键修复：
- 使用 UTF-8 StreamReader 读取，避免中文路径（如“三创”）被读成乱码（如“涓夊垱”）
- 采用水库抽样（reservoir sampling）避免 Get-Content 读入 31 万行占用终端/内存
- 输出统一为 UTF-8，保证路径不再被破坏
#>

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Create Small Training Set (10k+10k+Doubao) ===" -ForegroundColor Cyan
Write-Host ""

$originalTrain = "annotation/train_sdv5.txt"
$smallTrain = "annotation/train_sdv5_10k.txt"
$smallDoubaoMixed = "annotation/train_sdv5_10k_doubao.txt"
$doubaoOnly = "annotation/train_doubao_only.txt"

function Shuffle-Array {
    param(
        [Parameter(Mandatory = $true)] [string[]] $Array,
        [Parameter(Mandatory = $true)] [System.Random] $Rand
    )
    for ($i = $Array.Length - 1; $i -gt 0; $i--) {
        $j = $Rand.Next($i + 1)
        $tmp = $Array[$i]
        $Array[$i] = $Array[$j]
        $Array[$j] = $tmp
    }
    return $Array
}

function Read-LinesUtf8 {
    param([Parameter(Mandatory = $true)] [string] $Path)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $sr = [System.IO.StreamReader]::new($Path, $encoding, $true)
    try {
        while (-not $sr.EndOfStream) {
            $line = $sr.ReadLine()
            if ($null -ne $line -and $line.Trim().Length -gt 0) {
                $line
            }
        }
    }
    finally {
        $sr.Close()
    }
}

function Write-LinesUtf8NoBom {
    param(
        [Parameter(Mandatory = $true)] [string] $Path,
        [Parameter(Mandatory = $true)] [string[]] $Lines
    )
    $enc = [System.Text.UTF8Encoding]::new($false)
    $sw = [System.IO.StreamWriter]::new($Path, $false, $enc)
    try {
        foreach ($l in $Lines) {
            $sw.WriteLine($l)
        }
    } finally {
        $sw.Close()
    }
}

function Test-AnnotationPathsSample {
    param(
        [Parameter(Mandatory = $true)] [string] $AnnoPath,
        [int] $SampleSize = 300
    )
    $exists = 0
    $checked = 0
    foreach ($line in (Read-LinesUtf8 -Path $AnnoPath)) {
        if ($checked -ge $SampleSize) { break }
        $trim = $line.Trim()
        if ($trim.Length -eq 0) { continue }
        $lastSpace = $trim.LastIndexOf(' ')
        $imgPath = if ($lastSpace -gt 0) { $trim.Substring(0, $lastSpace) } else { $trim }
        $imgPath = $imgPath.Replace('\\', '/')
        if (Test-Path $imgPath) { $exists++ }
        $checked++
    }
    return @{ checked = $checked; exists = $exists }
}

# 检查原始数据
if (-not (Test-Path $originalTrain)) {
    Write-Host "Error: Original training file not found: $originalTrain" -ForegroundColor Red
    exit 1
}

Write-Host "Sampling 10,000 images from each class (streaming, UTF-8)..." -ForegroundColor Yellow

$rand = [System.Random]::new(20260123)
$targetPerClass = 10000
$aiSample = New-Object System.Collections.Generic.List[string]
$realSample = New-Object System.Collections.Generic.List[string]
$aiSeen = 0
$realSeen = 0

foreach ($line in (Read-LinesUtf8 -Path $originalTrain)) {
    $trim = $line.Trim()
    if ($trim.EndsWith(" 1")) {
        $aiSeen++
        if ($aiSample.Count -lt $targetPerClass) {
            [void]$aiSample.Add($trim)
        } else {
            $j = $rand.Next($aiSeen)
            if ($j -lt $targetPerClass) {
                $aiSample[$j] = $trim
            }
        }
    } elseif ($trim.EndsWith(" 0")) {
        $realSeen++
        if ($realSample.Count -lt $targetPerClass) {
            [void]$realSample.Add($trim)
        } else {
            $j = $rand.Next($realSeen)
            if ($j -lt $targetPerClass) {
                $realSample[$j] = $trim
            }
        }
    }
}

Write-Host "Original dataset (scanned):" -ForegroundColor Cyan
Write-Host "  AI lines seen: $aiSeen" -ForegroundColor Green
Write-Host "  Real lines seen: $realSeen" -ForegroundColor Green
Write-Host ""

if ($aiSample.Count -lt $targetPerClass -or $realSample.Count -lt $targetPerClass) {
    Write-Host "Error: Not enough samples collected. AI=$($aiSample.Count), Real=$($realSample.Count)" -ForegroundColor Red
    exit 1
}

$smallDataset = @($aiSample.ToArray() + $realSample.ToArray())
$smallDataset = Shuffle-Array -Array $smallDataset -Rand $rand

Write-LinesUtf8NoBom -Path $smallTrain -Lines $smallDataset

Write-Host "✓ Created small dataset: $smallTrain" -ForegroundColor Green
Write-Host "  AI: 10,000" -ForegroundColor Green
Write-Host "  Real: 10,000" -ForegroundColor Green
Write-Host "  Total: 20,000" -ForegroundColor Green
Write-Host ""

# 添加豆包数据（如果存在）
if (Test-Path $doubaoOnly) {
    Write-Host "Adding Doubao data..." -ForegroundColor Yellow
    $doubaoLines = @(Read-LinesUtf8 -Path $doubaoOnly)
    $doubaoCount = $doubaoLines.Count
    
    # 合并并打乱
    $mixedDataset = @($smallDataset + $doubaoLines)
    $mixedDataset = Shuffle-Array -Array $mixedDataset -Rand $rand
    Write-LinesUtf8NoBom -Path $smallDoubaoMixed -Lines $mixedDataset
    
    Write-Host "✓ Created mixed dataset: $smallDoubaoMixed" -ForegroundColor Green
    Write-Host "  SD1.5: 20,000 (50%)" -ForegroundColor Green
    Write-Host "  Doubao: $doubaoCount ($([math]::Round($doubaoCount/($doubaoCount+20000)*100, 1))%)" -ForegroundColor Green
    Write-Host "  Total: $($mixedDataset.Count)" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "Note: Doubao data not found at $doubaoOnly" -ForegroundColor Yellow
    Write-Host "Using SD1.5-only dataset." -ForegroundColor Yellow
    Copy-Item $smallTrain $smallDoubaoMixed
}

Write-Host "Validating generated annotation paths (sampled)..." -ForegroundColor Yellow
$stat = Test-AnnotationPathsSample -AnnoPath $smallDoubaoMixed -SampleSize 300
Write-Host ("  Exists(sample): {0}/{1}" -f $stat.exists, $stat.checked)
if ($stat.checked -gt 0 -and ($stat.exists / $stat.checked) -lt 0.95) {
    Write-Host "ERROR: Too many image paths do not exist in the generated annotation." -ForegroundColor Red
    Write-Host "This indicates encoding/path corruption. Please avoid editing the file with non-UTF8 tools." -ForegroundColor Red
    exit 2
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Small training set ready!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""
Write-Host "Training annotation: $smallDoubaoMixed"
Write-Host ""
Write-Host "Estimated training time:" -ForegroundColor Cyan
Write-Host "  Feature extraction: depends on model (SD1.5 faster; SDXL slower)"
Write-Host "  Model training: ~1 hour (20 epochs)"
Write-Host "  Total: ~2 hours" -ForegroundColor Yellow
Write-Host ""
Write-Host "To start training, run:"
Write-Host "  .\script\ps1\train_small_dataset.ps1" -ForegroundColor Yellow
