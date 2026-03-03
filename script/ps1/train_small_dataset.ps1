# 小数据集快速训练脚本
# 1万SD1.5 AI + 1万真实 + 豆包数据

$ErrorActionPreference = "Stop"

# Always run from project root.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
Set-Location -Path $ProjectRoot

Write-Host "=== Quick Training with Small Dataset ===" -ForegroundColor Cyan
Write-Host ""

$TRAIN_ANNO = "annotation/train_sdv5_10k_doubao.txt"
$VAL_ANNO = "annotation/val_sdv5.txt"
$TEST_ANNO = "annotation/test_sdv5_20k_smoke.txt"
$TRAIN_FEAT = "features/sdv5_train_10k_doubao_sd15"
$VAL_FEAT = "features/sdv5_val_sd15"
$TEST_FEAT = "features/sdv5_test_sdv5_20k_smoke_sd15"

# Scripts (after repo cleanup)
# Avoid hardcoding non-ASCII folder names (can get corrupted by encoding).
$TrainScriptItem = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "3_模型训练.py" -File -ErrorAction Stop | Select-Object -First 1
if ($null -eq $TrainScriptItem) { throw "Cannot find 3_模型训练.py under $ProjectRoot" }

$TrainScript = $TrainScriptItem.FullName
$TrainScriptDir = $TrainScriptItem.Directory.FullName

$ExtractScript = Join-Path $TrainScriptDir "2_特征提取.py"
if (-not (Test-Path $ExtractScript)) { throw "Missing extractor beside trainer: $ExtractScript" }

$MapScript = Join-Path $ProjectRoot "script\create_map_file.py"
if (-not (Test-Path $MapScript)) { throw "Missing map tool: $MapScript" }

# Outputs live next to the trainer script (e.g., 模型训练/output/...) but derived without hardcoding.
$OUTPUT = Join-Path $TrainScriptDir "output\Expsdv5_wmap_sd15_10k_doubao"

# ======= 可调参数（先保证稳定跑完） =======
$BATCH_SIZE = 16
$TEST_BATCH_SIZE = 16
# 解冻 CLIP 会显著增加显存占用；这里按需求在第 3 个 epoch 解冻，同时降低 batch
$BATCH_SIZE = 8
$FREEZE_CLIP_EPOCHS = 3

function Get-Utf8LineCount([string]$Path) {
    $reader = $null
    try {
        $reader = [System.IO.StreamReader]::new($Path, [System.Text.Encoding]::UTF8, $true)
        $count = 0
        while ($null -ne $reader.ReadLine()) { $count++ }
        return $count
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
    }
}

function Test-AnnotationPaths([string]$AnnoPath, [int]$SampleSize = 300) {
    $reader = $null
    try {
        $reader = [System.IO.StreamReader]::new($AnnoPath, [System.Text.Encoding]::UTF8, $true)
        $checked = 0
        $exists = 0
        while ($checked -lt $SampleSize) {
            $line = $reader.ReadLine()
            if ($null -eq $line) { break }
            $line = $line.Trim()
            if ($line.Length -eq 0) { continue }
            $imgPath = $null
            if ($line.Contains("`t")) {
                $parts = $line.Split("`t")
                $imgPath = $parts[0]
            } else {
                $lastSpace = $line.LastIndexOf(' ')
                if ($lastSpace -gt 0) {
                    $imgPath = $line.Substring(0, $lastSpace)
                } else {
                    $imgPath = $line
                }
            }
            $imgPath = $imgPath.Replace('\\', '/')
            if (Test-Path $imgPath) { $exists++ }
            $checked++
        }
        return @{ checked = $checked; exists = $exists }
    } finally {
        if ($null -ne $reader) { $reader.Dispose() }
    }
}

function Get-PtCount([string]$DirPath) {
    if (-not (Test-Path $DirPath)) { return 0 }
    return (Get-ChildItem -Path $DirPath -Recurse -Filter "*.pt" -File | Measure-Object).Count
}

function Write-CombinedMap([string]$OutPath, [string[]]$InPaths) {
    $writer = $null
    try {
        $writer = [System.IO.StreamWriter]::new($OutPath, $false, [System.Text.Encoding]::UTF8)
        foreach ($p in $InPaths) {
            if (-not (Test-Path $p)) { continue }
            $reader = $null
            try {
                $reader = [System.IO.StreamReader]::new($p, [System.Text.Encoding]::UTF8, $true)
                while ($true) {
                    $line = $reader.ReadLine()
                    if ($null -eq $line) { break }
                    $writer.WriteLine($line)
                }
            } finally {
                if ($null -ne $reader) { $reader.Dispose() }
            }
        }
    } finally {
        if ($null -ne $writer) { $writer.Dispose() }
    }
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)] [string] $Script,
        [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Args
    )
    & python $Script @Args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: python $Script failed (exit=$LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 检查标注文件
if (-not (Test-Path $TRAIN_ANNO)) {
    Write-Host "Error: Training annotation not found!" -ForegroundColor Red
    Write-Host "Please run: .\script\ps1\create_small_trainset.ps1 first"
    exit 1
}

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Training data: $TRAIN_ANNO"
Write-Host "  Features: $TRAIN_FEAT"
Write-Host "  Output: $OUTPUT"
if (Test-Path $TEST_ANNO) {
    Write-Host "  Test file: $TEST_ANNO" -ForegroundColor DarkGray
}
Write-Host ""

$trainCount = Get-Utf8LineCount $TRAIN_ANNO
$valCount = if (Test-Path $VAL_ANNO) { Get-Utf8LineCount $VAL_ANNO } else { 0 }
$testCount = if (Test-Path $TEST_ANNO) { Get-Utf8LineCount $TEST_ANNO } else { 0 }
Write-Host "Train lines: $trainCount" -ForegroundColor Green
Write-Host "Val lines:   $valCount" -ForegroundColor Green
if ($testCount -gt 0) {
    Write-Host "Test lines:  $testCount" -ForegroundColor Green
}
Write-Host ""

Write-Host "Sanity-checking annotation paths (sampled, no full scan)..." -ForegroundColor Yellow
$trainPathStat = Test-AnnotationPaths $TRAIN_ANNO 300
Write-Host ("  Train exists(sample): {0}/{1}" -f $trainPathStat.exists, $trainPathStat.checked)
if ($trainPathStat.checked -gt 0 -and ($trainPathStat.exists / $trainPathStat.checked) -lt 0.95) {
    Write-Host "" 
    Write-Host "ERROR: Too many training image paths do not exist." -ForegroundColor Red
    Write-Host "This usually means path encoding got corrupted (e.g. '三创' -> '涓夊垱')." -ForegroundColor Red
    Write-Host "Fix: re-generate $TRAIN_ANNO via .\script\ps1\create_small_trainset.ps1 (UTF-8)." -ForegroundColor Yellow
    exit 2
}

if (Test-Path $VAL_ANNO) {
    $valPathStat = Test-AnnotationPaths $VAL_ANNO 200
    Write-Host ("  Val exists(sample):   {0}/{1}" -f $valPathStat.exists, $valPathStat.checked)
    if ($valPathStat.checked -gt 0 -and ($valPathStat.exists / $valPathStat.checked) -lt 0.95) {
        Write-Host "ERROR: Too many validation image paths do not exist." -ForegroundColor Red
        exit 2
    }
}

if (Test-Path $TEST_ANNO) {
    $testPathStat = Test-AnnotationPaths $TEST_ANNO 200
    Write-Host ("  Test exists(sample):  {0}/{1}" -f $testPathStat.exists, $testPathStat.checked)
    if ($testPathStat.checked -gt 0 -and ($testPathStat.exists / $testPathStat.checked) -lt 0.95) {
        Write-Host "ERROR: Too many test image paths do not exist." -ForegroundColor Red
        exit 2
    }
}

Write-Host ""
Write-Host "[1/2] Extracting SD1.5 LaRE loss-map features..." -ForegroundColor Cyan
Write-Host "============================================"
Write-Host ""

# 训练集特征提取（目录存在也可能是不完整的，必须检查 .pt 数量）
$trainPt = Get-PtCount $TRAIN_FEAT
Write-Host "Train feature .pt files: $trainPt" -ForegroundColor Yellow
if ($trainPt -gt [Math]::Ceiling($trainCount * 1.05)) {
    Write-Host "WARNING: Train feature count looks higher than annotation lines." -ForegroundColor Yellow
    Write-Host "This may indicate stale features from previous runs. Consider deleting $TRAIN_FEAT and re-running." -ForegroundColor Yellow
}
if (($trainPt -lt [Math]::Floor($trainCount * 0.98)) -or (-not (Test-Path $TRAIN_FEAT))) {
    Write-Host "Extracting training features (will overwrite/resume existing .pt files)..." -ForegroundColor Yellow
    Invoke-PythonChecked $ExtractScript `
        --input_path $TRAIN_ANNO `
        --output_path $TRAIN_FEAT `
        --t 280 `
        --prompt "a photo" `
        --ensemble_size 1 `
        --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
        --img_size 256 256 `
        --n-gpus 1 `
        --dtype fp16 `
        --extract_batch_size 16
} else {
    Write-Host "Training features look complete, skipping extraction." -ForegroundColor Green
}

# 验证集特征（同样检查完整性）
if (Test-Path $VAL_ANNO) {
    $valPt = Get-PtCount $VAL_FEAT
    Write-Host "Val feature .pt files:   $valPt" -ForegroundColor Yellow
    if ($valCount -gt 0 -and $valPt -gt [Math]::Ceiling($valCount * 1.05)) {
        Write-Host "WARNING: Val feature count looks higher than annotation lines." -ForegroundColor Yellow
        Write-Host "This may indicate stale features from previous runs. Consider deleting $VAL_FEAT and re-running." -ForegroundColor Yellow
    }
    if (($valCount -gt 0) -and (($valPt -lt [Math]::Floor($valCount * 0.98)) -or (-not (Test-Path $VAL_FEAT)))) {
        Write-Host "Extracting validation features (will overwrite/resume existing .pt files)..." -ForegroundColor Yellow
        Invoke-PythonChecked $ExtractScript `
            --input_path $VAL_ANNO `
            --output_path $VAL_FEAT `
            --t 280 `
            --prompt "a photo" `
            --ensemble_size 1 `
            --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
            --img_size 256 256 `
            --n-gpus 1 `
            --dtype fp16 `
            --extract_batch_size 16
    } else {
        Write-Host "Validation features look complete, skipping extraction." -ForegroundColor Green
    }
} else {
    Write-Host "WARNING: Validation annotation not found, skipping val extraction." -ForegroundColor Yellow
}

# 测试集特征（smoke test），用于训练脚本内部的 Test Dataset 评估
if (Test-Path $TEST_ANNO) {
    $testPt = Get-PtCount $TEST_FEAT
    Write-Host "Test feature .pt files:  $testPt" -ForegroundColor Yellow
    if ($testCount -gt 0 -and $testPt -gt [Math]::Ceiling($testCount * 1.05)) {
        Write-Host "WARNING: Test feature count looks higher than annotation lines." -ForegroundColor Yellow
        Write-Host "This may indicate stale features from previous runs. Consider deleting $TEST_FEAT and re-running." -ForegroundColor Yellow
    }
    if (($testCount -gt 0) -and (($testPt -lt [Math]::Floor($testCount * 0.98)) -or (-not (Test-Path $TEST_FEAT)))) {
        Write-Host "Extracting test features (will overwrite/resume existing .pt files)..." -ForegroundColor Yellow
        Invoke-PythonChecked $ExtractScript `
            --input_path $TEST_ANNO `
            --output_path $TEST_FEAT `
            --t 280 `
            --prompt "a photo" `
            --ensemble_size 1 `
            --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" `
            --img_size 256 256 `
            --n-gpus 1 `
            --dtype fp16 `
            --extract_batch_size 16
    } else {
        Write-Host "Test features look complete, skipping extraction." -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "[2/2] Training classifier..." -ForegroundColor Cyan
Write-Host "============================================"
Write-Host ""

# 生成 map 文件
$TRAIN_MAP = "annotation/map_sdv5_10k_doubao_sd15.txt"
$VAL_MAP = "annotation/map_sdv5_val_sd15.txt"
$TEST_MAP = "annotation/map_sdv5_test_sdv5_20k_smoke_sd15.txt"

$COMBINED_MAP = "annotation/map_sdv5_10k_doubao_sd15_combined.txt"

Write-Host "Generating training map file..." -ForegroundColor Yellow
Invoke-PythonChecked $MapScript --feature_dir $TRAIN_FEAT --annotation_file $TRAIN_ANNO --output $TRAIN_MAP

if (Test-Path $VAL_ANNO) {
    Write-Host "Generating validation map file..." -ForegroundColor Yellow
    Invoke-PythonChecked $MapScript --feature_dir $VAL_FEAT --annotation_file $VAL_ANNO --output $VAL_MAP
}

if (Test-Path $TEST_ANNO) {
    Write-Host "Generating test map file..." -ForegroundColor Yellow
    Invoke-PythonChecked $MapScript --feature_dir $TEST_FEAT --annotation_file $TEST_ANNO --output $TEST_MAP
}

Write-Host "Merging train+val(+test) map file..." -ForegroundColor Yellow
$mapsToMerge = New-Object System.Collections.Generic.List[string]
[void]$mapsToMerge.Add($TRAIN_MAP)
if (Test-Path $VAL_MAP) { [void]$mapsToMerge.Add($VAL_MAP) }
if (Test-Path $TEST_MAP) { [void]$mapsToMerge.Add($TEST_MAP) }
Write-CombinedMap -OutPath $COMBINED_MAP -InPaths $mapsToMerge.ToArray()

if (Test-Path $TEST_ANNO) {
    Invoke-PythonChecked $TrainScript `
        --data_root "." `
        --train_file $TRAIN_ANNO `
        --val_file $VAL_ANNO `
        --test_file $TEST_ANNO `
        --map_file $COMBINED_MAP `
        --out_dir $OUTPUT `
        --batch_size $BATCH_SIZE `
        --test_batch_size $TEST_BATCH_SIZE `
        --freeze_clip_epochs $FREEZE_CLIP_EPOCHS `
        --epoches 20 `
        --lr 0.0001 `
        --clip_type RN50x64 `
        --num_class 2 `
        --use_amp
} else {
    Invoke-PythonChecked $TrainScript `
        --data_root "." `
        --train_file $TRAIN_ANNO `
        --val_file $VAL_ANNO `
        --map_file $COMBINED_MAP `
        --out_dir $OUTPUT `
        --batch_size $BATCH_SIZE `
        --test_batch_size $TEST_BATCH_SIZE `
        --freeze_clip_epochs $FREEZE_CLIP_EPOCHS `
        --epoches 20 `
        --lr 0.0001 `
        --clip_type RN50x64 `
        --num_class 2 `
        --use_amp
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "Training Completed!" -ForegroundColor Green
Write-Host "============================================"
Write-Host ""
Write-Host "Best model: $OUTPUT/Val_best.pth"
Write-Host ""
Write-Host "To test on Doubao images:"
Write-Host "  1. Update app.py checkpoint path"
Write-Host "  2. Set LARE_MODEL_TYPE=sd15 (or keep consistent with extractor)"
Write-Host "  3. Restart web demo"
