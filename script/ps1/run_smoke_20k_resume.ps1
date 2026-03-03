param(
  [int]$Seed = 0,
  [int]$TrainSize = 20000,
  [int]$TestSize = 20000,
  [switch]$Balanced,
  [switch]$ForceResplit,
  [switch]$NoResume,
  [int]$Epoches = 25,
  [int]$ValFreq = 1,
  [string]$ExpName = "sdv5_wmap_v7_smoke20k"
)

# One-click smoke training (train 20k / test 20k) with auto-resume
# Run from project root OR anywhere: .\模型训练\run_smoke_20k_resume.ps1

$ErrorActionPreference = "Stop"

function Get-DotEnvValue {
  param([string]$Path, [string]$Key)
  if (!(Test-Path $Path)) { return $null }
  $line = Select-String -Path $Path -Pattern ("^" + [regex]::Escape($Key) + "=") -SimpleMatch | Select-Object -First 1
  if ($null -eq $line) { return $null }
  return ($line.Line -split "=",2)[1]
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
Set-Location -Path $ProjectRoot

$TrainScriptItem = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "3_模型训练.py" -ErrorAction Stop | Select-Object -First 1
if ($null -eq $TrainScriptItem) { throw "Cannot find 3_模型训练.py under $ProjectRoot" }
$TrainScript = $TrainScriptItem.FullName
$TrainScriptDir = $TrainScriptItem.Directory.FullName

$DotEnvPath = Join-Path $ProjectRoot ".env"

# 默认启用 balanced（不传 -Balanced 也会均衡抽样）
if (-not $PSBoundParameters.ContainsKey('Balanced')) {
  $Balanced = $true
}

$AnnTrain = Join-Path $ProjectRoot "annotation\train_sdv5_20k_smoke.txt"
$AnnTest  = Join-Path $ProjectRoot "annotation\test_sdv5_20k_smoke.txt"

# ----- Step 1: (Re)generate smoke splits if needed -----
if ($ForceResplit -or !(Test-Path $AnnTrain) -or !(Test-Path $AnnTest)) {
  $balancedNote = $(if ($Balanced) { " (balanced)" } else { "" })
  Write-Host "[1/3] Generating smoke splits: train=$TrainSize test=$TestSize seed=$Seed$balancedNote"

  $splitArgs = @(
    ".\\script\\create_smoke_splits.py",
    "--input", ".\\annotation\\train_sdv5.txt",
    "--out_train", ".\\annotation\\train_sdv5_20k_smoke.txt",
    "--out_test", ".\\annotation\\test_sdv5_20k_smoke.txt",
    "--n_train", "$TrainSize",
    "--n_test", "$TestSize",
    "--seed", "$Seed"
  )
  if ($Balanced) { $splitArgs += "--balanced" }

  & python @splitArgs
  if ($LASTEXITCODE -ne 0) { throw "create_smoke_splits.py failed" }
}

# ----- Step 2: Find latest checkpoint for auto-resume -----
$OutBase = Join-Path $TrainScriptDir "output"
$ResumePath = $null

if (!$NoResume -and (Test-Path $OutBase)) {
  # 不依赖 EXP_NAME：找最新的 Exp*_Log_v* 目录下的 latest.pth
  $candidates = Get-ChildItem -Path $OutBase -Directory -Filter "Exp*_Log_v*" -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending
  foreach ($d in $candidates) {
    $p = Join-Path $d.FullName "latest.pth"
    if (Test-Path $p) { $ResumePath = $p; break }
  }
}

if ($ResumePath) {
  Write-Host "[2/3] Auto-resume enabled: $ResumePath"
} else {
  Write-Host "[2/3] No resume checkpoint found; starting fresh"
}

# ----- Step 3: Start training -----
Write-Host "[3/3] Launching training..."

$cmd = @(
  "python",
  $TrainScript,
  "--isTrain", "1",
  "--train_file", ".\\annotation\\train_sdv5_20k_smoke.txt",
  "--val_file", ".\\annotation\\test_sdv5_20k_smoke.txt",
  "--test_file", ".\\annotation\\test_sdv5_20k_smoke.txt",
  "--val_ratio", "0",
  "--seed", "$Seed",
  "--epoches", "$Epoches",
  "--val_freq", "$ValFreq",
  "--exp_name", "$ExpName"
)

if ($ResumePath) {
  $cmd += @("--resume", $ResumePath)
}

& $cmd[0] @($cmd[1..($cmd.Length-1)])
