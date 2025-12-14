# Quick calibration training script for PowerShell
# Usage: .\quick_calibrate.ps1 -CheckpointIn checkpoint.pth -YouTubeUrls @("URL1", "URL2", ...)

param(
    [Parameter(Mandatory=$true)]
    [string]$CheckpointIn,
    
    [Parameter(Mandatory=$true)]
    [string[]]$YouTubeUrls,
    
    [string]$CalibrationDir = "./calibration_data",
    [string]$CheckpointOut = "checkpoint_calibrated.pth",
    [int]$Epochs = 1,
    [int]$BatchSize = 16,
    [double]$LearningRate = 1e-4
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Quick Calibration Training" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Input checkpoint: $CheckpointIn"
Write-Host "YouTube URLs: $($YouTubeUrls.Count) videos"
Write-Host "Output directory: $CalibrationDir"
Write-Host "Output checkpoint: $CheckpointOut"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Prepare data
Write-Host "Step 1: Downloading videos and extracting faces..." -ForegroundColor Yellow
$urlsArg = $YouTubeUrls -join " "
python scripts/prepare_youtube_calibration.py `
    --urls $YouTubeUrls `
    --output_dir $CalibrationDir `
    --frame_interval 30

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to prepare calibration data" -ForegroundColor Red
    exit 1
}

# Step 2: Train calibration
Write-Host ""
Write-Host "Step 2: Running calibration training..." -ForegroundColor Yellow
python train/train_calibration.py `
    --data_dir $CalibrationDir `
    --checkpoint_in $CheckpointIn `
    --checkpoint_out $CheckpointOut `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr $LearningRate

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Calibration training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "[OK] Calibration complete!" -ForegroundColor Green
Write-Host "  Calibrated checkpoint: $CheckpointOut" -ForegroundColor Green
Write-Host ""
Write-Host "To use the calibrated checkpoint:" -ForegroundColor Yellow
Write-Host "  `$env:MODEL_CHECKPOINT='$CheckpointOut'" -ForegroundColor White
Write-Host "  python server.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green

