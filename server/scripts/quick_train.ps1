# Quick training script for FaceForensics dataset (PowerShell)

param(
    [string]$DataDir = "./data/faceforensics",
    [string]$CheckpointOut = "./checkpoint.pth",
    [int]$Epochs = 10,
    [int]$BatchSize = 32
)

Write-Host "=========================================="
Write-Host "FaceForensics Training Script"
Write-Host "=========================================="
Write-Host "Data directory: $DataDir"
Write-Host "Checkpoint output: $CheckpointOut"
Write-Host "Epochs: $Epochs"
Write-Host "Batch size: $BatchSize"
Write-Host "=========================================="

# Check if data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Host "ERROR: Data directory not found: $DataDir" -ForegroundColor Red
    Write-Host "Please provide a valid path to your FaceForensics dataset"
    exit 1
}

# Run training
python train/train_faceforensics.py `
    --data_dir "$DataDir" `
    --checkpoint_out "$CheckpointOut" `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --lr 1e-4 `
    --val_split 0.2 `
    --freeze_backbone

Write-Host ""
Write-Host "=========================================="
Write-Host "Training completed!"
Write-Host "Checkpoint saved to: $CheckpointOut"
Write-Host "=========================================="

