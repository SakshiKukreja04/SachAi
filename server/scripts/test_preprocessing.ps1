param(
    [Parameter(Mandatory=$true)]
    [string]$VideoPath,
    
    [Parameter(Mandatory=$false)]
    [string]$JobId = "test-job",
    
    [Parameter(Mandatory=$false)]
    [string]$OutDir = "./tmp"
)

$ErrorActionPreference = "Continue"
$JobDir = Join-Path $OutDir $JobId

Write-Host "Preprocessing Pipeline Test" -ForegroundColor Green
Write-Host "Video: $VideoPath" -ForegroundColor Cyan
Write-Host "Output: $JobDir" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $VideoPath)) {
    Write-Host "ERROR: Video file not found" -ForegroundColor Red
    exit 1
}

Write-Host "[1/5] Checking dependencies..." -ForegroundColor Yellow
try {
    python --version *>$null
    ffmpeg -version *>$null 2>&1
    Write-Host "OK: Dependencies found" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Missing ffmpeg or python" -ForegroundColor Red
    exit 1
}

Write-Host "[2/5] Running preprocessing..." -ForegroundColor Yellow
python ml/face_preprocess.py --video "$VideoPath" --out "$OutDir" --jobId "$JobId" --fps 1 --workers 4

$FramesDir = Join-Path $JobDir "frames"
$FacesDir = Join-Path $JobDir "faces"
$LandmarksJson = Join-Path $JobDir "landmarks.json"

Write-Host "[3/5] Validating output..." -ForegroundColor Yellow
$FrameCount = @(Get-ChildItem $FramesDir -Filter "frame_*.jpg" -ErrorAction SilentlyContinue).Count
$FaceCount = @(Get-ChildItem $FacesDir -Filter "face_*.jpg" -ErrorAction SilentlyContinue).Count

Write-Host "Frames: $FrameCount, Faces: $FaceCount" -ForegroundColor Green

Write-Host "[4/5] Checking JSON..." -ForegroundColor Yellow
if (Test-Path $LandmarksJson) {
    $json = Get-Content $LandmarksJson | ConvertFrom-Json
    Write-Host "JSON valid - jobId: $($json.jobId)" -ForegroundColor Green
} else {
    Write-Host "ERROR: landmarks.json not found" -ForegroundColor Red
    exit 1
}

Write-Host "[5/5] Done" -ForegroundColor Green
Write-Host ""
Write-Host "Output locations:" -ForegroundColor Cyan
Write-Host "  Frames: $FramesDir"
Write-Host "  Faces: $FacesDir"
Write-Host "  Landmarks: $LandmarksJson"
