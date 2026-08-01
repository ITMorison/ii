<#
.SYNOPSIS
Build OpenPose and install MMAction2 for the Security Monitoring project.
#>

param(
    [switch]$ForceOpenPose,
    [switch]$SkipMMAction2,
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
$BaseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenPoseDir = Join-Path $BaseDir 'openpose-master'
$MMActionDir = Join-Path $BaseDir 'mmaction2-main'
$VenvPython = Join-Path $BaseDir 'venv\Scripts\python.exe'
$VenvPip = Join-Path $BaseDir 'venv\Scripts\pip.exe'

function Write-Step($msg) { Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[+] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "[-] $msg" -ForegroundColor Red }

if (-not (Test-Path $VenvPython)) {
    Write-Fail "Virtual environment not found: $VenvPython"
    exit 1
}

Set-Location $BaseDir

# --- 1. Check tools ---
Write-Step "Checking tools..."

$tools = @{ 'git'='git --version'; 'cmake'='cmake --version'; 'python'="$VenvPython --version" }
foreach ($t in $tools.Keys) {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) {
        Write-Fail "$t not found in PATH. Install and retry."
        exit 1
    }
}

# Check Visual Studio (MSVC)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = ""
if (Test-Path $vswhere) {
    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
}
if (-not $vsInstall) {
    Write-Fail "Visual Studio (Desktop development with C++) not found. Install VS Build Tools."
    exit 1
}
Write-Ok "Visual Studio: $vsInstall"

# Find nvcc (CUDA)
$useCuda = $false
if ($Cuda) {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $useCuda = $true
        Write-Ok "CUDA found: $($nvcc.Source)"
    } else {
        Write-Host "[!] CUDA not found, will build CPU-only" -ForegroundColor Yellow
    }
} else {
    Write-Host "[i] Flag -Cuda:`$false, building CPU-only" -ForegroundColor Yellow
}

# --- 2. Build OpenPose ---
if (-not $ForceOpenPose -and (Test-Path (Join-Path $OpenPoseDir 'python\openpose\Release'))) {
    Write-Ok "OpenPose already built (Release exists), skipping build."
} else {
    Write-Step "Building OpenPose..."
    if (-not (Test-Path $OpenPoseDir)) {
        Write-Fail "Folder openpose-master not found: $OpenPoseDir"
        exit 1
    }

    $BuildDir = Join-Path $OpenPoseDir 'build'
    if (Test-Path $BuildDir) { Remove-Item $BuildDir -Recurse -Force }
    New-Item -Path $BuildDir -ItemType Directory | Out-Null
    Set-Location $BuildDir

    $cmakeArgs = @(
        '..',
        '-A', 'x64',
        '-DCMAKE_BUILD_TYPE=Release'
    )
    if ($useCuda) {
        $cmakeArgs += '-DCUDA_USE_STATIC_CUDA_RUNTIME=OFF'
    } else {
        $cmakeArgs += '-DBUILD_PYTHON=ON', '-DDOWNLOAD_BODY_25_MODEL=ON', '-DDOWNLOAD_HAND_MODEL=OFF', '-DDOWNLOAD_FACE_MODEL=OFF'
    }

    Write-Host "CMake: $($cmakeArgs -join ' ')"
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Fail "cmake failed"; exit 1 }

    # Build Release
    & cmake --build . --config Release --target INSTALL
    if ($LASTEXITCODE -ne 0) { Write-Fail "OpenPose build failed"; exit 1 }

    Write-Ok "OpenPose built."
    Set-Location $BaseDir
}

# Verify openpose.pyd exists
$OpenPosePyd = Join-Path $OpenPoseDir 'python\openpose\Release\openpose.pyd'
if (-not (Test-Path $OpenPosePyd)) {
    Write-Host "[!] openpose.pyd not found. OpenPose will be marked unavailable." -ForegroundColor Yellow
} else {
    Write-Ok "openpose.pyd found: $OpenPosePyd"
}

# Add Release path to PYTHONPATH for current session
$env:PYTHONPATH = "$OpenPoseDir\python\openpose\Release;$env:PYTHONPATH"

# --- 3. Install MMAction2 ---
if ($SkipMMAction2) {
    Write-Host "[i] Skipping MMAction2 install (flag -SkipMMAction2)."
} else {
    Write-Step "Installing MMAction2 and dependencies..."

    # Install base packages
    $packages = @(
        'wheel',
        'setuptools',
        'numpy',
        'torch',
        'torchvision',
        'torchaudio',
        'opencv-python',
        'Pillow',
        'scipy',
        'tqdm',
        'yapf',
        'timm',
        'fsspec'
    )

    foreach ($pkg in $packages) {
        Write-Host "  pip install $pkg"
        & $VenvPip install $pkg -q
    }

    # mmengine
    Write-Host "  pip install mmengine"
    & $VenvPip install mmengine -q

    # mmcv-full -- try prebuilt wheel for Windows torch2.1 cu121/cpu
    Write-Host "  pip install mmcv-full"
    $mmcvResult = & $VenvPip install 'mmcv-full==2.1.0' -f 'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html' -q 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [i] mmcv-full cu121 failed, trying CPU wheel..." -ForegroundColor Yellow
        & $VenvPip install 'mmcv-full==2.1.0' -f 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html' -q
    }

    # decord
    Write-Host "  pip install decord"
    & $VenvPip install decord -q

    # einops
    Write-Host "  pip install einops"
    & $VenvPip install einops -q

    # mmaction2
    Write-Host "  pip install -e $MMActionDir"
    & $VenvPip install -e $MMActionDir -q
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Failed to install MMAction2. Check dependencies above."
        exit 1
    }
    Write-Ok "MMAction2 installed."
}

# --- 4. Verify imports ---
Write-Step "Verifying imports..."
$testScript = @'
import sys, os
sys.path.insert(0, r'__OPENPOSE_DIR__')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'__OPENPOSE_DIR__\python\openpose\Release')
try:
    import openpose
    print("[OK] openpose imported")
except Exception as e:
    print("[WARN] openpose unavailable:", repr(e))

try:
    import mmaction
    print("[OK] mmaction imported")
except Exception as e:
    print("[WARN] mmaction unavailable:", repr(e))
'@
$testScript = $testScript -replace '__OPENPOSE_DIR__', [regex]::Escape($OpenPoseDir)
$testScript | Out-File -FilePath (Join-Path $BaseDir '_test_imports.py') -Encoding utf8
& $VenvPython (Join-Path $BaseDir '_test_imports.py')
Remove-Item (Join-Path $BaseDir '_test_imports.py')

Write-Ok "Done."
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Launch Streamlit:   .\venv\Scripts\streamlit.exe run app.py"
Write-Host "  2. If OpenPose is unavailable -- build it via CMake/VS (see OpenPose README)."
Write-Host "  3. If MMAction2 is unavailable -- check that mmengine and mmcv are installed."
