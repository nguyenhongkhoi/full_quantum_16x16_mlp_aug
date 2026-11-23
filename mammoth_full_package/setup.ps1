# One-time setup on a fresh Windows machine with an NVIDIA GPU and internet access.
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# It: finds or installs Python 3.11 (winget), makes .venv, installs PyTorch with CUDA, downloads Mammoth at commit
# e75a491c69fd729edeb01431afb753d9157d9a81, installs its requirements, copies our backbones and scripts into it,
# and runs test_quantum_backbones.py. Safe to run again: finished steps are skipped.
param([switch]$SkipTest)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest very slow on PowerShell 5.1
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$Sha = "e75a491c69fd729edeb01431afb753d9157d9a81"
$Mammoth = Join-Path $Root "mammoth"
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$TorchIndex = "https://download.pytorch.org/whl/cu124"

function Step($text) { Write-Host ""; Write-Host "==== $text ====" -ForegroundColor Cyan }
function Invoke-Checked($exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $exe $($arguments -join ' ')" }
}
function Find-BasePython {
    try {
        $found = & py -3.11 -c "import sys; print(sys.executable)"
        if ($LASTEXITCODE -eq 0 -and $found) { return $found.Trim() }
    } catch {}
    foreach ($candidate in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
                             "C:\Program Files\Python311\python.exe")) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

Step "1/7 GPU driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv }
else { Write-Host "WARNING: nvidia-smi not found. Install the NVIDIA driver first (https://www.nvidia.com/Download/index.aspx)." -ForegroundColor Yellow }

Step "2/7 Python 3.11"
if (-not (Test-Path $Py)) {
    $base = Find-BasePython
    if (-not $base) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "Installing Python 3.11 with winget (current user)..."
            winget install -e --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
            $base = Find-BasePython
        }
    }
    if (-not $base) {
        Write-Host "Python 3.11 not found and could not be installed automatically." -ForegroundColor Red
        Write-Host "Install it from https://www.python.org/downloads/ (version 3.11.x, tick 'Add python.exe to PATH'), then run this script again."
        exit 1
    }
    Write-Host "Base Python: $base"
    Step "3/7 Virtual environment .venv"
    Invoke-Checked $base @("-m", "venv", $Venv)
} else { Write-Host ".venv exists: $Py" }
Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")

Step "4/7 PyTorch with CUDA"
# Probe without failing: on a fresh venv this prints ModuleNotFoundError, which is expected.
$ErrorActionPreference = "Continue"
& $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
$hasTorch = ($LASTEXITCODE -eq 0)
if ($hasTorch) { & $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; $hasTorch = ($LASTEXITCODE -eq 0) }
$ErrorActionPreference = "Stop"
if (-not $hasTorch) {
    Invoke-Checked $Py @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchIndex)
}
Invoke-Checked $Py @("-c", "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')")
& $Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch cannot see the GPU. Check the NVIDIA driver (nvidia-smi) and run again." }

Step "5/7 Mammoth (commit $Sha)"
if (-not (Test-Path (Join-Path $Mammoth "main.py"))) {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $zip = Join-Path $env:TEMP "mammoth_$Sha.zip"
    $tmp = Join-Path $Root "_mammoth_download"
    Write-Host "Downloading https://github.com/aimagelab/mammoth/archive/$Sha.zip"
    Invoke-WebRequest -Uri "https://github.com/aimagelab/mammoth/archive/$Sha.zip" -OutFile $zip -UseBasicParsing
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp
    $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
    Move-Item $inner.FullName $Mammoth
    Remove-Item -Recurse -Force $tmp
} else { Write-Host "Mammoth already at $Mammoth" }
# requirements.txt without the torch lines (torch and torchvision are already the CUDA build)
$req = Join-Path $Root "requirements_mammoth_no_torch.txt"
Get-Content (Join-Path $Mammoth "requirements.txt") | Where-Object { $_ -notmatch '^\s*torch(vision|audio)?\b' } |
    Set-Content -Path $req -Encoding ascii
Invoke-Checked $Py @("-m", "pip", "install", "-r", $req)

Step "6/7 Copy our backbones and scripts into Mammoth"
Copy-Item -Recurse -Force (Join-Path $Root "mammoth_files\backbone\*") (Join-Path $Mammoth "backbone\")
foreach ($file in @("run_mammoth.py", "export_mammoth_logs.py", "test_quantum_backbones.py")) {
    Copy-Item -Force (Join-Path $Root "mammoth_files\$file") $Mammoth
}
Write-Host "Copied."
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "patch_mammoth.ps1")
if ($LASTEXITCODE -ne 0) { throw "patch_mammoth.ps1 failed" }

if (-not $SkipTest) {
    Step "7/7 Checks (no MNIST needed; about 1-3 minutes)"
    Push-Location $Mammoth
    try { Invoke-Checked $Py @("test_quantum_backbones.py", "--device", "cuda:0") } finally { Pop-Location }
}
Set-Content -Path (Join-Path $Root ".setup_done") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding ascii
Write-Host ""
Write-Host "SETUP DONE. Next: powershell -ExecutionPolicy Bypass -File .\run_all.ps1" -ForegroundColor Green
