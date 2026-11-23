# Fixes Mammoth (commit e75a491) for rot-mnist on Windows. Safe to run many times; run_all.ps1 and setup.ps1 call it.
#   powershell -ExecutionPolicy Bypass -File .\patch_mammoth.ps1
# 1. datasets/__init__.py imports EVERY file in datasets/. Files that need open_clip (seq_8vision.py, ...) crash the
#    import with "No module named 'open_clip'". They are not used by rot-mnist, so they are renamed to *.py.disabled.
#    Same for models/ files that import open_clip / clip.
# 2. utils/loggers.py line 387 formats the Task-IL value with :.2f, but for rot-mnist (domain-il) it is None, so
#    --enable_other_metrics 1 crashes after task 20 before logs.pyd is written. The line now prints None instead.
# 3. Installs six, requests, pandas into .venv if missing (six: "No module named 'six'" in main.py lecun_fix).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Mammoth = Join-Path $Root "mammoth"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path (Join-Path $Mammoth "main.py"))) { throw "Mammoth not found at $Mammoth. Run setup.ps1 first." }
$Utf8NoBom = New-Object System.Text.UTF8Encoding $false

Write-Host "==== Patch Mammoth ====" -ForegroundColor Cyan

# 1. Disable files that need open_clip / clip
$pattern = '(?m)^\s*(import|from)\s+(open_clip|clip)\b'
foreach ($sub in @("datasets", "models")) {
    $dir = Join-Path $Mammoth $sub
    Get-ChildItem -Path $dir -Filter "*.py" -File | Where-Object { $_.Name -ne "__init__.py" } | ForEach-Object {
        $text = [System.IO.File]::ReadAllText($_.FullName)
        if ($text -match $pattern) {
            $target = $_.FullName + ".disabled"
            if (Test-Path $target) { Remove-Item -Force $target }
            Move-Item -Path $_.FullName -Destination $target
            Write-Host "Disabled $sub\$($_.Name) (needs open_clip)"
        }
    }
}

# 2. utils/loggers.py: Task-IL metric may be None
$loggers = Join-Path $Mammoth "utils\loggers.py"
$old = '{metric:.2f} \t [Task-IL]: {metric_mask_class:.2f}'
$new = '{metric:.2f} \t [Task-IL]: {"None" if metric_mask_class is None else format(metric_mask_class, ".2f")}'
$text = [System.IO.File]::ReadAllText($loggers)
if ($text.Contains($new)) {
    Write-Host "utils\loggers.py already patched"
} elseif ($text.Contains($old)) {
    [System.IO.File]::WriteAllText($loggers, $text.Replace($old, $new), $Utf8NoBom)
    Write-Host "Patched utils\loggers.py (Task-IL None)"
} else {
    Write-Host "WARNING: utils\loggers.py line not found; check it by hand if Backward Transfer printing crashes." -ForegroundColor Yellow
}

# 3. Packages Mammoth imports without try/except but does not list in requirements.txt:
#    six (main.py lecun_fix, used to download MNIST), requests (datasets), pandas (our earlier runs).
foreach ($pkg in @("six", "requests", "pandas")) {
    $ErrorActionPreference = "Continue"
    & $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$pkg') else 1)"
    $has = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
    if (-not $has) {
        & $Py -m pip install $pkg
        if ($LASTEXITCODE -ne 0) { throw "pip install $pkg failed" }
    }
}

# Check: the modules main.py needs must import now
Push-Location $Mammoth
try {
    & $Py -c "import six.moves.urllib, datasets, backbone, models; print('import six, datasets, backbone, models OK')"
    if ($LASTEXITCODE -ne 0) { throw "Import still fails; send the error above." }
} finally { Pop-Location }
Write-Host "Patch done." -ForegroundColor Green
