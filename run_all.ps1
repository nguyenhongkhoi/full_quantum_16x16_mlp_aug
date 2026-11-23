# Run the whole Mammoth rot-mnist queue, one run after another. Runs setup.ps1 first if it has not been done.
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Only cnn-17k,qcnn-fusion-level32-qconv4x3
# Same configuration as all earlier project runs: ER, buffer 5120, Adam lr 1e-3, batch 32, replay minibatch 32,
# 1 epoch per task, 20 tasks, seed 0, validation 10% (complete), no augmentation, other metrics on.
# A run whose log has the final line is skipped; an unfinished log is renamed and the run starts again from task 1.
# After every run the CSVs are exported to .\ket_qua\.
param([string[]]$Only, [switch]$SkipSmoke)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$Mammoth = Join-Path $Root "mammoth"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Results = Join-Path $Root "ket_qua"

if (-not (Test-Path (Join-Path $Root ".setup_done"))) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "setup.ps1")
    if ($LASTEXITCODE -ne 0) { throw "setup.ps1 failed" }
}
# open_clip files, loggers.py None fix, pandas (also needed when setup.ps1 ran before this patch existed)
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "patch_mammoth.ps1")
if ($LASTEXITCODE -ne 0) { throw "patch_mammoth.ps1 failed" }

# Order: fast classical runs first, then the quantum model (slow), then its classical-encoder control.
$Queue = @("cnn-5k", "cnn-12k", "cnn-17k", "cnn-19k", "cnn-30k", "cnn-50k", "cnn-85k", "mlp-19k",
           "qcnn-fusion-level32-qconv4x3", "qcnn-fusion-level32-deep6p")
if ($Only) { $Queue = $Only | ForEach-Object { $_ -split "," } | Where-Object { $_ } }

$Common = @("--dataset", "rot-mnist", "--model", "er", "--optimizer", "adam", "--lr", "1e-3", "--batch_size", "32",
            "--n_epochs", "1", "--seed", "0", "--device", "cuda:0", "--validation", "10",
            "--validation_mode", "complete", "--enable_other_metrics", "1", "--num_workers", "0",
            "--buffer_size", "5120", "--minibatch_size", "32")

New-Item -ItemType Directory -Force -Path $Results | Out-Null
Push-Location $Mammoth
try {
    New-Item -ItemType Directory -Force -Path "logs" | Out-Null
    if (-not $SkipSmoke -and -not (Test-Path (Join-Path $Root ".smoke_done"))) {
        Write-Host "==== Smoke test: a few steps of the quantum model (downloads MNIST the first time) ====" -ForegroundColor Cyan
        & $Py main.py @Common --backbone qcnn-fusion-level32-qconv4x3 --debug_mode 1
        if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }
        Set-Content -Path (Join-Path $Root ".smoke_done") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding ascii
    }
    foreach ($backbone in $Queue) {
        $name = ($backbone -replace "-", "") + "_er_buf5120_seed0_full"
        $log = Join-Path "logs" "log_$name.txt"
        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "Accuracy for final task" -Quiet)) {
            Write-Host "Skip $backbone (finished: $log)"
            continue
        }
        if (Test-Path $log) {
            $old = Join-Path "logs" ("log_$name.incomplete_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".txt")
            Move-Item $log $old
            Write-Host "Unfinished log moved to $old; starting $backbone again from task 1."
        }
        Write-Host ""
        Write-Host "==== $backbone  ($(Get-Date -Format 'yyyy-MM-dd HH:mm')) -> $log ====" -ForegroundColor Cyan
        & $Py run_mammoth.py --name $name @Common --backbone $backbone
        if ($LASTEXITCODE -ne 0) { Write-Host "Run $backbone ended with exit code $LASTEXITCODE; continuing." -ForegroundColor Yellow }
        & $Py export_mammoth_logs.py
        foreach ($csv in @("mammoth_mean.csv", "mammoth_tasks.csv")) {
            if (Test-Path $csv) { Copy-Item -Force $csv $Results }
        }
        Copy-Item -Force $log $Results
    }
} finally { Pop-Location }
Write-Host ""
Write-Host "ALL DONE. Results: $Results (mammoth_mean.csv, mammoth_tasks.csv, logs)." -ForegroundColor Green
