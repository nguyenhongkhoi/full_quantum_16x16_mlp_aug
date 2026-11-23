# Smoke test: check that the code runs end to end in a few minutes (NOT a real training run).
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File .\smoke_test.ps1
# Uses only .\.venv (never the machine's shared Python). Outputs go to .\runs\_smoke and
# .\classical_cnn\runs\_smoke (old smoke outputs are overwritten; real runs are never touched).
# Each step: 2 epochs on 256 train + 256 validation images. Prints PASS / FAIL per step.
param(
    [string]$Device = "cuda",
    [int]$Limit = 256,
    [switch]$SkipTestModel
)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "Missing $py (create the venv first, see command.txt)"; exit 1 }
Set-Location $root
New-Item -ItemType Directory -Force -Path .\runs\_smoke, .\logs | Out-Null
$results = @()

function Invoke-Step([string]$name, [string[]]$arguments) {
    $log = ".\logs\smoke_$name.txt"
    Write-Host "=== $name ==="
    $start = Get-Date
    & $py @arguments *> $log
    $code = $LASTEXITCODE
    $sec = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    $status = if ($code -eq 0) { "PASS" } else { "FAIL" }
    Write-Host "$status ($sec s) -> $log"
    if ($code -ne 0) { Get-Content $log -Tail 15 | ForEach-Object { Write-Host "    $_" } }
    $script:results += [pscustomobject]@{ Step = $name; Status = $status; Seconds = $sec; Log = $log }
}

$common = @("--epochs", "2", "--limit-samples", "$Limit", "--batch-size", "64", "--patience", "10", "--seed", "0",
            "--split-seed", "20260911", "--aug-degrees", "5", "--aug-translate", "0.05", "--dropout", "0",
            "--device", $Device, "--data-root", ".\mnist_data")
$s1 = @("--front", "qconv4x3", "--orders", "row", "column", "morton", "--side-level", "32", "--expert-head", "samebasis",
        "--router", "quantum", "--router-source", "encoder", "--router-grid", "4", "4", "--router-qubits", "8",
        "--router-layers", "1", "--router-ansatz", "se", "--router-init-scale", "0.01", "--mix", "logits",
        "--gate-noise", "0", "--balance-weight", "0.01", "--entropy-weight", "0", "--lr", "1e-3")

if (-not $SkipTestModel) { Invoke-Step "test_model" @(".\test_model.py") }
Invoke-Step "S1_mnist"   (@(".\train.py", "--dataset", "mnist") + $s1 + $common + @("--out-dir", ".\runs\_smoke\S1_mnist"))
Invoke-Step "S1_fashion" (@(".\train.py", "--dataset", "fashion") + $s1 + $common + @("--out-dir", ".\runs\_smoke\S1_fashion"))
Invoke-Step "G2_mnist"   (@(".\train.py", "--dataset", "mnist") + $s1 + @("--share-gates", "--share-final") + $common + @("--out-dir", ".\runs\_smoke\G2_mnist"))
Invoke-Step "CNN_mnist"   (@(".\classical_cnn\run_cnn.py", "--dataset", "mnist", "--widths", "14", "20", "--lr", "1e-3", "--weight-decay", "0", "--lr-factor", "0.5", "--lr-patience", "3", "--min-lr", "1e-6") + $common + @("--out-dir", ".\classical_cnn\runs\_smoke\CNN_mnist"))
Invoke-Step "CNN_fashion" (@(".\classical_cnn\run_cnn.py", "--dataset", "fashion", "--widths", "14", "20", "--lr", "1e-3", "--weight-decay", "0", "--lr-factor", "0.5", "--lr-patience", "3", "--min-lr", "1e-6") + $common + @("--out-dir", ".\classical_cnn\runs\_smoke\CNN_fashion"))
Invoke-Step "CNN_evaluate_test" @(".\classical_cnn\evaluate_test.py", "--checkpoint", ".\classical_cnn\runs\_smoke\CNN_fashion\model_best_val_acc.pt", "--data-root", ".\mnist_data", "--device", $Device)
Invoke-Step "plot_all" @(".\plot_all.py", "--runs", ".\runs\_smoke", ".\classical_cnn\runs\_smoke", "--out-dir", ".\runs\_smoke\figures")

Write-Host ""
$results | Format-Table -AutoSize | Out-String | Write-Host
$failed = @($results | Where-Object { $_.Status -eq "FAIL" }).Count
Write-Host "SMOKE TEST: $($results.Count - $failed) PASS, $failed FAIL"
exit $failed
