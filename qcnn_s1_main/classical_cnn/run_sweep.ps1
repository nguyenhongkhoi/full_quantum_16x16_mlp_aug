# Sweep classical CNN sizes x learning rates (one after another). Run from inside the classical_cnn folder:
#   powershell -ExecutionPolicy Bypass -File .\run_sweep.ps1                      (MNIST)
#   powershell -ExecutionPolicy Bypass -File .\run_sweep.ps1 -Dataset fashion      (Fashion-MNIST)
#   powershell -ExecutionPolicy Bypass -File .\run_sweep.ps1 -Widths "14 20","13 18" -Lrs "1e-3"
# Runs that already have summary.json are skipped, so running the script again resumes the sweep.
# Default widths = CNN with the same parameter count as the QCNN models (FREE count):
#   "14 20" = 9,378  ~ S1 (9,389)      "10 21" = 9,370 ~ S1
#   "13 18" = 7,708  ~ G1 (7,709)      "12 18" = 7,534 ~ G2 (7,549)
#   "17 25" = 14,389 ~ M1 (14,429)
param(
    [string]$Dataset = "mnist",
    [string[]]$Widths = @("14 20", "10 21", "13 18", "12 18", "17 25"),
    [string[]]$Lrs = @("3e-3", "1e-3", "3e-4"),
    [int]$Seed = 0
)
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path .\logs | Out-Null
foreach ($w in $Widths) {
    $ab = $w.Split(" ")
    foreach ($lr in $Lrs) {
        $name = "${Dataset}_w$($ab[0])_$($ab[1])_lr${lr}_seed${Seed}"
        $out = ".\runs\$name"
        if (Test-Path "$out\summary.json") { Write-Host "Skip $name (done)"; continue }
        Write-Host "=== $name ==="
        python .\run_cnn.py --dataset $Dataset --widths $ab[0] $ab[1] --lr $lr --seed $Seed --dropout 0 --weight-decay 0 --aug-degrees 5 --aug-translate 0.05 --aug-interpolation bilinear --batch-size 64 --epochs 50 --patience 10 --lr-factor 0.5 --lr-patience 3 --min-lr 1e-6 --split-seed 20260911 --device cuda --data-root ..\mnist_data --out-dir $out | Tee-Object -FilePath ".\logs\log_$name.txt"
    }
}
python .\tong_hop.py
