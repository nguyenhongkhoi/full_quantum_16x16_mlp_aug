# Sweep CNN sizes x learning rates, one after another (seed 0). Run from inside classical_cnn_16x16:
#   powershell -ExecutionPolicy Bypass -File .\run_sweep.ps1
# Runs that already have summary.json are skipped, so running the script again resumes the sweep.
# Edit $presets / $lrs / $seed below to run fewer.
$env:PYTHONIOENCODING = "utf-8"
$presets = @("cnn1k", "cnn2k", "cnn3k", "cnn5k", "cnn12k", "cnn17k", "cnn19k", "cnn30k", "cnn50k", "cnn85k")
$lrs = @("3e-3", "1e-3", "3e-4")
$seed = 0
New-Item -ItemType Directory -Force -Path .\logs | Out-Null
foreach ($preset in $presets) {
    foreach ($lr in $lrs) {
        $name = "${preset}_lr${lr}_seed${seed}"
        $out = ".\runs\$name"
        if (Test-Path "$out\summary.json") { Write-Host "Skip $name (done)"; continue }
        Write-Host "=== $name ==="
        python .\run_cnn.py --preset $preset --lr $lr --seed $seed --dropout 0 --weight-decay 0 --aug-degrees 5 --aug-translate 0.05 --batch-size 64 --epochs 50 --patience 10 --lr-factor 0.5 --lr-patience 3 --min-lr 1e-6 --split-seed 20260911 --device cuda --data-root .\mnist_data --out-dir $out | Tee-Object -FilePath ".\logs\log_$name.txt"
    }
}
python .\tong_hop.py
