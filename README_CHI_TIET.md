# FullQuantum 32x32 padded: three phases + MLP readout + train augmentation

This experiment pads each 28x28 MNIST image with a two-pixel black border on every side, preserving its original pixels in a 32x32 input. It uses the three-phase FullQuantumQCNN recipe, an MLP Pauli readout, and RandomAffine training augmentation. It starts from random initialization; 16x16 checkpoints cannot be loaded because the tensor-network core has two additional pooling blocks.

## What stays the same

- 28x28 grayscale MNIST padded to 32x32 (no resizing), 3 quantum channels, 8 final qubits, seven tensor-network pooling blocks, strongly entangling core, 4 layers, merge rounds 2, and reupload in phase 3.
- Relative to the 16x16 model, this adds two pooling blocks per branch: about 4.1x as many pair operations and 128 additional trainable core parameters per branch.
- Same fixed MNIST train/validation split: 50,000 train and 10,000 validation examples, split seed 20260911. Official test data is not used.
- Same three-phase schedule and Adam settings as the original run:
  - Phase 1: fresh model, Adam LR 2e-3, 20 epochs; carry forward the checkpoint with best validation accuracy.
  - Phase 2: continue from phase-1 best, new Adam LR 2e-4, 5 epochs; carry forward the checkpoint with best validation accuracy.
  - Phase 3: add reupload, continue from phase-2 best, Adam core LR 1e-4/readout LR 1e-3, ReduceLROnPlateau, up to 60 epochs, early stop after 10 epochs without validation-loss improvement; retain best validation-loss checkpoint.
- Batch size 64.

## The two experiment changes

1. Readout per branch: `Linear(276,256) -> LayerNorm(256) -> GELU -> Dropout(0.1) -> Linear(256,10)`.
2. Training only: `RandomAffine(degrees=8, translate=0.05)` before `ToTensor`. Validation remains unaugmented.

## Run on the AnyDesk computer

Copy this entire folder into the AnyDesk project root next to `runs/`, `mnist_data/`, and `venv/`. The command below keeps the original FullQuantum seed, split, batch size, phase lengths, learning rates, patience, and reupload setting; this run changes the readout to MLP and augments train images. From any PowerShell location, run the venv Python by its full path (adjust the root path if the AnyDesk project is elsewhere):

```powershell
D:\tensornetwork_mnist\tensornetwork_mnist\venv\Scripts\python.exe D:\tensornetwork_mnist\tensornetwork_mnist\full_quantum_16x16_mlp_aug\run_mlp_aug.py --models full_quantum --device cuda --seed 0 --split-seed 20260911 `
  --fq-batch-size 64 `
  --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 `
  --fq-refine-epochs 5 --fq-refine-lr 2e-4 `
  --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 --fq-patience 10 --fq-reupload `
  --aug-degrees 8 --aug-translate 0.05 `
  --out-dir D:\tensornetwork_mnist\tensornetwork_mnist\full_quantum_16x16_mlp_aug\runs\mlp_aug_32pad_3phase
```

The script saves separate best and last checkpoints for each phase, phase completion markers, combined and per-phase CSV histories, configuration JSON, split indices, summary, plots, and final best checkpoint under `runs/mlp_aug_32pad_3phase/`. Re-running with the same settings resumes from the last saved epoch. A configuration mismatch is rejected; use `--no-resume` only with a new, empty output directory.

During train and validation, the terminal updates the current batch, running loss/accuracy, and elapsed seconds in place. At the end of each epoch it prints the epoch metrics, both learning rates, elapsed time, and `*best`. The combined CSV and current phase CSV are rewritten after every completed epoch. Phase 3 can early-stop, just like the original run.

The AnyDesk venv should already contain compatible PyTorch and torchvision. `requirements.txt` lists only NumPy and Matplotlib; do not replace the existing CUDA-enabled PyTorch installation for this experiment.

## Controls that keep the comparison with the original fair

- The added two pooling blocks introduce trainable gates, so this is not a parameter-matched comparison with the 16x16 model. The historical 16x16 score in the summary is a reference only.
- Loader shuffle order per phase is the same as in `run_convergence.py`. Augmentation/dropout get their own stream per phase (`seed + 10000*phase` for phases 2 and 3), otherwise phase 2 would replay phase 1's augmented images epoch for epoch.
- RandomAffine resamples bilinearly by default (`--aug-interpolation bilinear`); torchvision's default `nearest` leaves jagged edges that the unaugmented validation digits never have.
- `history_full_quantum_mlp_aug.csv` has the same columns (including `phase`) and UTF-8 encoding as the original `history_full_quantum.csv`.
- `summary_mlp_aug.json` reports the saved checkpoint (phase 3, lowest validation loss). It lists the historical 16x16 score separately; that comparison is descriptive because the input resolution differs. The maximum validation accuracy over all epochs is listed separately and is not the checkpoint metric.
- Train accuracy is measured on augmented images with dropout on, so it is not comparable to the baseline train accuracy (98.58%). Compare validation only.

## Ablations (forAI/plan.txt 15.1B)

Defaults run the experiment (MLP + augmentation). Use a new `--out-dir` for each variant:

- MLP only: `--aug-degrees 0 --aug-translate 0` (augmentation fully off, ToTensor only)
- Augmentation only: `--readout linear`
- No-augmentation 32x32-padded reference: `--readout linear --aug-degrees 0 --aug-translate 0` (not the historical 16x16 baseline).
cd D:\tensornetwork_mnist\tensornetwork_mnist
.\venv\Scripts\Activate.ps1

python .\full_quantum_16x16_mlp_aug\run_mlp_aug.py --models full_quantum --device cuda --seed 0 --split-seed 20260911 --data-root .\mnist_data --fq-batch-size 64 --fq-pretrain-epochs 20 --fq-pretrain-lr 2e-3 --fq-refine-epochs 5 --fq-refine-lr 2e-4 --fq-epochs 60 --fq-core-lr 1e-4 --fq-readout-lr 1e-3 --fq-patience 10 --fq-reupload --aug-degrees 8 --aug-translate 0.05 --out-dir .\full_quantum_16x16_mlp_aug\runs\mlp_aug_3phase
ss
