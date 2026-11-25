# CNN co dien cung co S1 (MNIST + Fashion-MNIST 16x16)

May RTX 4090, code branch smoke-test (bd4c90f = main + --limit-samples), torch 2.5.1+cu121.
SmallCNN: Conv(1->a) BN ReLU Pool, Conv(a->b) BN ReLU Pool, Linear(16b,b) ReLU, Linear(b,10).
Seed 0, split-seed 20260911, aug xoay +-5 / dich +-0,05, dropout 0, Adam, ReduceLROnPlateau x0,5 patience 3,
dung som patience 10, toi da 50 epoch. MAX = val acc cao nhat; CK = val acc tai val loss thap nhat.

| Dataset | widths (tham so) | lr | MAX val | CK val |
|---|---|---|---:|---:|
| MNIST | 14 20 (9.378) | 3e-3 | 99,25 | 99,20 |
| MNIST | 14 20 | 1e-3 | 99,22 | 99,12 |
| MNIST | 14 20 | 3e-4 | 99,06 | 99,02 |
| MNIST | 10 21 (9.370) | 3e-3 | 99,23 | 99,22 |
| MNIST | 10 21 | 1e-3 | 99,13 | 99,09 |
| MNIST | 10 21 | 3e-4 | 98,99 | 98,96 |
| Fashion | 14 20 | 3e-3 | 90,87 | 90,56 |
| Fashion | 14 20 | 1e-3 | 90,20 | 90,20 |
| Fashion | 14 20 | 3e-4 | 89,98 | 89,90 |
| Fashion | 10 21 | 3e-3 | 90,41 | 90,37 |
| Fashion | 10 21 | 1e-3 | 90,28 | 90,21 |
| Fashion | 10 21 | 3e-4 | 89,22 | 89,14 |

So QCNN S1 (9.389 FREE) MNIST MAX 99,10 = CK: CNN tot nhat hon 0,10-0,15 -> ngang (< 0,30, mot seed).
Moi run: history.csv, summary.json, config.json, convergence.png, train_log.txt (khong co checkpoint .pt).
