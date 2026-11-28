# S1 tren 4090 (2026-11-25 -> 28)

Code branch smoke-test (bd4c90f), torch 2.5.1+cu121, lenh S1 day du trong runs/*/config.json. 50 epoch, patience 10, dropout 0.

| Run | MAX val | CK |
|---|---:|---:|
| MNIST S1 seed 0 (may cu, CSV ep46) | 99.10 | 99.10 |
| MNIST S1 seed 1 | 99.01 (ep 47) | 98.97 |
| MNIST S1 seed 2 | 98.87 (ep 47) | 98.80 |
| Fashion S1 seed 0 | 90.00 (ep 44) | 90.00 |

CNN cung co (folder cnn_s1_4090/): MNIST 14-20 99.25, Fashion 14-20 90.87.
figures/ = ve lai o may code (bo run smoke). Khong co .pt (de tren 4090: D:\tensornetwork_mnist\qcnn_s1\runs).
