# main

BAN CHINH S1: encoder quanvolution qconv4x3 -> 3 cay QCNN level32 (quet row/column/Morton) -> head samebasis Linear(204,10) moi cay -> router luong tu 8 qubit tron logit. Lenh S1: --expert-head samebasis. Cung folder co cac tuy chon M1 (linear), N1 (nocorr), H1/H2, S3 (--share-final).

**Ket qua:** S1 MAX val 99,10 (ep 46), 9.389 tham so FREE (bo pha 9.135), quantum 32,7%. M1 99,03 (14.429); N1 98,42 (6.869).

- Folder goc: `fullquantum_16x16_frontQuan4x3_Moe`
- Lenh chay day du: `command.txt`
- Mo ta chi tiet: `README_CHI_TIET.md`

MNIST 16x16, seed 0, split-seed 20260911, aug (xoay +-5, dich +-0,05), `--dropout 0`. So tham so bao theo FREE (unitary 2 qubit 16, 1 qubit 4).