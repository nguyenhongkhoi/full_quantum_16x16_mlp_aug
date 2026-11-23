# sharedgate

S1 + --share-gates: khoi 8 qubit cuoi moi lop dung chung 1 unitary 1 qubit + 1 unitary 2 qubit (cay 808 -> 248 tham so). G2 = G1 + --share-final.

**Ket qua:** G1 MAX val 98,88 (nguoi dung bao), 7.709 FREE; dat nguong >= 98,80 so S1. G2 (7.549) chua chay.

- Folder goc: `fullquantum_16x16_frontQuan4x3_Moe_sharedgate`
- Lenh chay day du: `command.txt`
- Mo ta chi tiet: `README_CHI_TIET.md`

MNIST 16x16, seed 0, split-seed 20260911, aug (xoay +-5, dich +-0,05), `--dropout 0`. So tham so bao theo FREE (unitary 2 qubit 16, 1 qubit 4).