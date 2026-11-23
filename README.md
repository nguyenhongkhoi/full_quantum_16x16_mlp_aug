# linhead

Encoder qconv4x3 + 3 cay level32, head moi cay Linear(372,10), trung binh deu (L1); --model cnn3 doi chung.

**Ket qua:** L1 ~98 (nguoi dung bao, chua co CSV).

- Folder goc: `full_quantum_16x16_level32_qconv_linhead`
- Lenh chay day du: `command.txt`
- Mo ta chi tiet: `README_CHI_TIET.md`

MNIST 16x16, seed 0, split-seed 20260911, aug (xoay +-5, dich +-0,05), `--dropout 0`. So tham so bao theo FREE (unitary 2 qubit 16, 1 qubit 4).