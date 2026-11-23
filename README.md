# level32-qconv

Run 2c: encoder qconv4x3 (quanvolution 3 tang) -> 3 cay level32 -> head MLP gop (front_qconv4x3_proj10). Co them deep6p/deep12p (encoder CNN).

**Ket qua:** 2c MAX val 99,03 (ep 29), 14.864 FREE (dem cu 17.264).

- Folder goc: `full_quantum_16x16_level32_qconv`
- Lenh chay day du: `command.txt`
- Mo ta chi tiet: `README_CHI_TIET.md`

MNIST 16x16, seed 0, split-seed 20260911, aug (xoay +-5, dich +-0,05), `--dropout 0`. So tham so bao theo FREE (unitary 2 qubit 16, 1 qubit 4).