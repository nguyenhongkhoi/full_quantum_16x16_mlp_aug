"""Core: mạch cây tensor network (phần code chép nguyên từ full_quantum_16x16_level32_qconv/models).

unitary.py      unitary học được U = exp(iH) (1 và 2 qubit)
tn_core.py      khối conv-pool (ghép cặp, U_c, U_p, lấy vết), các lớp strongly-entangling, QuantumTNCore
reupload.py     ReuploadQuantumTNCore: re-upload ở 8 qubit cuối; forward_levels giữ lại một tầng của cây
pauli.py        đặc trưng Pauli chính xác của ma trận mật độ 8 qubit cuối
measurement.py  vector Bloch của một tầng, kích thước đặc trưng
scan_orders.py  thứ tự lá: hàng / cột / Morton
"""
from .measurement import BRANCH_FEATURES, FINAL_QUBITS, PAIRS, PauliFeatures, bloch
from .reupload import ReuploadQuantumTNCore
from .scan_orders import SPATIAL_ORDERS, spatial_permutation

__all__ = ["ReuploadQuantumTNCore", "PauliFeatures", "bloch", "spatial_permutation", "SPATIAL_ORDERS",
           "BRANCH_FEATURES", "FINAL_QUBITS", "PAIRS"]
