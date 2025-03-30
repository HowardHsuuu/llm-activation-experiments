# analysis/diff_matrix.py
import numpy as np

def compute_diff_matrix(hidden_states1: np.ndarray, hidden_states2: np.ndarray, top_k: int, alpha: float = 1.0) -> np.ndarray:
    """
    hidden_states1, hidden_states2: (num_layers, hidden_size) 的 numpy array
    計算兩組 hidden state 的差值，然後每層僅保留絕對值最大的 top_k 差值，其餘置零，
    最後乘上 alpha。
    """
    diff = hidden_states1 - hidden_states2
    num_layers, hidden_size = diff.shape
    diff_top = np.zeros_like(diff)
    for i in range(num_layers):
        layer_diff = diff[i]
        top_indices = np.argsort(np.abs(layer_diff))[-top_k:]
        mask = np.zeros_like(layer_diff, dtype=bool)
        mask[top_indices] = True
        diff_top[i] = layer_diff * mask
    return diff_top * alpha
