import math
import numpy as np


def bit_error_rate(expected_bits: str, predicted_bits: str) -> float:
    if not expected_bits:
        return 0.0
    n = min(len(expected_bits), len(predicted_bits))
    mismatches = sum(expected_bits[i] != predicted_bits[i] for i in range(n))
    mismatches += abs(len(expected_bits) - len(predicted_bits))
    return mismatches / len(expected_bits)


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = 255.0) -> float:
    """Expects uint8 or float arrays of matching shape."""
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)