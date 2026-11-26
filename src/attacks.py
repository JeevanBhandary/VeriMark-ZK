import io
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def apply_gaussian_noise_tensor(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)


def apply_blur_tensor(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    weight = torch.ones((x.shape[1], 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
    weight = weight / float(kernel_size * kernel_size)
    padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return torch.clamp(F.conv2d(padded, weight, groups=x.shape[1]), 0.0, 1.0)


def jpeg_compress_pil(img_01: np.ndarray, quality: int = 75) -> np.ndarray:
    uint8_img = np.clip(img_01 * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(uint8_img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def apply_splice_tamper(img_01: np.ndarray, box_ratio: float = 0.25) -> np.ndarray:
    out = img_01.copy()
    h, w, _ = out.shape
    bh = max(4, int(h * box_ratio))
    bw = max(4, int(w * box_ratio))
    y0 = h // 4
    x0 = w // 4
    patch = np.flip(out[y0 : y0 + bh, x0 : x0 + bw], axis=1).copy()
    out[y0 : y0 + bh, x0 : x0 + bw] = patch
    return out