import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks import (
    apply_blur_tensor,
    apply_gaussian_noise_tensor,
    apply_splice_tamper,
    jpeg_compress_pil,
)
from src.crypto_prov import ProvenanceAuthority
from src.metrics import bit_error_rate, compute_psnr
from src.watermark_net import WatermarkDecoder, WatermarkEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep max_bit_errors tolerance across attacks.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/verimark_baseline.pt")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tolerances", type=str, default="0,2,4,8,12")
    parser.add_argument("--out-csv", type=str, default="outputs/results/ablation_tolerance.csv")
    return parser.parse_args()


def logits_to_bits_str(logits: torch.Tensor) -> list[str]:
    bits = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()
    return ["".join(str(int(b)) for b in row) for row in bits]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tolerances = [int(x.strip()) for x in args.tolerances.split(",") if x.strip()]

    ckpt = torch.load(args.checkpoint, map_location=device)
    msg_bits = int(ckpt.get("msg_bits", 32))
    alpha = float(ckpt.get("alpha", 0.04))

    encoder = WatermarkEncoder(msg_bits=msg_bits, alpha=alpha).to(device)
    decoder = WatermarkDecoder(msg_bits=msg_bits).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    transform = T.Compose([T.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    n_samples = min(args.num_samples, len(test_set))
    images = torch.stack([test_set[i][0] for i in range(n_samples)], dim=0).to(device)

    authority = ProvenanceAuthority()
    records = [
        authority.issue_payload(asset_id=f"cifar10_test_{i:04d}", model_id="verimark-codec-v1", bit_length=msg_bits)
        for i in range(n_samples)
    ]
    msg_tensor = torch.tensor(
        [[int(ch) for ch in rec.payload_bits] for rec in records],
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        watermarked = torch.cat(
            [encoder(images[s : min(s + args.batch_size, n_samples)], msg_tensor[s : min(s + args.batch_size, n_samples)])
             for s in range(0, n_samples, args.batch_size)],
            dim=0,
        )

    orig_uint8 = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    wm_uint8 = (watermarked.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    mean_psnr = float(np.mean([compute_psnr(orig_uint8[i], wm_uint8[i]) for i in range(n_samples)]))
    wm_np = watermarked.cpu().numpy().transpose(0, 2, 3, 1)

    attacks = {
        "clean": watermarked.clone(),
        "noise_0.02": apply_gaussian_noise_tensor(watermarked, sigma=0.02),
        "blur_3x3": apply_blur_tensor(watermarked, kernel_size=3),
        "jpeg_q75": torch.from_numpy(
            np.stack([jpeg_compress_pil(wm_np[i], quality=75) for i in range(n_samples)], axis=0).transpose(0, 3, 1, 2)
        ).to(device=device, dtype=torch.float32),
        "jpeg_q50": torch.from_numpy(
            np.stack([jpeg_compress_pil(wm_np[i], quality=50) for i in range(n_samples)], axis=0).transpose(0, 3, 1, 2)
        ).to(device=device, dtype=torch.float32),
        "splice_25pct": torch.from_numpy(
            np.stack([apply_splice_tamper(wm_np[i], box_ratio=0.25) for i in range(n_samples)], axis=0).transpose(0, 3, 1, 2)
        ).to(device=device, dtype=torch.float32),
    }

    rows = []
    for cond_name, atk_imgs in attacks.items():
        with torch.no_grad():
            logits = torch.cat(
                [decoder(atk_imgs[s : min(s + args.batch_size, n_samples)]) for s in range(0, n_samples, args.batch_size)],
                dim=0,
            )
        pred_bits_list = logits_to_bits_str(logits)
        mean_ber = float(np.mean([bit_error_rate(records[i].payload_bits, pred_bits_list[i]) for i in range(n_samples)]))

        for tol in tolerances:
            ver_rate = float(
                np.mean(
                    [
                        1.0 if authority.verify_payload(records[i], pred_bits_list[i], max_bit_errors=tol)["verified"] else 0.0
                        for i in range(n_samples)
                    ]
                )
            )
            rows.append(
                {
                    "condition": cond_name,
                    "max_bit_errors": tol,
                    "max_ber_allowed": round(tol / msg_bits, 4),
                    "mean_psnr_db": round(mean_psnr, 4),
                    "mean_ber": round(mean_ber, 4),
                    "verify_pass_rate": round(ver_rate, 4),
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved tolerance ablation ({len(rows)} rows, PSNR={mean_psnr:.2f} dB) to {out_path}")
    for tol in tolerances:
        sub = {r["condition"]: r["verify_pass_rate"] for r in rows if r["max_bit_errors"] == tol}
        print(
            f"tol={tol:2d} (BER<={tol/msg_bits:.2f}) | "
            f"clean={sub['clean']:.3f} | noise={sub['noise_0.02']:.3f} | "
            f"jpeg75={sub['jpeg_q75']:.3f} | splice={sub['splice_25pct']:.3f}"
        )


if __name__ == "__main__":
    main()