import argparse
import csv
from pathlib import Path
import sys
import time
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
    parser = argparse.ArgumentParser(description="Benchmark VeriMark-ZK codec and provenance verifier.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/verimark_baseline.pt")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-bit-errors", type=int, default=2)
    parser.add_argument("--out-csv", type=str, default="outputs/results/benchmark_results.csv")
    return parser.parse_args()


def logits_to_bits_str(logits: torch.Tensor) -> list[str]:
    bits = (torch.sigmoid(logits) > 0.5).to(torch.uint8).cpu().numpy()
    return ["".join(str(int(b)) for b in row) for row in bits]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    msg_bits = int(ckpt.get("msg_bits", 32))

    encoder = WatermarkEncoder(msg_bits=msg_bits, alpha=0.08).to(device)
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

    t0 = time.perf_counter()
    watermarked_list = []
    with torch.no_grad():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            wm = encoder(images[start:end], msg_tensor[start:end])
            watermarked_list.append(wm)
    watermarked = torch.cat(watermarked_list, dim=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    encode_ms_per_img = ((time.perf_counter() - t0) * 1000.0) / n_samples

    orig_uint8 = (images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    wm_uint8 = (watermarked.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().clip(0, 255).astype(np.uint8)
    psnr_vals = [compute_psnr(orig_uint8[i], wm_uint8[i]) for i in range(n_samples)]
    mean_psnr = float(np.mean(psnr_vals))

    wm_np = watermarked.detach().cpu().numpy().transpose(0, 2, 3, 1)

    attack_batches: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        attack_batches["clean"] = watermarked.clone()
        attack_batches["noise_0.02"] = apply_gaussian_noise_tensor(watermarked, sigma=0.02)
        attack_batches["blur_3x3"] = apply_blur_tensor(watermarked, kernel_size=3)

        jpeg75 = np.stack([jpeg_compress_pil(wm_np[i], quality=75) for i in range(n_samples)], axis=0)
        attack_batches["jpeg_q75"] = torch.from_numpy(jpeg75.transpose(0, 3, 1, 2)).to(device=device, dtype=torch.float32)

        jpeg50 = np.stack([jpeg_compress_pil(wm_np[i], quality=50) for i in range(n_samples)], axis=0)
        attack_batches["jpeg_q50"] = torch.from_numpy(jpeg50.transpose(0, 3, 1, 2)).to(device=device, dtype=torch.float32)

        spliced = np.stack([apply_splice_tamper(wm_np[i], box_ratio=0.25) for i in range(n_samples)], axis=0)
        attack_batches["splice_25pct"] = torch.from_numpy(spliced.transpose(0, 3, 1, 2)).to(
            device=device, dtype=torch.float32
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\nBenchmark on {n_samples} CIFAR-10 test images | Mean PSNR: {mean_psnr:.2f} dB | Encode: {encode_ms_per_img:.2f} ms/img")
    print(f"{'Condition':<16} | {'Mean BER':<10} | {'Bit Acc':<10} | {'Verify Rate':<12} | {'Decode ms/img':<13}")
    print("-" * 76)

    for cond_name, atk_imgs in attack_batches.items():
        t_dec = time.perf_counter()
        logits_parts = []
        with torch.no_grad():
            for start in range(0, n_samples, args.batch_size):
                end = min(start + args.batch_size, n_samples)
                logits_parts.append(decoder(atk_imgs[start:end]))
        logits = torch.cat(logits_parts, dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        decode_ms_per_img = ((time.perf_counter() - t_dec) * 1000.0) / n_samples

        pred_bits_list = logits_to_bits_str(logits)
        bers = [bit_error_rate(records[i].payload_bits, pred_bits_list[i]) for i in range(n_samples)]
        verified_flags = [
            1.0 if authority.verify_payload(records[i], pred_bits_list[i], max_bit_errors=args.max_bit_errors)["verified"] else 0.0
            for i in range(n_samples)
        ]

        mean_ber = float(np.mean(bers))
        mean_acc = 1.0 - mean_ber
        verify_rate = float(np.mean(verified_flags))

        print(
            f"{cond_name:<16} | {mean_ber:<10.4f} | {mean_acc:<10.4f} | {verify_rate:<12.4f} | {decode_ms_per_img:<13.2f}"
        )
        rows.append(
            {
                "condition": cond_name,
                "num_samples": n_samples,
                "mean_psnr_db": round(mean_psnr, 4),
                "mean_ber": round(mean_ber, 4),
                "mean_bit_acc": round(mean_acc, 4),
                "verify_pass_rate": round(verify_rate, 4),
                "encode_ms_per_img": round(encode_ms_per_img, 4),
                "decode_ms_per_img": round(decode_ms_per_img, 4),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()