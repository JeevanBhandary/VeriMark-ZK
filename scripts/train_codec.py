import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attacks import apply_blur_tensor, apply_gaussian_noise_tensor
from src.watermark_net import WatermarkDecoder, WatermarkEncoder


def apply_diff_jpeg_approx(x: torch.Tensor, levels: int = 16) -> torch.Tensor:
    """Smooth straight-through quantization approximation for JPEG resilience during training."""
    if levels <= 1:
        return x
    scale = float(levels - 1)
    q = torch.round(torch.clamp(x, 0.0, 1.0) * scale) / scale
    # Straight-through estimator (STE): forward quantized, backward identity
    return torch.clamp(x + (q - x).detach(), 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight watermark encoder/decoder.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--msg-bits", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--img-weight", type=float, default=25.0)
    parser.add_argument("--msg-weight", type=float, default=10.0)
    parser.add_argument("--jpeg-aug", action="store_true", help="Enable STE quantization noise for JPEG resilience.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="outputs/checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | alpha={args.alpha} | msg_bits={args.msg_bits}")

    transform = T.Compose([T.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    encoder = WatermarkEncoder(msg_bits=args.msg_bits, alpha=args.alpha).to(device)
    decoder = WatermarkDecoder(msg_bits=args.msg_bits).to(device)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        total_bit_acc = 0.0

        for images, _ in loader:
            images = images.to(device)
            bsz = images.size(0)
            msgs = torch.randint(0, 2, (bsz, args.msg_bits), device=device, dtype=torch.float32)

            watermarked = encoder(images, msgs)
            attacked = watermarked

            if torch.rand(1).item() < 0.7:
                attacked = apply_gaussian_noise_tensor(attacked, sigma=0.015)
            if torch.rand(1).item() < 0.4:
                attacked = apply_blur_tensor(attacked, kernel_size=3)
            if args.jpeg_aug and torch.rand(1).item() < 0.6:
                levels = int(torch.randint(12, 33, (1,)).item())
                attacked = apply_diff_jpeg_approx(attacked, levels=levels)

            pred_logits = decoder(attacked)
            loss_img = mse_loss(watermarked, images)
            loss_msg = bce_loss(pred_logits, msgs)
            loss = args.img_weight * loss_img + args.msg_weight * loss_msg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_bits = (torch.sigmoid(pred_logits) > 0.5).float()
                bit_acc = (pred_bits == msgs).float().mean().item()

            total_loss += loss.item()
            total_bit_acc += bit_acc

        scheduler.step()
        n_batches = len(loader)
        curr_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"LR: {curr_lr:.6f} | "
            f"Loss: {total_loss / n_batches:.4f} | "
            f"Train Bit Acc: {total_bit_acc / n_batches:.4f}"
        )

    ckpt_path = save_dir / "verimark_baseline.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "msg_bits": args.msg_bits,
            "alpha": args.alpha,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()