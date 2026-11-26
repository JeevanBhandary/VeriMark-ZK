import torch
import torch.nn as nn


class WatermarkEncoder(nn.Module):
    """Embeds a binary message vector into an RGB image tensor in [0, 1]."""

    def __init__(self, msg_bits: int = 32, alpha: float = 0.04) -> None:
        super().__init__()
        self.msg_bits = msg_bits
        self.alpha = alpha

        self.msg_proj = nn.Sequential(
            nn.Linear(msg_bits, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.conv_in = nn.Sequential(
            nn.Conv2d(3 + 64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        msg_feat = self.msg_proj(msg).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        x = torch.cat([image, msg_feat], dim=1)
        residual = self.conv_in(x)
        return torch.clamp(image + self.alpha * residual, 0.0, 1.0)


class WatermarkDecoder(nn.Module):
    """Extracts logits for each payload bit from a possibly distorted image."""

    def __init__(self, msg_bits: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, msg_bits),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.features(image)
        return self.head(feat)