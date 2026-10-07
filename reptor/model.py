from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import RepTorBlock


class Stem(nn.Sequential):
    def __init__(self, in_channels: int = 40, out_channels: int = 16) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class RepTor(nn.Module):
    """RepTor KWS network. Expected input shape: [batch, 40, time_frames]."""

    def __init__(self, kernels: Sequence[int], num_classes: int = 12, in_features: int = 40,
                 dropout: float = 0.2, width_mult: float = 1.0, deploy: bool = False) -> None:
        super().__init__()
        if len(kernels) != 6:
            raise ValueError(f"Expected 6 block kernels, got {len(kernels)}.")

        base_channels = [16, 24, 24, 32, 32, 48, 48]
        channels = [max(1, int(round(c * width_mult))) for c in base_channels]
        strides = [2, 1, 2, 1, 2, 1]

        self.stem = Stem(in_channels=in_features, out_channels=channels[0])
        self.blocks = nn.Sequential(*[
            RepTorBlock(channels[i], channels[i + 1], kmax=kmax, stride=stride, deploy=deploy)
            for i, (kmax, stride) in enumerate(zip(kernels, strides))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, MFCC, T] input, got shape {tuple(x.shape)}.")
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)

    @torch.no_grad()
    def switch_to_deploy(self) -> "RepTor":
        for module in self.modules():
            if isinstance(module, RepTorBlock):
                module.switch_to_deploy()
        return self


def reptor9(num_classes: int = 12, in_features: int = 40, dropout: float = 0.2,
            width_mult: float = 1.0, deploy: bool = False) -> RepTor:
    return RepTor([9, 9, 9, 9, 9, 9], num_classes, in_features, dropout, width_mult, deploy)


def reptor_a(num_classes: int = 12, in_features: int = 40, dropout: float = 0.2,
             width_mult: float = 1.0, deploy: bool = False) -> RepTor:
    return RepTor([7, 9, 5, 9, 5, 5], num_classes, in_features, dropout, width_mult, deploy)
