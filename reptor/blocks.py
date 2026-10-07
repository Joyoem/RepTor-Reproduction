from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_KERNELS = (9, 7, 5, 3, 1)


def _fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse Conv1d + BatchNorm1d using BN running statistics."""
    if conv.bias is None:
        conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
    else:
        conv_bias = conv.bias

    running_std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / running_std
    fused_weight = conv.weight * scale.reshape(-1, 1, 1)
    fused_bias = bn.bias + (conv_bias - bn.running_mean) * scale
    return fused_weight, fused_bias


def _fuse_identity_bn(channels: int, kernel_size: int, bn: nn.BatchNorm1d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Represent identity + BN as an equivalent Conv1d."""
    kernel = torch.zeros((channels, channels, kernel_size), device=bn.weight.device, dtype=bn.weight.dtype)
    center = kernel_size // 2
    idx = torch.arange(channels, device=kernel.device)
    kernel[idx, idx, center] = 1.0

    running_std = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / running_std
    fused_weight = kernel * scale.reshape(-1, 1, 1)
    fused_bias = bn.bias - bn.running_mean * scale
    return fused_weight, fused_bias


def _pad_kernel_to(kernel: torch.Tensor, target_kernel_size: int) -> torch.Tensor:
    """Symmetrically zero-pad a Conv1d kernel to target size."""
    current = kernel.size(-1)
    if current == target_kernel_size:
        return kernel
    if current > target_kernel_size:
        raise ValueError(f"Kernel size {current} is larger than target {target_kernel_size}.")
    diff = target_kernel_size - current
    if diff % 2 != 0:
        raise ValueError("RepTor uses odd kernels; symmetric padding must be integral.")
    pad = diff // 2
    return F.pad(kernel, (pad, pad))


class ConvBN(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    @property
    def conv(self) -> nn.Conv1d:
        return self[0]

    @property
    def bn(self) -> nn.BatchNorm1d:
        return self[1]


class RepTorBlock(nn.Module):
    """Training: multi-branch temporal convs (+ optional identity BN). Deploy: one Conv1d."""

    def __init__(self, in_channels: int, out_channels: int, kmax: int, stride: int = 1, deploy: bool = False) -> None:
        super().__init__()
        if kmax not in _VALID_KERNELS:
            raise ValueError(f"kmax must be one of {_VALID_KERNELS}, got {kmax}.")
        if stride not in (1, 2):
            raise ValueError("This reproduction currently expects stride 1 or 2.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kmax = kmax
        self.stride = stride
        self.deploy = deploy
        self.activation = nn.ReLU(inplace=True)

        if deploy:
            self.reparam_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kmax, stride=stride, padding=kmax // 2, bias=True)
        else:
            branch_kernels = [k for k in _VALID_KERNELS if k <= kmax]
            self.branches = nn.ModuleList([
                ConvBN(in_channels, out_channels, kernel_size=k, stride=stride) for k in branch_kernels
            ])
            self.identity_bn = nn.BatchNorm1d(in_channels) if stride == 1 and in_channels == out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.activation(self.reparam_conv(x))

        out = None
        for branch in self.branches:
            branch_out = branch(x)
            out = branch_out if out is None else out + branch_out
        if self.identity_bn is not None:
            out = out + self.identity_bn(x)
        return self.activation(out)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias

        kernel_sum = None
        bias_sum = None
        for branch in self.branches:
            kernel, bias = _fuse_conv_bn(branch.conv, branch.bn)
            kernel = _pad_kernel_to(kernel, self.kmax)
            kernel_sum = kernel if kernel_sum is None else kernel_sum + kernel
            bias_sum = bias if bias_sum is None else bias_sum + bias

        if self.identity_bn is not None:
            kernel, bias = _fuse_identity_bn(self.in_channels, self.kmax, self.identity_bn)
            kernel_sum = kernel_sum + kernel
            bias_sum = bias_sum + bias
        return kernel_sum, bias_sum

    @torch.no_grad()
    def switch_to_deploy(self) -> "RepTorBlock":
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kmax, stride=self.stride, padding=self.kmax // 2, bias=True).to(device=kernel.device, dtype=kernel.dtype)
        conv.weight.copy_(kernel)
        conv.bias.copy_(bias)
        del self.branches
        del self.identity_bn
        self.reparam_conv = conv
        self.deploy = True
        return self
