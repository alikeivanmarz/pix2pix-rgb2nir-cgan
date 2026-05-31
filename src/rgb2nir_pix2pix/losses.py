"""Losses for pix2pix training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Pix2PixLoss:
    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_gradient: float = 0.0,
        lambda_vessel: float = 0.0,
        lambda_highpass: float = 0.0,
    ) -> None:
        self.lambda_l1 = float(lambda_l1)
        self.lambda_gradient = float(lambda_gradient)
        self.lambda_vessel = float(lambda_vessel)
        self.lambda_highpass = float(lambda_highpass)
        self.adv = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.gradient = SobelGradientLoss()
        self.vessel = DarkLineVesselLoss()
        self.highpass = HighPassDetailLoss()

    def discriminator_loss(
        self, real_logits: torch.Tensor, fake_logits: torch.Tensor
    ) -> torch.Tensor:
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)
        return 0.5 * (self.adv(real_logits, real_targets) + self.adv(fake_logits, fake_targets))

    def generator_loss(
        self, fake_logits: torch.Tensor, fake_nir: torch.Tensor, real_nir: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        adv_loss = self.adv(fake_logits, torch.ones_like(fake_logits))
        l1_loss = self.l1(fake_nir, real_nir)
        gradient_loss = self.gradient(fake_nir, real_nir)
        vessel_loss = self.vessel(fake_nir, real_nir)
        highpass_loss = self.highpass(fake_nir, real_nir)
        total = (
            adv_loss
            + self.lambda_l1 * l1_loss
            + self.lambda_gradient * gradient_loss
            + self.lambda_vessel * vessel_loss
            + self.lambda_highpass * highpass_loss
        )
        return total, {
            "g_adv": float(adv_loss.detach().cpu()),
            "g_l1": float(l1_loss.detach().cpu()),
            "g_gradient": float(gradient_loss.detach().cpu()),
            "g_vessel": float(vessel_loss.detach().cpu()),
            "g_highpass": float(highpass_loss.detach().cpu()),
            "g_total": float(total.detach().cpu()),
        }


def to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class SobelGradientLoss(nn.Module):
    """L1 loss between fixed Sobel gradients of predicted and target NIR."""

    def __init__(self) -> None:
        super().__init__()
        gx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        gy = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("gx", gx)
        self.register_buffer("gy", gy)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_u = to_unit_interval(pred)
        target_u = to_unit_interval(target)
        gx = self.gx.to(device=pred.device, dtype=pred.dtype)
        gy = self.gy.to(device=pred.device, dtype=pred.dtype)
        pred_gx = F.conv2d(pred_u, gx, padding=1)
        pred_gy = F.conv2d(pred_u, gy, padding=1)
        target_gx = F.conv2d(target_u, gx, padding=1)
        target_gy = F.conv2d(target_u, gy, padding=1)
        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)


class DarkLineVesselLoss(nn.Module):
    """Differentiable fixed-filter loss for dark line-like vein responses.

    This is not a replacement for Frangi evaluation. It is a lightweight
    training signal that encourages the predicted NIR to preserve dark ridges in
    the paired target while remaining fully differentiable on MPS.
    """

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                # vertical dark line: bright left/right, dark center column
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, -4.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                # horizontal dark line: bright above/below, dark center row
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, -4.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                # main diagonal dark line
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -4.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                # anti-diagonal dark line
                [
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, -4.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 4.0
        self.register_buffer("kernels", kernels)

    def response(self, x: torch.Tensor) -> torch.Tensor:
        x_u = to_unit_interval(x)
        kernels = self.kernels.to(device=x.device, dtype=x.dtype)
        response = F.conv2d(x_u, kernels, padding=2)
        response = F.relu(response)
        return response.max(dim=1, keepdim=True).values

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.response(pred), self.response(target))


class HighPassDetailLoss(nn.Module):
    """L1 loss between local-contrast detail maps of predicted and target NIR."""

    def __init__(self) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        self.register_buffer("kernel", kernel_2d.view(1, 1, 5, 5))

    def detail(self, x: torch.Tensor) -> torch.Tensor:
        x_u = to_unit_interval(x)
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        smooth = F.conv2d(F.pad(x_u, (2, 2, 2, 2), mode="reflect"), kernel)
        return x_u - smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.detail(pred), self.detail(target))
