"""Pix2pix generator and discriminator models."""

from __future__ import annotations

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat((x, skip), dim=1)


class ResizeConvUp(nn.Module):
    """Upsample + Conv2d block to reduce transposed-convolution checkerboards."""

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.0, mode: str = "bilinear"
    ) -> None:
        super().__init__()
        if mode == "bilinear":
            upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        elif mode == "nearest":
            upsample = nn.Upsample(scale_factor=2, mode=mode)
        else:
            raise ValueError(f"Unsupported resize-conv mode: {mode}")
        layers: list[nn.Module] = [
            upsample,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat((x, skip), dim=1)


class UNetGenerator(nn.Module):
    """U-Net generator for 128x128 RGB->NIR pix2pix."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 64, dropout: float = 0.5):
        super().__init__()
        self.down1 = UNetDown(in_channels, base, normalize=False)
        self.down2 = UNetDown(base, base * 2)
        self.down3 = UNetDown(base * 2, base * 4)
        self.down4 = UNetDown(base * 4, base * 8)
        self.down5 = UNetDown(base * 8, base * 8)
        self.down6 = UNetDown(base * 8, base * 8)
        self.down7 = UNetDown(base * 8, base * 8, normalize=False)

        self.up1 = UNetUp(base * 8, base * 8, dropout=dropout)
        self.up2 = UNetUp(base * 16, base * 8, dropout=dropout)
        self.up3 = UNetUp(base * 16, base * 8, dropout=dropout)
        self.up4 = UNetUp(base * 16, base * 4)
        self.up5 = UNetUp(base * 8, base * 2)
        self.up6 = UNetUp(base * 4, base)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


class ResizeConvUNetGenerator(nn.Module):
    """U-Net generator with resize-convolution upsampling for 128x128 RGB->NIR."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base: int = 64,
        dropout: float = 0.5,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if mode == "bilinear":
            final_upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        elif mode == "nearest":
            final_upsample = nn.Upsample(scale_factor=2, mode=mode)
        else:
            raise ValueError(f"Unsupported resize-conv mode: {mode}")
        self.down1 = UNetDown(in_channels, base, normalize=False)
        self.down2 = UNetDown(base, base * 2)
        self.down3 = UNetDown(base * 2, base * 4)
        self.down4 = UNetDown(base * 4, base * 8)
        self.down5 = UNetDown(base * 8, base * 8)
        self.down6 = UNetDown(base * 8, base * 8)
        self.down7 = UNetDown(base * 8, base * 8, normalize=False)

        self.up1 = ResizeConvUp(base * 8, base * 8, dropout=dropout, mode=mode)
        self.up2 = ResizeConvUp(base * 16, base * 8, dropout=dropout, mode=mode)
        self.up3 = ResizeConvUp(base * 16, base * 8, dropout=dropout, mode=mode)
        self.up4 = ResizeConvUp(base * 16, base * 4, mode=mode)
        self.up5 = ResizeConvUp(base * 8, base * 2, mode=mode)
        self.up6 = ResizeConvUp(base * 4, base, mode=mode)
        self.final = nn.Sequential(
            final_upsample,
            nn.Conv2d(base * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator for 128x128 conditional pairs."""

    def __init__(self, in_channels: int = 4, base: int = 64) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base, normalize=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            nn.Conv2d(base * 4, base * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, rgb: torch.Tensor, nir: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([rgb, nir], dim=1))


def build_models(
    generator_base: int,
    discriminator_base: int,
    dropout: float,
    generator_type: str = "transposed",
) -> tuple[nn.Module, nn.Module]:
    if generator_type == "transposed":
        generator = UNetGenerator(base=generator_base, dropout=dropout)
    elif generator_type == "resizeconv":
        generator = ResizeConvUNetGenerator(base=generator_base, dropout=dropout, mode="bilinear")
    elif generator_type == "nearestconv":
        generator = ResizeConvUNetGenerator(base=generator_base, dropout=dropout, mode="nearest")
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")
    discriminator = PatchDiscriminator(base=discriminator_base)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator
