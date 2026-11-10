from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class HighFrequencyResidual(nn.Module):
    """Frozen Laplacian high-pass with learnable residual scale (init 0)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.high_pass = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )
        with torch.no_grad():
            self.high_pass.weight.copy_(
                laplacian.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            )
        self.high_pass.weight.requires_grad_(False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        detail = self.act(self.norm(self.high_pass(inputs)))
        return inputs + self.scale * detail


def eca_adaptive_kernel_size(
    channels: int,
    gamma: float = 2.0,
    beta: float = 1.0,
) -> int:
    """ECA-Net odd 1D kernel from channel count: |log2(C)/γ + b/γ|_odd."""
    channels = int(channels)
    if channels < 1:
        raise ValueError(f"channels must be positive, got {channels}")
    t = int(abs(math.log2(channels) / float(gamma) + float(beta) / float(gamma)))
    if t % 2 == 0:
        t += 1
    return max(3, t)


def _conv_kxk_out_channels(conv_kxk: nn.Module) -> int:
    """Resolve output channels of MobileViT local conv_kxk (pre-attn wrap)."""
    module: nn.Module | None = conv_kxk
    if isinstance(module, nn.Sequential):
        module = next(
            (child for child in module if not _is_local_attention(child)),
            None,
        )
    if module is None:
        raise ValueError("Could not find conv inside conv_kxk.")
    if isinstance(module, LocalRepKxk):
        return int(module.channels)
    if hasattr(module, "conv") and hasattr(module.conv, "out_channels"):
        return int(module.conv.out_channels)
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels)
    raise ValueError(f"Unsupported conv_kxk type: {type(module).__name__}")


class EfficientChannelAttention(nn.Module):
    """ECA: GAP → 1D conv over channels → sigmoid gate (almost zero params)."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"eca_kernel must be odd positive, got {kernel_size}")
        self.kernel_size = k
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) → (B,C,1,1) → (B,1,C) → conv1d → (B,C,1,1)
        y = self.pool(inputs)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return inputs * self.sigmoid(y)


class DualPoolEfficientChannelAttention(nn.Module):
    """ECA with GAP+GMP descriptors concatenated into Conv1d(2→1)."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError(f"eca_kernel must be odd positive, got {kernel_size}")
        self.kernel_size = k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) → avg/max (B,1,C) → cat (B,2,C) → conv1d → (B,C,1,1)
        avg = self.avg_pool(inputs).squeeze(-1).transpose(-1, -2)
        maximum = self.max_pool(inputs).squeeze(-1).transpose(-1, -2)
        y = torch.cat([avg, maximum], dim=1)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return inputs * self.sigmoid(y)


class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al.): direction-aware channel gates."""

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        channels = int(channels)
        reduction = max(1, int(reduction))
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        x_h = self.pool_h(inputs)
        x_w = self.pool_w(inputs).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return inputs * a_h * a_w


def _is_local_attention(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            EfficientChannelAttention,
            DualPoolEfficientChannelAttention,
            CoordinateAttention,
        ),
    )


class GeMPool2d(nn.Module):
    """Generalized-Mean pooling with learnable p (GAP when p=1, →max as p↑)."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.p = nn.Parameter(torch.ones(1) * float(p))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) → (B, C)
        p = self.p.clamp(min=self.eps)
        x = inputs.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p).flatten(1)


class MeanStdPool2d(nn.Module):
    """Global average + spatial std pooling; concat to (B, 2C)."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) → (B, 2C)
        mean = inputs.mean(dim=(2, 3))
        mean_map = mean.unsqueeze(-1).unsqueeze(-1)
        var = ((inputs - mean_map) ** 2).mean(dim=(2, 3))
        std = torch.sqrt(var + self.eps)
        return torch.cat((mean, std), dim=1)


class MeanStdGatedPool2d(nn.Module):
    """Mean∥Std with zero-init channel gate on the std branch.

    feature = concat(mean, tanh(gate) * std); gate=0 ⇒ std half inactive (~GAP).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        self.eps = float(eps)
        self.gate = nn.Parameter(torch.zeros(1, channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) → (B, 2C)
        mean = inputs.mean(dim=(2, 3))
        mean_map = mean.unsqueeze(-1).unsqueeze(-1)
        var = ((inputs - mean_map) ** 2).mean(dim=(2, 3))
        std = torch.sqrt(var + self.eps)
        gated_std = torch.tanh(self.gate) * std
        return torch.cat((mean, gated_std), dim=1)


class GapGemGatedPool2d(nn.Module):
    """GAP∥GeM with zero-init channel gate on the GeM branch.

    feature = concat(GAP, tanh(gate) * GeM); gate=0 ⇒ GeM half inactive (~GAP).
    """

    def __init__(
        self,
        channels: int,
        gem_p: float = 3.0,
        gem_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        self.gem = GeMPool2d(p=float(gem_p), eps=float(gem_eps))
        self.gate = nn.Parameter(torch.zeros(1, channels))
        # These are scale/shape parameters rather than ordinary weights.
        # AdamW decay would continuously pull p and the zero-init gate toward
        # zero, weakening the intended GeM branch independently of the loss.
        self.gem.p._exclude_from_weight_decay = True
        self.gate._exclude_from_weight_decay = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C, H, W) → (B, 2C)
        gap = F.adaptive_avg_pool2d(inputs, 1).flatten(1)
        gem = self.gem(inputs)
        gated_gem = torch.tanh(self.gate) * gem
        return torch.cat((gap, gated_gem), dim=1)


class ClasswiseDefectEvidencePooling(nn.Module):
    """Class-wise spatial evidence pooling for localized fruit defects.

    A 1x1 classifier first produces one evidence map per class. Each class then
    owns an independent spatial distribution over that map. The focused score
    is mixed with the ordinary global-average score according to both a learned
    class gate and the concentration of the spatial evidence. Diffuse evidence
    therefore stays close to GAP, while a compact defect response can receive
    more weight.

    Compared with ``GAP -> Linear``, the 1x1 classifier has exactly the same
    classifier weights and the only additional trainable values are one gate
    logit per class.
    """

    def __init__(
        self,
        channels: int,
        num_classes: int,
        temperature: float = 1.0,
        gate_init: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_classes = int(num_classes)
        temperature = float(temperature)
        gate_init = float(gate_init)
        if channels < 1 or num_classes < 2:
            raise ValueError(
                "channels must be positive and num_classes must be >= 2"
            )
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between 0 and 1")

        self.temperature = temperature
        self.eps = float(eps)
        self.evidence_classifier = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(
            torch.full((num_classes,), gate_logit, dtype=torch.float32)
        )
        # This is a mixing coefficient, not an ordinary model weight. AdamW
        # decay would bias it back toward sigmoid(0)=0.5.
        self.gate_logits._exclude_from_weight_decay = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "ClasswiseDefectEvidencePooling expects a BCHW tensor, "
                f"got shape={tuple(inputs.shape)}"
            )

        evidence_maps = self.evidence_classifier(inputs)
        evidence = evidence_maps.flatten(2)  # (B, K, HW)
        gap_logits = evidence.mean(dim=-1)

        attention = torch.softmax(evidence / self.temperature, dim=-1)
        focused_logits = (attention * evidence).sum(dim=-1)

        spatial_size = int(evidence.shape[-1])
        if spatial_size > 1:
            entropy = -(
                attention * attention.clamp_min(self.eps).log()
            ).sum(dim=-1)
            concentration = 1.0 - entropy / math.log(float(spatial_size))
            concentration = concentration.clamp(min=0.0, max=1.0)
        else:
            concentration = torch.zeros_like(gap_logits)

        class_gate = torch.sigmoid(self.gate_logits).unsqueeze(0)
        mix = class_gate * concentration
        return gap_logits + mix * (focused_logits - gap_logits)


def _fuse_conv_bn(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold BN into a dense conv; returns (weight[out,in,kH,kW], bias[out])."""
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    weight = conv.weight * scale
    bias = bn.bias - bn.running_mean * bn.weight / std
    if conv.bias is not None:
        bias = bias + conv.bias * bn.weight / std
    return weight, bias


def _pad_kernel_to_3x3(weight: torch.Tensor) -> torch.Tensor:
    """Pad a dense conv kernel to 3×3 (centered)."""
    _, _, kh, kw = weight.shape
    if kh == 3 and kw == 3:
        return weight
    if kh > 3 or kw > 3:
        raise ValueError(f"Cannot pad kernel larger than 3×3: got {(kh, kw)}")
    pad_h = (3 - kh) // 2
    pad_w = (3 - kw) // 2
    return F.pad(weight, (pad_w, pad_w, pad_h, pad_h))


class LocalRepKxk(nn.Module):
    """Train-time multi-branch local conv; deploy as one dense 3×3 + SiLU.

    Branches (all groups=1): 3×3 + 1×3 + 3×1, each with BN. Inference fuses to a
    single 3×3 Conv (RepVGG-style), keeping MobileViT block interfaces unchanged.
    """

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        channels = int(channels)
        self.channels = channels
        self.deploy = bool(deploy)
        self.act = nn.SiLU(inplace=True)
        if self.deploy:
            self.reparam = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            return
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.branch_1x3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 3),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.branch_3x1 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    @classmethod
    def from_conv_norm_act(cls, conv_norm_act: nn.Module) -> "LocalRepKxk":
        """Replace timm ConvNormAct(3×3) while copying weights into the 3×3 branch."""
        if not hasattr(conv_norm_act, "conv") or not hasattr(conv_norm_act, "bn"):
            raise TypeError("from_conv_norm_act expects ConvNormAct-like module.")
        conv = conv_norm_act.conv
        bn = conv_norm_act.bn
        channels = int(conv.in_channels)
        if int(conv.out_channels) != channels:
            raise ValueError("LocalRepKxk expects square in/out channels.")
        if tuple(conv.kernel_size) != (3, 3):
            raise ValueError(
                f"LocalRepKxk expects a 3×3 source conv, got {conv.kernel_size}."
            )
        module = cls(channels, deploy=False)
        with torch.no_grad():
            module.branch_3x3[0].weight.copy_(conv.weight)
            if conv.bias is not None:
                # source is bias-free in XXS; keep zero if present unexpectedly
                pass
            module.branch_3x3[1].weight.copy_(bn.weight)
            module.branch_3x3[1].bias.copy_(bn.bias)
            module.branch_3x3[1].running_mean.copy_(bn.running_mean)
            module.branch_3x3[1].running_var.copy_(bn.running_var)
            module.branch_3x3[1].eps = float(bn.eps)
            # Asymmetric branches start at 0 so train-time ≈ original 3×3 path.
            nn.init.zeros_(module.branch_1x3[0].weight)
            nn.init.zeros_(module.branch_3x1[0].weight)
            nn.init.ones_(module.branch_1x3[1].weight)
            nn.init.zeros_(module.branch_1x3[1].bias)
            nn.init.zeros_(module.branch_1x3[1].running_mean)
            nn.init.ones_(module.branch_1x3[1].running_var)
            nn.init.ones_(module.branch_3x1[1].weight)
            nn.init.zeros_(module.branch_3x1[1].bias)
            nn.init.zeros_(module.branch_3x1[1].running_mean)
            nn.init.ones_(module.branch_3x1[1].running_var)
        return module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.reparam(inputs))
        fused = (
            self.branch_3x3(inputs)
            + self.branch_1x3(inputs)
            + self.branch_3x1(inputs)
        )
        return self.act(fused)

    def get_equivalent_kernel_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w33, b33 = _fuse_conv_bn(self.branch_3x3[0], self.branch_3x3[1])
        w13, b13 = _fuse_conv_bn(self.branch_1x3[0], self.branch_1x3[1])
        w31, b31 = _fuse_conv_bn(self.branch_3x1[0], self.branch_3x1[1])
        weight = w33 + _pad_kernel_to_3x3(w13) + _pad_kernel_to_3x3(w31)
        bias = b33 + b13 + b31
        return weight, bias

    def fuse(self) -> "LocalRepKxk":
        if self.deploy:
            return self
        weight, bias = self.get_equivalent_kernel_bias()
        fused = LocalRepKxk(self.channels, deploy=True)
        with torch.no_grad():
            fused.reparam.weight.copy_(weight)
            fused.reparam.bias.copy_(bias)
        return fused

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        weight, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        with torch.no_grad():
            self.reparam.weight.copy_(weight)
            self.reparam.bias.copy_(bias)
        self.__delattr__("branch_3x3")
        self.__delattr__("branch_1x3")
        self.__delattr__("branch_3x1")
        self.deploy = True

    def reparameterize(self) -> None:
        self.switch_to_deploy()


class LightTailEnhance(nn.Module):
    """DW3x3 -> 1x1 compress -> 1x1 expand, zero-init residual (no SE/ECA)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(16, int(channels) // int(reduction))
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enhanced = self.expand(self.reduce(self.depthwise(inputs)))
        return inputs + self.gamma * enhanced


def _mobilevit_local_global(
    block: nn.Module,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run MobileViT local CNN + unfold/Transformer/fold + proj.

    Returns:
        shortcut (X), local features after conv_1x1 (transformer dim),
        global features after conv_proj (block out channels).
    """
    shortcut = inputs
    local_feat = block.conv_kxk(inputs)
    local_feat = block.conv_1x1(local_feat)

    patch_h, patch_w = block.patch_size
    batch, channels, height, width = local_feat.shape
    new_h = math.ceil(height / patch_h) * patch_h
    new_w = math.ceil(width / patch_w) * patch_w
    num_patch_h = new_h // patch_h
    num_patch_w = new_w // patch_w
    num_patches = num_patch_h * num_patch_w
    interpolate = False
    features = local_feat
    if new_h != height or new_w != width:
        features = F.interpolate(
            features,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        local_feat = F.interpolate(
            local_feat,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        interpolate = True

    features = (
        features.reshape(
            batch * channels * num_patch_h, patch_h, num_patch_w, patch_w
        ).transpose(1, 2)
    )
    features = (
        features.reshape(batch, channels, num_patches, block.patch_area)
        .transpose(1, 3)
        .reshape(batch * block.patch_area, num_patches, -1)
    )

    features = block.transformer(features)
    features = block.norm(features)

    features = features.contiguous().view(
        batch,
        block.patch_area,
        num_patches,
        -1,
    )
    features = features.transpose(1, 3).reshape(
        batch * channels * num_patch_h,
        num_patch_w,
        patch_h,
        patch_w,
    )
    features = features.transpose(1, 2).reshape(
        batch,
        channels,
        num_patch_h * patch_h,
        num_patch_w * patch_w,
    )
    if interpolate:
        features = F.interpolate(
            features,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        local_feat = F.interpolate(
            local_feat,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    global_feat = block.conv_proj(features)
    return shortcut, local_feat, global_feat


def _kxk_kernel_size(conv_kxk: nn.Module) -> int:
    """Resolve n×n kernel from conv_kxk (ConvNormAct / LocalRepKxk / +ECA/CA)."""
    module: nn.Module | None = conv_kxk
    if isinstance(module, nn.Sequential):
        module = next(
            (child for child in module if not _is_local_attention(child)),
            None,
        )
    if module is None:
        raise ValueError("Could not find conv inside conv_kxk.")
    if isinstance(module, LocalRepKxk):
        # Deploy and train forms are always equivalent to a dense 3×3 for MFF.
        return 3
    conv = module.conv if hasattr(module, "conv") else module
    if isinstance(conv, LocalRepKxk):
        return 3
    kernel = conv.kernel_size
    if isinstance(kernel, tuple):
        return int(kernel[0])
    return int(kernel)


class MFFMobileVitBlock(nn.Module):
    """MobileViT block with MFF-style local/global fusion.

    Keeps the original conv_fusion path and adds a zero-init residual toward
    MFF fusion so training starts near plain XXS.
    """

    def __init__(
        self,
        block: nn.Module,
        expose_global: bool = False,
        channel_gamma: bool = False,
    ) -> None:
        super().__init__()
        if block.conv_fusion is None:
            raise ValueError("MFF fusion requires conv_fusion to exist.")
        self.block = block
        self.expose_global = bool(expose_global)
        # LGRC toggles this only for its training forward.  Keeping the flag
        # false preserves the memory use and outputs of every existing model.
        self.capture_relations = False
        self.last_local_feat: torch.Tensor | None = None
        self.last_global_feat: torch.Tensor | None = None
        out_channels = int(block.conv_proj.conv.out_channels)
        transformer_channels = int(block.conv_1x1.out_channels)
        kernel = _kxk_kernel_size(block.conv_kxk)

        self.local_align = nn.Conv2d(
            transformer_channels, out_channels, kernel_size=1, bias=False
        )
        self.local_fuse = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.global_fuse = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.mff_fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=kernel,
                padding=kernel // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        # Scalar or per-channel mix; zeros keep start ≈ original conv_fusion.
        if bool(channel_gamma):
            self.gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1))
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block = self.block
        shortcut, local_feat, global_feat = _mobilevit_local_global(block, inputs)
        if self.capture_relations:
            self.last_local_feat = local_feat
        if self.expose_global or self.capture_relations:
            self.last_global_feat = global_feat
        local_aligned = self.local_align(local_feat)
        local_global = self.local_fuse(local_aligned) + self.global_fuse(
            global_feat
        )
        mff_fused = self.mff_fusion(torch.cat((shortcut, local_global), dim=1))
        static_fused = block.conv_fusion(torch.cat((shortcut, global_feat), dim=1))
        return static_fused + self.gamma * (mff_fused - static_fused)


class MFFResConcatMobileVitBlock(nn.Module):
    """Wheat P-MobileViT style residual-then-concat fusion on MFF path.

    C = X + 1x1(local_aligned); Y = 1x1(cat(C, global)).
    Keeps original conv_fusion with zero-init residual (same start as MFF).
    """

    def __init__(
        self,
        block: nn.Module,
        expose_global: bool = False,
        channel_gamma: bool = False,
    ) -> None:
        super().__init__()
        if block.conv_fusion is None:
            raise ValueError("MFF res-concat fusion requires conv_fusion.")
        self.block = block
        self.expose_global = bool(expose_global)
        self.last_global_feat: torch.Tensor | None = None
        out_channels = int(block.conv_proj.conv.out_channels)
        transformer_channels = int(block.conv_1x1.out_channels)

        self.local_align = nn.Conv2d(
            transformer_channels, out_channels, kernel_size=1, bias=False
        )
        # Branch B: 1x1 on aligned local features (paper Fig.3 style).
        self.residual_branch = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.res_concat_fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        if bool(channel_gamma):
            self.gamma = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1))
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block = self.block
        shortcut, local_feat, global_feat = _mobilevit_local_global(block, inputs)
        if self.expose_global:
            self.last_global_feat = global_feat
        local_aligned = self.local_align(local_feat)
        fused_local = shortcut + self.residual_branch(local_aligned)
        mff_fused = self.res_concat_fusion(
            torch.cat((fused_local, global_feat), dim=1)
        )
        static_fused = block.conv_fusion(torch.cat((shortcut, global_feat), dim=1))
        return static_fused + self.gamma * (mff_fused - static_fused)


class SSGResConcatMobileVitBlock(nn.Module):
    """Stage-selective ResFus with a bounded, weight-decay-free gate.

    The original ResFus block uses an unconstrained scalar ``gamma``.  This
    variant performs a convex interpolation between the original MobileViT
    fusion and the residual-then-concat fusion:

        output = static + sigmoid(gate_logit) * (resfus - static)

    ``gate_logit`` is initialized from ``gate_init`` and excluded from AdamW
    weight decay because it is a mixing coefficient rather than a model
    weight.  Stage selection is controlled by the existing stage flags.
    """

    def __init__(
        self,
        block: nn.Module,
        expose_global: bool = False,
        gate_init: float = 0.10,
    ) -> None:
        super().__init__()
        if block.conv_fusion is None:
            raise ValueError("SSG-ResFus requires conv_fusion.")
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError("gate_init must be strictly between 0 and 1.")

        self.block = block
        self.expose_global = bool(expose_global)
        self.last_global_feat: torch.Tensor | None = None
        out_channels = int(block.conv_proj.conv.out_channels)
        transformer_channels = int(block.conv_1x1.out_channels)

        self.local_align = nn.Conv2d(
            transformer_channels, out_channels, kernel_size=1, bias=False
        )
        self.residual_branch = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.res_concat_fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        gate_logit = math.log(float(gate_init) / (1.0 - float(gate_init)))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit))
        self.gate_logit._exclude_from_weight_decay = True
        self.out_channels = out_channels

    @property
    def fusion_weight(self) -> torch.Tensor:
        """Return the bounded ResFus interpolation weight."""
        return torch.sigmoid(self.gate_logit)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        block = self.block
        shortcut, local_feat, global_feat = _mobilevit_local_global(block, inputs)
        if self.expose_global:
            self.last_global_feat = global_feat

        local_aligned = self.local_align(local_feat)
        fused_local = shortcut + self.residual_branch(local_aligned)
        resfus = self.res_concat_fusion(
            torch.cat((fused_local, global_feat), dim=1)
        )
        static = block.conv_fusion(torch.cat((shortcut, global_feat), dim=1))
        weight = self.fusion_weight.to(dtype=static.dtype)
        return static + weight * (resfus - static)


class MobileViTXXSDFN(nn.Module):
    """Defect-focused MobileViT-XXS whole-network redesign.

    - Shallow high-frequency residual (after stage1)
    - Optional local RepKxk (3×3+1×3+3×1 → deploy 3×3) replacing conv_kxk
    - Optional local-ECA or local-CA inside MobileViT blocks
      (after conv_kxk, before conv_1x1; mutually exclusive;
      ECA stages use independent use_stage*_local_eca flags)
    - Optional shallow Bottleneck-ECA on stage0/1 (timm attn slot after DW)
    - MFF-style fusion in all MobileViT blocks (stages 2/3/4)
    - Optional Stage4 global-feat auxiliary classifier (train-only)
    - Optional dual-scale head: Stage3 (proj+GAP) ∥ Stage4 (GAP) → MLP
    - Optional Mean+Std pooling head (concat GAP∥spatial-std → 2C)
    - Optional gated Mean∥Std (tanh gate on std, zero-init ≈ GAP)
    - Optional gated GAP∥GeM (tanh gate on GeM, zero-init ≈ GAP)
    - Light tail enhance after final_conv (no SE/ECA on the tail)
    """

    _STAGE_OUT_CHANNELS = {0: 16, 1: 24, 2: 48, 3: 64, 4: 80}

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        use_high_frequency: bool = True,
        use_mff: bool = True,
        use_stage2_mff: bool = True,
        use_stage3_mff: bool = True,
        use_stage4_mff: bool = True,
        use_tail_enhance: bool = True,
        tail_reduction: int = 4,
        mff_variant: str = "default",
        use_local_eca: bool = False,
        eca_kernel: int = 3,
        use_adaptive_eca_kernel: bool = False,
        eca_gamma: float = 2.0,
        eca_beta: float = 1.0,
        use_dual_pool_eca: bool = False,
        use_stage2_local_eca: bool = True,
        use_stage3_local_eca: bool = True,
        use_stage4_local_eca: bool = True,
        use_local_ca: bool = False,
        ca_reduction: int = 32,
        use_shallow_eca: bool = False,
        use_stage0_eca: bool = True,
        use_stage1_eca: bool = True,
        use_local_rep: bool = False,
        use_global_aux: bool = False,
        use_gem_pool: bool = False,
        gem_p: float = 3.0,
        use_mean_std_pool: bool = False,
        use_mean_std_gate_pool: bool = False,
        use_gap_gem_gate_pool: bool = False,
        use_class_evidence_pool: bool = False,
        evidence_temperature: float = 1.0,
        evidence_gate_init: float = 0.05,
        mean_std_eps: float = 1e-5,
        use_dual_scale_head: bool = False,
        dual_scale_mid_dim: int = 96,
        dual_scale_hidden: int = 256,
        use_mff_channel_gamma: bool = False,
        ssg_resfus_gate_init: float = 0.10,
        head_dropout: float | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-DFN requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        feature_dim = int(self.backbone.num_features)

        self.use_high_frequency = bool(use_high_frequency)
        self.high_frequency = (
            HighFrequencyResidual(24) if self.use_high_frequency else nn.Identity()
        )
        self.mff_variant = str(mff_variant).strip().lower() or "default"
        self.use_local_eca = bool(use_local_eca)
        self.eca_kernel = int(eca_kernel)
        self.use_adaptive_eca_kernel = bool(use_adaptive_eca_kernel)
        self.eca_gamma = float(eca_gamma)
        self.eca_beta = float(eca_beta)
        self.use_dual_pool_eca = bool(use_dual_pool_eca)
        self.use_stage2_local_eca = bool(use_stage2_local_eca)
        self.use_stage3_local_eca = bool(use_stage3_local_eca)
        self.use_stage4_local_eca = bool(use_stage4_local_eca)
        self.use_local_ca = bool(use_local_ca)
        self.ca_reduction = int(ca_reduction)
        self.use_shallow_eca = bool(use_shallow_eca)
        self.use_local_rep = bool(use_local_rep)
        self.use_global_aux = bool(use_global_aux)
        self.use_gem_pool = bool(use_gem_pool)
        self.use_mean_std_pool = bool(use_mean_std_pool)
        self.use_mean_std_gate_pool = bool(use_mean_std_gate_pool)
        self.use_gap_gem_gate_pool = bool(use_gap_gem_gate_pool)
        self.use_class_evidence_pool = bool(use_class_evidence_pool)
        self.use_dual_scale_head = bool(use_dual_scale_head)
        self.use_mff_channel_gamma = bool(use_mff_channel_gamma)
        self.ssg_resfus_gate_init = float(ssg_resfus_gate_init)
        self.global_aux_block: nn.Module | None = None

        pool_flags = [
            self.use_gem_pool,
            self.use_mean_std_pool,
            self.use_mean_std_gate_pool,
            self.use_gap_gem_gate_pool,
            self.use_class_evidence_pool,
        ]
        if self.use_local_eca and self.use_local_ca:
            raise ValueError(
                "use_local_eca and use_local_ca are mutually exclusive "
                "(same local-branch mount point)."
            )
        if self.use_dual_scale_head and any(pool_flags):
            raise ValueError(
                "use_dual_scale_head is mutually exclusive with pooling-head variants."
            )
        if sum(bool(f) for f in pool_flags) > 1:
            raise ValueError(
                "use_gem_pool / use_mean_std_pool / use_mean_std_gate_pool / "
                "use_gap_gem_gate_pool / use_class_evidence_pool are mutually "
                "exclusive."
            )

        if self.use_global_aux and not bool(use_stage4_mff and use_mff):
            raise ValueError(
                "use_global_aux=True requires use_mff=True and use_stage4_mff=True "
                "(aux reads Stage4 global_feat from the MFF-wrapped block)."
            )

        if self.use_shallow_eca:
            self._inject_shallow_bottleneck_eca(
                use_stage0=bool(use_stage0_eca),
                use_stage1=bool(use_stage1_eca),
                eca_kernel=self.eca_kernel,
                adaptive_kernel=self.use_adaptive_eca_kernel,
                eca_gamma=self.eca_gamma,
                eca_beta=self.eca_beta,
            )

        # Local-rep first (replaces conv_kxk), then optional ECA/CA, then MFF wrap.
        if self.use_local_rep:
            self._inject_local_rep(
                use_stage2=bool(use_stage2_mff),
                use_stage3=bool(use_stage3_mff),
                use_stage4=bool(use_stage4_mff),
            )

        if self.use_local_eca:
            self._inject_local_eca(
                use_stage2=self.use_stage2_local_eca,
                use_stage3=self.use_stage3_local_eca,
                use_stage4=self.use_stage4_local_eca,
                eca_kernel=self.eca_kernel,
                adaptive_kernel=self.use_adaptive_eca_kernel,
                eca_gamma=self.eca_gamma,
                eca_beta=self.eca_beta,
                dual_pool=self.use_dual_pool_eca,
            )

        if self.use_local_ca:
            self._inject_local_ca(
                use_stage2=bool(use_stage2_mff),
                use_stage3=bool(use_stage3_mff),
                use_stage4=bool(use_stage4_mff),
                ca_reduction=self.ca_reduction,
            )

        if bool(use_mff):
            self._wrap_mff_blocks(
                use_stage2_mff=bool(use_stage2_mff),
                use_stage3_mff=bool(use_stage3_mff),
                use_stage4_mff=bool(use_stage4_mff),
                mff_variant=self.mff_variant,
                expose_global_stage4=self.use_global_aux,
                channel_gamma=self.use_mff_channel_gamma,
                ssg_gate_init=self.ssg_resfus_gate_init,
            )

        self.use_tail_enhance = bool(use_tail_enhance)
        self.tail_enhance = (
            LightTailEnhance(feature_dim, reduction=tail_reduction)
            if self.use_tail_enhance
            else nn.Identity()
        )

        if self.use_gem_pool:
            self.pool = GeMPool2d(p=float(gem_p))
        elif self.use_gap_gem_gate_pool:
            self.pool = GapGemGatedPool2d(
                channels=feature_dim, gem_p=float(gem_p)
            )
        elif self.use_mean_std_gate_pool:
            self.pool = MeanStdGatedPool2d(
                channels=feature_dim, eps=float(mean_std_eps)
            )
        elif self.use_mean_std_pool:
            self.pool = MeanStdPool2d(eps=float(mean_std_eps))
        else:
            self.pool = None
        head_drop = float(dropout if head_dropout is None else head_dropout)
        self.dropout = nn.Dropout(p=head_drop)

        if self.use_class_evidence_pool:
            self.stage3_proj = None
            self.classifier = ClasswiseDefectEvidencePooling(
                channels=feature_dim,
                num_classes=num_classes,
                temperature=float(evidence_temperature),
                gate_init=float(evidence_gate_init),
            )
            self.num_features = feature_dim
        elif self.use_dual_scale_head:
            stage3_ch = self._STAGE_OUT_CHANNELS[3]
            stage4_ch = self._STAGE_OUT_CHANNELS[4]
            mid_dim = int(dual_scale_mid_dim)
            hidden = int(dual_scale_hidden)
            if mid_dim < 1 or hidden < 1:
                raise ValueError(
                    "dual_scale_mid_dim and dual_scale_hidden must be >= 1"
                )
            self.stage3_proj = nn.Sequential(
                nn.Conv2d(stage3_ch, mid_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_dim),
                nn.SiLU(inplace=True),
            )
            fused_dim = mid_dim + stage4_ch
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Dropout(p=head_drop),
                nn.Linear(hidden, num_classes, bias=True),
            )
            self.num_features = fused_dim
            # Head MLP already has dropout; keep outer dropout as Identity.
            self.dropout = nn.Identity()
        else:
            self.stage3_proj = None
            head_in = (
                feature_dim * 2
                if (
                    self.use_mean_std_pool
                    or self.use_mean_std_gate_pool
                    or self.use_gap_gem_gate_pool
                )
                else feature_dim
            )
            self.classifier = nn.Linear(head_in, num_classes)
            self.num_features = head_in

        if self.use_global_aux:
            assert self.global_aux_block is not None
            aux_channels = int(self.global_aux_block.out_channels)
            self.aux_head = nn.Linear(aux_channels, num_classes)
        else:
            self.aux_head = None

    def _inject_local_rep(
        self,
        use_stage2: bool,
        use_stage3: bool,
        use_stage4: bool,
    ) -> None:
        """Replace MobileViT conv_kxk with LocalRepKxk (3×3+1×3+3×1)."""
        from timm.models.mobilevit import MobileVitBlock

        stage_flags = {
            2: use_stage2,
            3: use_stage3,
            4: use_stage4,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if not isinstance(block, MobileVitBlock):
                    continue
                conv_kxk = block.conv_kxk
                if isinstance(conv_kxk, LocalRepKxk):
                    continue
                if isinstance(conv_kxk, nn.Sequential):
                    raise RuntimeError(
                        "local_rep must be injected before wrapping conv_kxk "
                        "in Sequential (e.g. before local ECA)."
                    )
                block.conv_kxk = LocalRepKxk.from_conv_norm_act(conv_kxk)

    def _resolve_eca_kernel(
        self,
        channels: int,
        eca_kernel: int,
        adaptive_kernel: bool,
        eca_gamma: float,
        eca_beta: float,
    ) -> int:
        if adaptive_kernel:
            return eca_adaptive_kernel_size(
                channels, gamma=eca_gamma, beta=eca_beta
            )
        return int(eca_kernel)

    def _inject_shallow_bottleneck_eca(
        self,
        use_stage0: bool,
        use_stage1: bool,
        eca_kernel: int,
        adaptive_kernel: bool = False,
        eca_gamma: float = 2.0,
        eca_beta: float = 1.0,
    ) -> None:
        """Replace stage0/1 BottleneckBlock.attn Identity with ECA (after DW)."""
        from timm.models.byobnet import BottleneckBlock

        stage_flags = {
            0: use_stage0,
            1: use_stage1,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block in stage:
                if not isinstance(block, BottleneckBlock):
                    continue
                if isinstance(block.attn, EfficientChannelAttention):
                    continue
                if not isinstance(block.attn, nn.Identity):
                    raise RuntimeError(
                        f"stage{stage_index} Bottleneck.attn is "
                        f"{type(block.attn).__name__}, expected Identity "
                        "before shallow ECA injection."
                    )
                mid_channels = int(block.conv2_kxk.conv.out_channels)
                k = self._resolve_eca_kernel(
                    mid_channels,
                    eca_kernel,
                    adaptive_kernel,
                    eca_gamma,
                    eca_beta,
                )
                block.attn = EfficientChannelAttention(kernel_size=k)

    def _inject_local_eca(
        self,
        use_stage2: bool,
        use_stage3: bool,
        use_stage4: bool,
        eca_kernel: int,
        adaptive_kernel: bool = False,
        eca_gamma: float = 2.0,
        eca_beta: float = 1.0,
        dual_pool: bool = False,
    ) -> None:
        """Place ECA after conv_kxk (BN+SiLU / LocalRep), before conv_1x1."""
        from timm.models.mobilevit import MobileVitBlock

        attn_cls = (
            DualPoolEfficientChannelAttention
            if dual_pool
            else EfficientChannelAttention
        )
        stage_flags = {
            2: use_stage2,
            3: use_stage3,
            4: use_stage4,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if not isinstance(block, MobileVitBlock):
                    continue
                if isinstance(block.conv_kxk, nn.Sequential) and any(
                    _is_local_attention(m) for m in block.conv_kxk
                ):
                    continue
                channels = _conv_kxk_out_channels(block.conv_kxk)
                k = self._resolve_eca_kernel(
                    channels,
                    eca_kernel,
                    adaptive_kernel,
                    eca_gamma,
                    eca_beta,
                )
                block.conv_kxk = nn.Sequential(
                    block.conv_kxk,
                    attn_cls(kernel_size=k),
                )

    def _inject_local_ca(
        self,
        use_stage2: bool,
        use_stage3: bool,
        use_stage4: bool,
        ca_reduction: int,
    ) -> None:
        """Place Coordinate Attention after conv_kxk, before conv_1x1."""
        from timm.models.mobilevit import MobileVitBlock

        stage_flags = {
            2: use_stage2,
            3: use_stage3,
            4: use_stage4,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if not isinstance(block, MobileVitBlock):
                    continue
                if isinstance(block.conv_kxk, nn.Sequential) and any(
                    _is_local_attention(m) for m in block.conv_kxk
                ):
                    continue
                channels = _conv_kxk_out_channels(block.conv_kxk)
                block.conv_kxk = nn.Sequential(
                    block.conv_kxk,
                    CoordinateAttention(
                        channels=channels, reduction=ca_reduction
                    ),
                )

    def _wrap_mff_blocks(
        self,
        use_stage2_mff: bool,
        use_stage3_mff: bool,
        use_stage4_mff: bool,
        mff_variant: str = "default",
        expose_global_stage4: bool = False,
        channel_gamma: bool = False,
        ssg_gate_init: float = 0.10,
    ) -> None:
        from timm.models.mobilevit import MobileVitBlock

        variant = str(mff_variant).strip().lower() or "default"
        if variant in {"default", "mff"}:
            wrapper = MFFMobileVitBlock
        elif variant in {"res_concat", "resfus", "wheat"}:
            wrapper = MFFResConcatMobileVitBlock
        elif variant in {
            "ssg_res_concat",
            "ssg_resfus",
            "bounded_res_concat",
        }:
            wrapper = SSGResConcatMobileVitBlock
        else:
            raise ValueError(
                f"Unknown mff_variant={mff_variant!r}; "
                "use 'default', 'res_concat', or 'ssg_res_concat'."
            )

        stage_flags = {
            2: use_stage2_mff,
            3: use_stage3_mff,
            4: use_stage4_mff,
        }
        for stage_index, enabled in stage_flags.items():
            if not enabled:
                continue
            stage = self.backbone.stages[stage_index]
            for block_index, block in enumerate(stage):
                if isinstance(block, MobileVitBlock):
                    expose = bool(expose_global_stage4 and stage_index == 4)
                    if wrapper is SSGResConcatMobileVitBlock:
                        if channel_gamma:
                            raise ValueError(
                                "SSG-ResFus uses its own bounded scalar gate; "
                                "channel_gamma must be false."
                            )
                        wrapped = wrapper(
                            block,
                            expose_global=expose,
                            gate_init=float(ssg_gate_init),
                        )
                    else:
                        wrapped = wrapper(
                            block,
                            expose_global=expose,
                            channel_gamma=bool(channel_gamma),
                        )
                    stage[block_index] = wrapped
                    if expose:
                        self.global_aux_block = wrapped

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)
        features = self.high_frequency(features)
        features = self.backbone.stages[2](features)
        features = self.backbone.stages[3](features)
        if self.use_dual_scale_head:
            assert self.stage3_proj is not None
            stage3_feat = features
            features = self.backbone.stages[4](features)
            stage3_vec = F.adaptive_avg_pool2d(
                self.stage3_proj(stage3_feat), 1
            ).flatten(1)
            stage4_vec = F.adaptive_avg_pool2d(features, 1).flatten(1)
            return torch.cat((stage3_vec, stage4_vec), dim=1)

        features = self.backbone.stages[4](features)
        features = self.backbone.final_conv(features)
        features = self.tail_enhance(features)
        if self.use_class_evidence_pool:
            return features
        if (
            self.use_gem_pool
            or self.use_mean_std_pool
            or self.use_mean_std_gate_pool
            or self.use_gap_gem_gate_pool
        ):
            assert self.pool is not None
            return self.pool(features)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def _aux_logits_from_cached_global(self) -> torch.Tensor:
        if self.aux_head is None or self.global_aux_block is None:
            raise RuntimeError("Global aux head is not configured on this model.")
        global_feat = getattr(self.global_aux_block, "last_global_feat", None)
        if global_feat is None:
            raise RuntimeError(
                "Stage4 global_feat was not cached; run forward_features first."
            )
        pooled = F.adaptive_avg_pool2d(global_feat, 1).flatten(1)
        return self.aux_head(pooled)

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        features = self.forward_features(inputs)
        logits = self.classifier(self.dropout(features))
        returned_features = (
            F.adaptive_avg_pool2d(features, 1).flatten(1)
            if self.use_class_evidence_pool
            else features
        )
        if return_aux:
            aux_logits = self._aux_logits_from_cached_global()
            if return_features:
                return logits, returned_features, aux_logits
            return logits, aux_logits
        if return_features:
            return logits, returned_features
        return logits


def mobilevit_xxs_dfn(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSDFN:
    return MobileViTXXSDFN(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
