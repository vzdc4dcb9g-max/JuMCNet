from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_dfn import MFFMobileVitBlock, MobileViTXXSDFN


def _normalized_entropy(probabilities: torch.Tensor, eps: float) -> torch.Tensor:
    if probabilities.shape[-1] < 2:
        return probabilities.new_zeros(())
    values = probabilities.clamp_min(eps)
    entropy = -(values * values.log()).sum(dim=-1).mean()
    return entropy / math.log(probabilities.shape[-1])


def _diagnostic_state(
    logits: torch.Tensor,
    usage: torch.Tensor,
    entropy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "auxiliary_logits": logits,
        "component_usage": usage,
        "component_entropy": entropy,
    }


class ResidualEncodingCore(nn.Module):
    """Compact differentiable residual encoder with optional balancing."""

    def __init__(
        self,
        in_channels: int,
        reduced_channels: int,
        num_codewords: int,
        assignment_scale: float,
        sinkhorn_iterations: int = 0,
        sinkhorn_epsilon: float = 0.08,
        sinkhorn_mix: float = 0.5,
        descriptor_dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(in_channels, reduced_channels) < 1 or num_codewords < 2:
            raise ValueError("Residual encoding dimensions are invalid")
        if assignment_scale <= 0.0 or sinkhorn_epsilon <= 0.0:
            raise ValueError("Assignment scales must be positive")
        if not 0.0 <= sinkhorn_mix <= 1.0:
            raise ValueError("sinkhorn_mix must be in [0, 1]")
        if not 0.0 <= descriptor_dropout < 1.0:
            raise ValueError("descriptor_dropout must be in [0, 1)")
        self.reduced_channels = int(reduced_channels)
        self.num_codewords = int(num_codewords)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_mix = float(sinkhorn_mix)
        self.descriptor_dropout = float(descriptor_dropout)
        self.eps = float(eps)
        self.reduce = nn.Conv2d(
            int(in_channels), self.reduced_channels, kernel_size=1, bias=True
        )
        self.codewords = nn.Parameter(
            torch.empty(self.num_codewords, self.reduced_channels)
        )
        raw_scale = math.log(math.expm1(float(assignment_scale)))
        self.raw_assignment_scale = nn.Parameter(
            torch.full((self.num_codewords,), raw_scale)
        )
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.num_codewords * self.reduced_channels

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.reduce.weight, mode="fan_out")
        nn.init.zeros_(self.reduce.bias)
        nn.init.uniform_(
            self.codewords,
            -1.0 / math.sqrt(self.reduced_channels),
            1.0 / math.sqrt(self.reduced_channels),
        )

    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        batch, descriptors, codewords = scores.shape
        matrix = torch.exp(
            (scores / self.sinkhorn_epsilon).clamp(min=-50.0, max=50.0)
        )
        matrix = matrix.reshape(batch * descriptors, codewords).transpose(0, 1)
        matrix = matrix / matrix.sum().clamp_min(self.eps)
        samples = matrix.shape[1]
        for _ in range(self.sinkhorn_iterations):
            matrix = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(self.eps)
            matrix = matrix / float(codewords)
            matrix = matrix / matrix.sum(dim=0, keepdim=True).clamp_min(self.eps)
            matrix = matrix / float(samples)
        matrix = matrix * float(samples)
        return matrix.transpose(0, 1).reshape(batch, descriptors, codewords)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("ResidualEncodingCore expects BCHW features")
        descriptors = self.reduce(features).flatten(2).transpose(1, 2)
        descriptor_mask: torch.Tensor | None = None
        if self.training and self.descriptor_dropout > 0.0:
            descriptor_mask = torch.rand(
                descriptors.shape[:2], device=descriptors.device
            ).ge(self.descriptor_dropout)
            descriptor_mask[:, 0] = True
        residuals = descriptors.unsqueeze(2) - self.codewords.view(
            1, 1, self.num_codewords, self.reduced_channels
        )
        distance = residuals.float().square().sum(dim=-1)
        scales = F.softplus(self.raw_assignment_scale.float()).clamp_min(self.eps)
        scores = -distance * scales.view(1, 1, -1)
        if self.sinkhorn_iterations > 0:
            unconstrained = torch.softmax(scores, dim=-1)
            balanced = self._sinkhorn(scores)
            assignments = (
                (1.0 - self.sinkhorn_mix) * unconstrained
                + self.sinkhorn_mix * balanced
            )
        else:
            assignments = torch.softmax(scores, dim=-1)
        encoding_weights = assignments
        if descriptor_mask is not None:
            encoding_weights = encoding_weights * descriptor_mask.unsqueeze(-1)
        encoded = (encoding_weights.unsqueeze(-1) * residuals.float()).sum(dim=1)
        encoded = F.normalize(encoded, p=2, dim=-1, eps=self.eps)
        encoded = F.normalize(encoded.flatten(1), p=2, dim=-1, eps=self.eps)
        if descriptor_mask is None:
            diagnostic_assignments = assignments.reshape(-1, self.num_codewords)
        else:
            diagnostic_assignments = assignments[descriptor_mask]
        usage = diagnostic_assignments.mean(dim=0)
        entropy = _normalized_entropy(diagnostic_assignments, self.eps)
        return encoded, usage, entropy


class ProjectedAuxHead(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, num_classes: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(int(input_dim), int(projection_dim)),
            nn.LayerNorm(int(projection_dim)),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(int(projection_dim), int(num_classes))

    def classify(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.project(descriptor))


class BalancedSinkhornResidualAux(ProjectedAuxHead):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        reduced = int(config.get("reduced_channels", 48))
        codewords = int(config.get("num_codewords", 4))
        super().__init__(
            reduced * codewords,
            int(config.get("projection_dim", 64)),
            num_classes,
        )
        self.core = ResidualEncodingCore(
            in_channels=in_channels,
            reduced_channels=reduced,
            num_codewords=codewords,
            assignment_scale=float(config.get("assignment_scale", 1.0)),
            sinkhorn_iterations=int(config.get("sinkhorn_iterations", 3)),
            sinkhorn_epsilon=float(config.get("sinkhorn_epsilon", 0.08)),
            sinkhorn_mix=float(config.get("sinkhorn_mix", 0.5)),
        )

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        descriptor, usage, entropy = self.core(stage3)
        return _diagnostic_state(self.classify(descriptor), usage, entropy)


class SpatialPyramidResidualAux(ProjectedAuxHead):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        reduced = int(config.get("reduced_channels", 32))
        codewords = int(config.get("num_codewords", 4))
        super().__init__(
            5 * reduced * codewords,
            int(config.get("projection_dim", 96)),
            num_classes,
        )
        self.core = ResidualEncodingCore(
            in_channels=in_channels,
            reduced_channels=reduced,
            num_codewords=codewords,
            assignment_scale=float(config.get("assignment_scale", 1.0)),
        )

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        height, width = stage3.shape[-2:]
        h_mid, w_mid = max(1, height // 2), max(1, width // 2)
        regions = [
            stage3,
            stage3[..., :h_mid, :w_mid],
            stage3[..., :h_mid, w_mid:],
            stage3[..., h_mid:, :w_mid],
            stage3[..., h_mid:, w_mid:],
        ]
        outputs = [self.core(region) for region in regions]
        descriptor = torch.cat([item[0] for item in outputs], dim=1)
        usage = torch.stack([item[1] for item in outputs]).mean(dim=0)
        entropy = torch.stack([item[2] for item in outputs]).mean()
        return _diagnostic_state(self.classify(descriptor), usage, entropy)


class MultiScaleResidualAux(ProjectedAuxHead):
    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        num_classes: int,
        config: dict[str, Any],
    ) -> None:
        reduced = int(config.get("reduced_channels", 32))
        codewords = int(config.get("num_codewords", 4))
        super().__init__(
            2 * reduced * codewords,
            int(config.get("projection_dim", 96)),
            num_classes,
        )
        common = dict(
            reduced_channels=reduced,
            num_codewords=codewords,
            assignment_scale=float(config.get("assignment_scale", 1.0)),
        )
        self.stage2_core = ResidualEncodingCore(stage2_channels, **common)
        self.stage3_core = ResidualEncodingCore(stage3_channels, **common)

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        if stage2 is None:
            raise RuntimeError("Multi-scale residual auxiliary requires Stage2")
        second = self.stage2_core(stage2)
        third = self.stage3_core(stage3)
        descriptor = torch.cat((second[0], third[0]), dim=1)
        usage = 0.5 * (second[1] + third[1])
        entropy = 0.5 * (second[2] + third[2])
        return _diagnostic_state(self.classify(descriptor), usage, entropy)


class BilinearMomentAux(ProjectedAuxHead):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        channels = int(config.get("bilinear_channels", 24))
        super().__init__(
            channels * channels,
            int(config.get("projection_dim", 96)),
            num_classes,
        )
        self.eps = 1e-6
        self.reduce = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        descriptors = self.reduce(stage3).flatten(2).float()
        descriptors = descriptors - descriptors.mean(dim=2, keepdim=True)
        covariance = descriptors @ descriptors.transpose(1, 2)
        covariance = covariance / max(1, descriptors.shape[-1] - 1)
        vector = covariance.flatten(1)
        vector = vector.sign() * vector.abs().clamp_min(self.eps).sqrt()
        vector = F.normalize(vector, p=2, dim=1, eps=self.eps)
        logits = self.classify(vector)
        groups = torch.chunk(descriptors.square().mean(dim=2), 4, dim=1)
        energy = torch.stack([group.mean(dim=1) for group in groups], dim=1)
        components = energy / energy.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return _diagnostic_state(
            logits, components.mean(dim=0), _normalized_entropy(components, self.eps)
        )


class HighPassMomentAux(ProjectedAuxHead):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        channels = int(config.get("highpass_channels", 32))
        super().__init__(
            channels * 6,
            int(config.get("projection_dim", 64)),
            num_classes,
        )
        self.eps = 1e-6
        self.reduce = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)

    @staticmethod
    def _moments(values: torch.Tensor) -> list[torch.Tensor]:
        flat = values.flatten(2).float()
        return [
            flat.abs().mean(dim=2),
            flat.std(dim=2, unbiased=False),
            flat.abs().amax(dim=2),
        ]

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        features = self.reduce(stage3)
        dx = features[..., :, 1:] - features[..., :, :-1]
        dy = features[..., 1:, :] - features[..., :-1, :]
        descriptor = torch.cat(self._moments(dx) + self._moments(dy), dim=1)
        logits = self.classify(descriptor)
        energy = torch.stack(
            (
                dx.float().square().mean(dim=(1, 2, 3)),
                dy.float().square().mean(dim=(1, 2, 3)),
                dx.float().abs().mean(dim=(1, 2, 3)),
                dy.float().abs().mean(dim=(1, 2, 3)),
            ),
            dim=1,
        )
        components = energy / energy.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return _diagnostic_state(
            logits, components.mean(dim=0), _normalized_entropy(components, self.eps)
        )


class RegionGatedAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        channels = int(config.get("region_channels", 48))
        self.eps = 1e-6
        self.reduce = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)
        self.classifier = nn.Linear(channels, num_classes)
        self.gate = nn.Linear(channels, 1)

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        features = self.reduce(stage3)
        quadrants = F.adaptive_avg_pool2d(features, (2, 2)).flatten(2).transpose(1, 2)
        global_descriptor = features.mean(dim=(2, 3)).unsqueeze(1)
        regions = torch.cat((global_descriptor, quadrants), dim=1)
        weights = torch.softmax(self.gate(regions).squeeze(-1), dim=1)
        logits = (weights.unsqueeze(-1) * self.classifier(regions)).sum(dim=1)
        return _diagnostic_state(
            logits, weights.mean(dim=0), _normalized_entropy(weights, self.eps)
        )


class CosinePrototypeAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(config.get("prototype_dim", 64))
        self.num_classes = int(num_classes)
        self.eps = 1e-6
        self.reduce = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.project = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(inplace=True)
        )
        self.prototypes = nn.Parameter(torch.empty(num_classes, hidden))
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(float(config.get("prototype_scale", 10.0))))
        )
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        values = self.reduce(stage3).flatten(2).float()
        descriptor = torch.cat(
            (
                values.mean(dim=2),
                values.std(dim=2, unbiased=False),
                values.amax(dim=2),
            ),
            dim=1,
        )
        embedding = F.normalize(self.project(descriptor), dim=1, eps=self.eps)
        prototypes = F.normalize(self.prototypes.float(), dim=1, eps=self.eps)
        scale = self.logit_scale.float().exp().clamp(1.0, 30.0)
        logits = scale * (embedding @ prototypes.transpose(0, 1))
        probabilities = torch.softmax(logits.detach(), dim=1)
        return _diagnostic_state(
            logits,
            probabilities.mean(dim=0),
            _normalized_entropy(probabilities, self.eps),
        )


class MaskedDescriptorResidualAux(ProjectedAuxHead):
    def __init__(self, in_channels: int, num_classes: int, config: dict[str, Any]) -> None:
        reduced = int(config.get("reduced_channels", 48))
        codewords = int(config.get("num_codewords", 4))
        super().__init__(
            reduced * codewords,
            int(config.get("projection_dim", 64)),
            num_classes,
        )
        self.core = ResidualEncodingCore(
            in_channels=in_channels,
            reduced_channels=reduced,
            num_codewords=codewords,
            assignment_scale=float(config.get("assignment_scale", 1.0)),
            descriptor_dropout=float(config.get("descriptor_dropout", 0.15)),
        )

    def forward(self, stage3: torch.Tensor, stage2: torch.Tensor | None = None):
        descriptor, usage, entropy = self.core(stage3)
        return _diagnostic_state(self.classify(descriptor), usage, entropy)


HEADS = {
    "balanced_sinkhorn": BalancedSinkhornResidualAux,
    "spatial_pyramid": SpatialPyramidResidualAux,
    "bilinear_moment": BilinearMomentAux,
    "highpass_moment": HighPassMomentAux,
    "region_gated": RegionGatedAux,
    "cosine_prototype": CosinePrototypeAux,
    "masked_descriptor": MaskedDescriptorResidualAux,
}


class MobileViTXXSOpt3Aux(MobileViTXXSDFN):
    """LocalECA deployable path with one removable Optimization-3.0 head."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        opt3_variant: str = "balanced_sinkhorn",
        opt3_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if str(kwargs.get("mff_variant", "default")).strip().lower() not in {
            "default",
            "mff",
        }:
            raise ValueError("Optimization-3.0 heads require default MFF")
        if not bool(kwargs.get("use_mff", True)) or not bool(
            kwargs.get("use_stage3_mff", True)
        ):
            raise ValueError("Optimization-3.0 heads require Stage3 MFF")
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            **kwargs,
        )
        config = dict(opt3_config or {})
        stage3_blocks = [
            block
            for block in self.backbone.stages[3]
            if isinstance(block, MFFMobileVitBlock)
        ]
        if len(stage3_blocks) != 1:
            raise RuntimeError("Expected exactly one Stage3 MFF block")
        stage3_block = stage3_blocks[0]
        stage3_channels = int(stage3_block.block.conv_1x1.out_channels)
        stage2_block: MFFMobileVitBlock | None = None
        if opt3_variant == "multi_scale":
            stage2_blocks = [
                block
                for block in self.backbone.stages[2]
                if isinstance(block, MFFMobileVitBlock)
            ]
            if len(stage2_blocks) != 1:
                raise RuntimeError("Expected exactly one Stage2 MFF block")
            stage2_block = stage2_blocks[0]
            stage2_channels = int(stage2_block.block.conv_1x1.out_channels)
            head: nn.Module = MultiScaleResidualAux(
                stage2_channels,
                stage3_channels,
                int(num_classes),
                config,
            )
        else:
            if opt3_variant not in HEADS:
                raise ValueError(f"Unknown Optimization-3.0 variant: {opt3_variant}")
            head = HEADS[opt3_variant](stage3_channels, int(num_classes), config)
        object.__setattr__(self, "opt3_stage3_block", stage3_block)
        object.__setattr__(self, "opt3_stage2_block", stage2_block)
        self.opt3_variant = str(opt3_variant)
        self.opt3_aux_head: nn.Module | None = head
        self.inference_auxiliary_stripped = False

    def prepare_for_inference(self) -> MobileViTXXSOpt3Aux:
        self.opt3_aux_head = None
        for block in (self.opt3_stage2_block, self.opt3_stage3_block):
            if block is not None:
                block.capture_relations = False
                block.last_local_feat = None
                block.last_global_feat = None
        self.inference_auxiliary_stripped = True
        return self

    def _forward_rcaux(self, inputs: torch.Tensor):
        if self.opt3_aux_head is None:
            raise RuntimeError("Optimization-3.0 auxiliary head was stripped")
        blocks = [
            block
            for block in (self.opt3_stage2_block, self.opt3_stage3_block)
            if block is not None
        ]
        for block in blocks:
            block.capture_relations = True
        try:
            features = self.forward_features(inputs)
            stage3 = self.opt3_stage3_block.last_local_feat
            stage2 = (
                self.opt3_stage2_block.last_local_feat
                if self.opt3_stage2_block is not None
                else None
            )
            if stage3 is None or (
                self.opt3_variant == "multi_scale" and stage2 is None
            ):
                raise RuntimeError("Optimization-3.0 feature capture failed")
            logits = self.classifier(self.dropout(features))
            state = self.opt3_aux_head(stage3, stage2)
        finally:
            for block in blocks:
                block.capture_relations = False
                block.last_local_feat = None
                block.last_global_feat = None
        return logits, state

    def forward(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
        return_aux: bool = False,
        return_rcaux: bool = False,
    ):
        if return_aux:
            raise ValueError("Optimization-3.0 heads do not combine with global aux")
        if return_features and return_rcaux:
            raise ValueError("return_features and return_rcaux are exclusive")
        if return_rcaux:
            return self._forward_rcaux(inputs)
        return super().forward(inputs, return_features=return_features)


def mobilevit_xxs_opt3_aux(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSOpt3Aux:
    return MobileViTXXSOpt3Aux(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )


def _variant_builder(variant: str, **kwargs: Any) -> MobileViTXXSOpt3Aux:
    kwargs["opt3_variant"] = variant
    return mobilevit_xxs_opt3_aux(**kwargs)


def mobilevit_xxs_bsrcaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("balanced_sinkhorn", **kwargs)


def mobilevit_xxs_sprcaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("spatial_pyramid", **kwargs)


def mobilevit_xxs_msrcaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("multi_scale", **kwargs)


def mobilevit_xxs_bimaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("bilinear_moment", **kwargs)


def mobilevit_xxs_hpaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("highpass_moment", **kwargs)


def mobilevit_xxs_rgaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("region_gated", **kwargs)


def mobilevit_xxs_cpaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("cosine_prototype", **kwargs)


def mobilevit_xxs_mdrcaux(**kwargs: Any) -> MobileViTXXSOpt3Aux:
    return _variant_builder("masked_descriptor", **kwargs)

