from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import nn

from src.models.mobilevit_xxs_cep import CEPModule
from src.models.mobilevit_xxs_dfn import HighFrequencyResidual


class MobileViTXXSHFCEP(nn.Module):
    """MobileViT-XXS + shallow HF residual + deep CEP modules.

    Composition for ablation:
    - High-frequency residual after stage1 (same as DFN-HF)
    - CEP (CA + ECA + PoolFormer) on stages 2/3/4 and final_conv
    - No MFF / Tail; Model EMA is controlled by training config.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        dropout: float = 0.0,
        use_high_frequency: bool = True,
        ca_reduction: int = 16,
        eca_kernel: int = 3,
        pool_size: int = 3,
        mlp_ratio: float = 2.0,
        use_ca: bool = True,
        use_eca: bool = True,
        use_poolformer: bool = True,
        use_stage2_cep: bool = True,
        use_stage3_cep: bool = True,
        use_stage4_cep: bool = True,
        use_final_cep: bool = True,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("MobileViT-XXS-HF-CEP requires timm.") from error

        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
        )
        feature_dim = int(self.backbone.num_features)

        self.use_high_frequency = bool(use_high_frequency)
        self.high_frequency = (
            HighFrequencyResidual(24)
            if self.use_high_frequency
            else nn.Identity()
        )

        def build_cep(channels: int) -> nn.Module:
            return CEPModule(
                channels=channels,
                ca_reduction=ca_reduction,
                eca_kernel=eca_kernel,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                use_ca=use_ca,
                use_eca=use_eca,
                use_poolformer=use_poolformer,
            )

        self.cep_stage2 = build_cep(48) if use_stage2_cep else nn.Identity()
        self.cep_stage3 = build_cep(64) if use_stage3_cep else nn.Identity()
        self.cep_stage4 = build_cep(80) if use_stage4_cep else nn.Identity()
        self.cep_final = (
            build_cep(feature_dim) if use_final_cep else nn.Identity()
        )

        self.dropout = nn.Dropout(p=float(dropout))
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.num_features = feature_dim

    def forward_features(self, inputs):
        features = self.backbone.stem(inputs)
        features = self.backbone.stages[0](features)
        features = self.backbone.stages[1](features)
        features = self.high_frequency(features)
        features = self.backbone.stages[2](features)
        features = self.cep_stage2(features)
        features = self.backbone.stages[3](features)
        features = self.cep_stage3(features)
        features = self.backbone.stages[4](features)
        features = self.cep_stage4(features)
        features = self.backbone.final_conv(features)
        features = self.cep_final(features)
        return F.adaptive_avg_pool2d(features, 1).flatten(1)

    def forward(
        self,
        inputs,
        return_features: bool = False,
    ):
        features = self.forward_features(inputs)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features
        return logits


def mobilevit_xxs_hf_cep(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
    **kwargs: Any,
) -> MobileViTXXSHFCEP:
    return MobileViTXXSHFCEP(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        **kwargs,
    )
