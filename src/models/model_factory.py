from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

from src.models.julitemobileatt_paperfaithful import (
    julitemobileatt_paperfaithful,
)
from src.models.mobilevit_xxs_dfn import mobilevit_xxs_dfn
from src.models.jumcnet import jumcnet
from src.models.mobilevit_xxs_plain_rcaux import mobilevit_xxs_plain_rcaux


def _replace_torchvision_classifier(
    model: nn.Module,
    num_classes: int,
    dropout: float,
) -> None:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier.children())
            for index in range(len(layers) - 1, -1, -1):
                if isinstance(layers[index], nn.Linear):
                    in_features = layers[index].in_features
                    if index > 0 and isinstance(layers[index - 1], nn.Dropout):
                        layers[index - 1] = nn.Dropout(p=dropout)
                    layers[index] = nn.Linear(in_features, num_classes)
                    model.classifier = nn.Sequential(*layers)
                    return

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return

    raise ValueError(
        "Cannot replace the classification head automatically. Add an explicit implementation for this model in src/models."
    )


def create_model(
    model_config: dict[str, Any],
    num_classes: int,
    class_names: list[str] | None = None,
) -> nn.Module:
    model_name = str(model_config["model_name"])
    source = str(model_config.get("model_source", "torchvision")).lower()
    pretrained = bool(model_config.get("pretrained", False))
    dropout = float(model_config.get("dropout", 0.0))

    if source == "custom":
        if model_name == "mobilevit_xxs_ema_rcaux":
            forbidden = {
                "use_high_frequency": model_config.get("use_high_frequency", False),
                "use_mff": model_config.get("use_mff", False),
                "use_local_eca": model_config.get("use_local_eca", False),
                "use_tail_enhance": model_config.get("use_tail_enhance", False),
            }
            enabled = [name for name, value in forbidden.items() if bool(value)]
            if enabled:
                raise ValueError(
                    "mobilevit_xxs_ema_rcaux forbids experimental main-path "
                    f"modules: {enabled}"
                )
            rcaux_config = dict(model_config.get("rcaux", {}))
            return mobilevit_xxs_plain_rcaux(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                architecture_name=str(
                    model_config.get("architecture_name", "mobilevit_xxs")
                ),
                rcaux_reduced_channels=int(
                    rcaux_config.get("reduced_channels", 48)
                ),
                rcaux_num_codewords=int(rcaux_config.get("num_codewords", 4)),
                rcaux_projection_dim=int(
                    rcaux_config.get("projection_dim", 64)
                ),
                rcaux_assignment_scale=float(
                    rcaux_config.get("assignment_scale", 1.0)
                ),
            )
        rcaux_builders = {
            "jumcnet": jumcnet,
        }
        if model_name in rcaux_builders:
            rcaux_config = dict(model_config.get("rcaux", {}))
            rcaux_kwargs: dict[str, Any] = {
                "num_classes": num_classes,
                "pretrained": pretrained,
                "dropout": dropout,
                "use_high_frequency": bool(
                    model_config.get("use_high_frequency", False)
                ),
                "use_mff": bool(model_config.get("use_mff", True)),
                "use_stage2_mff": bool(
                    model_config.get("use_stage2_mff", True)
                ),
                "use_stage3_mff": bool(
                    model_config.get("use_stage3_mff", True)
                ),
                "use_stage4_mff": bool(
                    model_config.get("use_stage4_mff", True)
                ),
                "use_tail_enhance": bool(
                    model_config.get("use_tail_enhance", False)
                ),
                "tail_reduction": int(model_config.get("tail_reduction", 4)),
                "mff_variant": str(model_config.get("mff_variant", "default")),
                "use_local_eca": bool(model_config.get("use_local_eca", True)),
                "eca_kernel": int(model_config.get("eca_kernel", 3)),
                "use_stage2_local_eca": bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                "use_stage3_local_eca": bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                "use_stage4_local_eca": bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                "rcaux_reduced_channels": int(
                    rcaux_config.get("reduced_channels", 48)
                ),
                "rcaux_num_codewords": int(
                    rcaux_config.get("num_codewords", 4)
                ),
                "rcaux_projection_dim": int(
                    rcaux_config.get("projection_dim", 64)
                ),
                "rcaux_assignment_scale": float(
                    rcaux_config.get("assignment_scale", 1.0)
                ),
            }
            return rcaux_builders[model_name](**rcaux_kwargs)
        if model_name == "mobilevit_xxs_ema_mff_local_eca":
            return mobilevit_xxs_dfn(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", True)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", True)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(
                    model_config.get(
                        "mff_variant",
                        'default',
                    )
                ),
                use_local_eca=bool(
                    model_config.get(
                        "use_local_eca",
                        True,
                    )
                ),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_adaptive_eca_kernel=bool(
                    model_config.get(
                        "use_adaptive_eca_kernel",
                        False,
                    )
                ),
                eca_gamma=float(model_config.get("eca_gamma", 2.0)),
                eca_beta=float(model_config.get("eca_beta", 1.0)),
                use_dual_pool_eca=bool(
                    model_config.get(
                        "use_dual_pool_eca",
                        False,
                    )
                ),
                use_stage2_local_eca=bool(
                    model_config.get(
                        "use_stage2_local_eca",
                        True,
                    )
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                use_local_ca=bool(
                    model_config.get(
                        "use_local_ca",
                        False,
                    )
                ),
                ca_reduction=int(model_config.get("ca_reduction", 32)),
                use_shallow_eca=bool(
                    model_config.get(
                        "use_shallow_eca",
                        False,
                    )
                ),
                use_stage0_eca=bool(model_config.get("use_stage0_eca", True)),
                use_stage1_eca=bool(model_config.get("use_stage1_eca", True)),
                use_local_rep=bool(
                    model_config.get(
                        "use_local_rep",
                        False,
                    )
                ),
                use_global_aux=bool(
                    model_config.get(
                        "use_global_aux",
                        False,
                    )
                ),
                use_gem_pool=bool(
                    model_config.get(
                        "use_gem_pool",
                        False,
                    )
                ),
                gem_p=float(model_config.get("gem_p", 3.0)),
                use_mean_std_pool=bool(
                    model_config.get(
                        "use_mean_std_pool",
                        False,
                    )
                ),
                use_mean_std_gate_pool=bool(
                    model_config.get(
                        "use_mean_std_gate_pool",
                        False,
                    )
                ),
                use_gap_gem_gate_pool=bool(
                    model_config.get(
                        "use_gap_gem_gate_pool",
                        False,
                    )
                ),
                use_class_evidence_pool=bool(
                    model_config.get(
                        "use_class_evidence_pool",
                        False,
                    )
                ),
                evidence_temperature=float(
                    model_config.get("evidence_temperature", 1.0)
                ),
                evidence_gate_init=float(
                    model_config.get("evidence_gate_init", 0.05)
                ),
                mean_std_eps=float(model_config.get("mean_std_eps", 1e-5)),
                use_dual_scale_head=bool(
                    model_config.get(
                        "use_dual_scale_head",
                        False,
                    )
                ),
                dual_scale_mid_dim=int(
                    model_config.get("dual_scale_mid_dim", 96)
                ),
                dual_scale_hidden=int(
                    model_config.get("dual_scale_hidden", 256)
                ),
                use_mff_channel_gamma=bool(
                    model_config.get(
                        "use_mff_channel_gamma",
                        False,
                    )
                ),
                ssg_resfus_gate_init=float(
                    model_config.get("ssg_resfus_gate_init", 0.10)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
            )
        if model_name == "julitemobileatt_paperfaithful":
            if pretrained:
                raise ValueError(
                    "julitemobileatt_paperfaithful follows the paper's "
                    "from-scratch protocol; set pretrained=false."
                )
            return julitemobileatt_paperfaithful(
                num_classes=num_classes,
                dropout=dropout,
                se_reduction=int(model_config.get("se_reduction", 16)),
                spatial_kernel=int(model_config.get("spatial_kernel", 3)),
            )
        raise ValueError(f"Unsupported custom model: {model_name}")

    if source == "torchvision":
        weights = "DEFAULT" if pretrained else None
        try:
            model = models.get_model(model_name, weights=weights)
        except ValueError as error:
            raise ValueError(
                f"Model '{model_name}' is not available in torchvision. "
                "Set model_source to timm and use a timm model name."
            ) from error
        _replace_torchvision_classifier(model, num_classes, dropout)
        return model

    if source == "timm":
        try:
            import timm
        except ImportError as error:
            raise ImportError("Install timm before using timm models.") from error
        architecture_name = str(
            model_config.get("architecture_name", model_name)
        )
        return timm.create_model(
            architecture_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )

    raise ValueError(f"Unsupported model source: {source}")
