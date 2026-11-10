from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision import models

from src.models.cg_mta_mobilevit import cg_mta_mobilevit_xxs
from src.models.efficientvim import efficientvim_m1
from src.models.julitemobileatt import julitemobileatt
from src.models.julitemobileatt_paperfaithful import (
    julitemobileatt_paperfaithful,
)
from src.models.jujubenet import jujubenet
from src.models.mobilevit_xxs_deep_se import mobilevit_xxs_deep_se
from src.models.mobilevit_xxs_ls import mobilevit_xxs_ls
from src.models.mobilevit_xxs_afusion import mobilevit_xxs_afusion
from src.models.mobilevit_xxs_cep import mobilevit_xxs_cep
from src.models.mobilevit_xxs_cgrh import mobilevit_xxs_cgrh
from src.models.mobilevit_xxs_cgrh_aux import mobilevit_xxs_cgrh_aux
from src.models.mobilevit_xxs_apd_aux import mobilevit_xxs_apd_aux
from src.models.mobilevit_xxs_aocg_aux import mobilevit_xxs_aocg_aux
from src.models.mobilevit_xxs_odpm_cgrh_trainonly import (
    mobilevit_xxs_odpm_cgrh_trainonly,
)
from src.models.mobilevit_xxs_odpm_cgrh_hard_v2 import (
    mobilevit_xxs_odpm_cgrh_hard_v2,
)
from src.models.mobilevit_xxs_odpm_cgrh_coverage_v3 import (
    mobilevit_xxs_odpm_cgrh_coverage_v3,
)
from src.models.mobilevit_xxs_auto_cgrh import mobilevit_xxs_auto_cgrh
from src.models.mobilevit_xxs_lmc import mobilevit_xxs_lmc
from src.models.mobilevit_xxs_soft_dynamic_cgrh import (
    mobilevit_xxs_soft_dynamic_cgrh,
)
from src.models.mobilevit_xxs_auto_cgrh_fast import (
    mobilevit_xxs_auto_cgrh_fast,
)
from src.models.mobilevit_xxs_dphg import mobilevit_xxs_dphg
from src.models.mobilevit_xxs_dfn import mobilevit_xxs_dfn
from src.models.mobilevit_xxs_lgrc import mobilevit_xxs_lgrc
from src.models.jumcnet import jumcnet
from src.models.mobilevit_xxs_plain_rcaux import mobilevit_xxs_plain_rcaux
from src.models.mobilevit_xxs_rcaux_mha import mobilevit_xxs_rcaux_mha
from src.models.mobilevit_xxs_rcaux_pn import mobilevit_xxs_rcaux_pn
from src.models.mobilevit_xxs_rcaux_paw import mobilevit_xxs_rcaux_paw
from src.models.mobilevit_xxs_rcaux_sa import mobilevit_xxs_rcaux_sa
from src.models.mobilevit_xxs_rcaux_hnm import mobilevit_xxs_rcaux_hnm
from src.models.mobilevit_xxs_rcaux_kexp import mobilevit_xxs_rcaux_kexp
from src.models.mobilevit_xxs_rcaux_m2g import mobilevit_xxs_rcaux_m2g
from src.models.mobilevit_xxs_rcaux_ardrop import mobilevit_xxs_rcaux_ardrop
from src.models.mobilevit_xxs_ufb_rcaux import mobilevit_xxs_ufb_rcaux
from src.models.mobilevit_xxs_opt3_aux import (
    mobilevit_xxs_bimaux,
    mobilevit_xxs_bsrcaux,
    mobilevit_xxs_cpaux,
    mobilevit_xxs_hpaux,
    mobilevit_xxs_mdrcaux,
    mobilevit_xxs_msrcaux,
    mobilevit_xxs_rgaux,
    mobilevit_xxs_sprcaux,
)
from src.models.mobilevit_xxs_headdrop import mobilevit_xxs_headdrop
from src.models.mobilevit_xxs_hf_cep import mobilevit_xxs_hf_cep
from src.models.mobilevit_xxs_mid_bypass import mobilevit_xxs_mid_bypass
from src.models.mobilevit_xxs_sddg import mobilevit_xxs_sddg
from src.models.mobilevit_xxs_ema_mff_mid_bypass import (
    mobilevit_xxs_ema_mff_mid_bypass,
    mobilevit_xxs_ema_mff_mid_sgate,
)
from src.models.freq_aux import (
    julitemobileatt_faf,
    mobilevit_xxs_faf,
    repvit_m0_9_faf,
)


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
        "The classification head cannot be replaced automatically. Add an explicit implementation for this model in src/models."
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
        if model_name in {"cg_mta_mobilevit_xxs", "mta_mobilevit_xxs"}:
            return cg_mta_mobilevit_xxs(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", True)
                ),
                use_morphology=bool(
                    model_config.get("use_morphology", True)
                ),
                use_adaptive_gate=bool(
                    model_config.get("use_adaptive_gate", True)
                ),
                morphology_scale=float(
                    model_config.get("morphology_scale", 0.10)
                ),
                gate_reduction=int(
                    model_config.get("gate_reduction", 8)
                ),
            )
        if model_name in {"mobilevit_xxs_ema_se", "mobilevit_xxs_deep_se"}:
            return mobilevit_xxs_deep_se(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                se_reduction=int(model_config.get("se_reduction", 8)),
                use_stage2_se=bool(model_config.get("use_stage2_se", True)),
                use_stage3_se=bool(model_config.get("use_stage3_se", True)),
                use_stage4_se=bool(model_config.get("use_stage4_se", True)),
                use_final_se=bool(model_config.get("use_final_se", True)),
            )
        if model_name == "mobilevit_xxs_ls":
            return mobilevit_xxs_ls(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                large_kernel=int(model_config.get("large_kernel", 7)),
                small_kernel=int(model_config.get("small_kernel", 3)),
                channels_per_group=int(
                    model_config.get("channels_per_group", 8)
                ),
                use_stage2_ls=bool(model_config.get("use_stage2_ls", True)),
                use_stage3_ls=bool(model_config.get("use_stage3_ls", True)),
                use_stage4_ls=bool(model_config.get("use_stage4_ls", True)),
                use_final_ls=bool(model_config.get("use_final_ls", True)),
            )
        if model_name == "mobilevit_xxs_afusion":
            return mobilevit_xxs_afusion(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                gate_reduction=int(model_config.get("gate_reduction", 4)),
                use_stage3_fusion=bool(
                    model_config.get("use_stage3_fusion", False)
                ),
                use_stage4_fusion=bool(
                    model_config.get("use_stage4_fusion", True)
                ),
            )
        if model_name == "mobilevit_xxs_cep":
            return mobilevit_xxs_cep(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                ca_reduction=int(model_config.get("ca_reduction", 16)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                pool_size=int(model_config.get("pool_size", 3)),
                mlp_ratio=float(model_config.get("mlp_ratio", 2.0)),
                use_ca=bool(model_config.get("use_ca", True)),
                use_eca=bool(model_config.get("use_eca", True)),
                use_poolformer=bool(model_config.get("use_poolformer", True)),
                use_stage2_cep=bool(model_config.get("use_stage2_cep", True)),
                use_stage3_cep=bool(model_config.get("use_stage3_cep", True)),
                use_stage4_cep=bool(model_config.get("use_stage4_cep", True)),
                use_final_cep=bool(model_config.get("use_final_cep", True)),
            )
        if model_name == "mobilevit_xxs_mid_bypass":
            return mobilevit_xxs_mid_bypass(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                bypass_stage=int(model_config.get("bypass_stage", 2)),
                bypass_channels=int(model_config.get("bypass_channels", 64)),
                se_reduction=int(model_config.get("se_reduction", 8)),
                use_se_head=bool(model_config.get("use_se_head", True)),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_lgrc":
            return mobilevit_xxs_lgrc(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                lgrc_hidden_dim=int(
                    model_config.get("lgrc", {}).get("hidden_dim", 128)
                ),
                lgrc_num_pairs=int(
                    model_config.get("lgrc", {}).get("num_pairs", 64)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_headdrop":
            return mobilevit_xxs_headdrop(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=float(model_config.get("head_dropout", 0.10)),
            )
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
            "mobilevit_xxs_ema_mff_local_eca_rcaux_ufb": (
                mobilevit_xxs_ufb_rcaux
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_mha": (
                mobilevit_xxs_rcaux_mha
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_pn": (
                mobilevit_xxs_rcaux_pn
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_paw": (
                mobilevit_xxs_rcaux_paw
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_sa": (
                mobilevit_xxs_rcaux_sa
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_hnm": (
                mobilevit_xxs_rcaux_hnm
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_kexp": (
                mobilevit_xxs_rcaux_kexp
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_m2g": (
                mobilevit_xxs_rcaux_m2g
            ),
            "mobilevit_xxs_ema_mff_local_eca_rcaux_ardrop": (
                mobilevit_xxs_rcaux_ardrop
            ),
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
            if model_name.endswith("_sa"):
                rcaux_kwargs.update(
                    rcaux_saliency_mix=float(
                        rcaux_config.get("saliency_mix", 0.50)
                    ),
                    rcaux_saliency_temperature=float(
                        rcaux_config.get("saliency_temperature", 1.0)
                    ),
                )
            elif model_name.endswith("_m2g"):
                rcaux_kwargs.update(
                    rcaux_second_order_dim=int(
                        rcaux_config.get("second_order_dim", 16)
                    ),
                    rcaux_max_second_order_gate=float(
                        rcaux_config.get("max_second_order_gate", 0.30)
                    ),
                )
            elif model_name.endswith("_ardrop"):
                rcaux_kwargs.update(
                    rcaux_auxiliary_dropout=float(
                        rcaux_config.get("auxiliary_dropout", 0.10)
                    )
                )
            return rcaux_builders[model_name](**rcaux_kwargs)
        opt3_builders = {
            "mobilevit_xxs_ema_mff_local_eca_bsrcaux": (
                mobilevit_xxs_bsrcaux
            ),
            "mobilevit_xxs_ema_mff_local_eca_sprcaux": (
                mobilevit_xxs_sprcaux
            ),
            "mobilevit_xxs_ema_mff_local_eca_bimaux": mobilevit_xxs_bimaux,
            "mobilevit_xxs_ema_mff_local_eca_msrcaux": (
                mobilevit_xxs_msrcaux
            ),
            "mobilevit_xxs_ema_mff_local_eca_rgaux": mobilevit_xxs_rgaux,
            "mobilevit_xxs_ema_mff_local_eca_cpaux": mobilevit_xxs_cpaux,
            "mobilevit_xxs_ema_mff_local_eca_hpaux": mobilevit_xxs_hpaux,
            "mobilevit_xxs_ema_mff_local_eca_mdrcaux": (
                mobilevit_xxs_mdrcaux
            ),
        }
        if model_name in opt3_builders:
            opt3_config = dict(model_config.get("opt3_aux", {}))
            return opt3_builders[model_name](
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                opt3_config=opt3_config,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
            )
        if model_name in {
            "mobilevit_xxs_ema_mff_local_eca_cgrh",
            "mobilevit_xxs_ema_mff_local_eca_apd_cgrh_v1",
        }:
            cgrh_config = dict(model_config.get("cgrh", {}))
            if class_names is None:
                raise ValueError(
                    "CGRH requires class_names when create_model is called"
                )
            pair_mode = str(cgrh_config.get("pair_mode", "explicit")).lower()
            if pair_mode == "all":
                pair_names = [
                    [class_names[first], class_names[second]]
                    for first in range(len(class_names))
                    for second in range(first + 1, len(class_names))
                ]
            elif pair_mode == "explicit":
                pair_names = list(cgrh_config.get("pairs", []))
            else:
                raise ValueError(
                    "cgrh.pair_mode must be either 'explicit' or 'all'; "
                    f"got {pair_mode!r}"
                )
            name_to_index = {
                str(name): index for index, name in enumerate(class_names)
            }
            pair_indices: list[tuple[int, int]] = []
            for pair in pair_names:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(
                        f"Each cgrh.pairs entry must contain 2 names: {pair}"
                    )
                missing = [str(name) for name in pair if str(name) not in name_to_index]
                if missing:
                    raise ValueError(
                        f"Unknown classes in cgrh pair {pair}; missing={missing}"
                    )
                pair_indices.append(
                    (name_to_index[str(pair[0])], name_to_index[str(pair[1])])
                )
            return mobilevit_xxs_cgrh(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                cgrh_pair_indices=pair_indices,
                cgrh_projection_dim=int(
                    cgrh_config.get("projection_dim", 64)
                ),
                cgrh_projection_dropout=float(
                    cgrh_config.get("projection_dropout", 0.0)
                ),
                cgrh_margin_threshold=float(
                    cgrh_config.get("margin_threshold", 0.20)
                ),
                cgrh_refine_scale=float(
                    cgrh_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_cgrh_aux":
            cgrh_aux_config = dict(model_config.get("cgrh_aux", {}))
            if class_names is None:
                raise ValueError("CGRH-Aux requires class_names")
            pair_mode = str(
                cgrh_aux_config.get("pair_mode", "explicit")
            ).lower()
            if pair_mode == "all":
                pair_names = [
                    [class_names[first], class_names[second]]
                    for first in range(len(class_names))
                    for second in range(first + 1, len(class_names))
                ]
            elif pair_mode == "explicit":
                pair_names = list(cgrh_aux_config.get("pairs", []))
            else:
                raise ValueError(
                    "cgrh_aux.pair_mode must be 'explicit' or 'all'; "
                    f"got {pair_mode!r}"
                )
            name_to_index = {
                str(name): index for index, name in enumerate(class_names)
            }
            pair_indices: list[tuple[int, int]] = []
            for pair in pair_names:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Invalid CGRH-Aux pair entry: {pair}")
                missing = [
                    str(name)
                    for name in pair
                    if str(name) not in name_to_index
                ]
                if missing:
                    raise ValueError(
                        "Unknown classes in CGRH-Aux pair "
                        f"{pair}; missing={missing}"
                    )
                pair_indices.append(
                    (name_to_index[str(pair[0])], name_to_index[str(pair[1])])
                )
            return mobilevit_xxs_cgrh_aux(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                cgrh_aux_pair_indices=pair_indices,
                cgrh_aux_projection_dim=int(
                    cgrh_aux_config.get("projection_dim", 64)
                ),
                cgrh_aux_projection_dropout=float(
                    cgrh_aux_config.get("projection_dropout", 0.0)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_apd_aux_v2":
            apd_config = dict(model_config.get("apd_aux", {}))
            if class_names is None:
                raise ValueError("APD-Aux requires class_names")
            pair_mode = str(apd_config.get("pair_mode", "all")).lower()
            if pair_mode == "all":
                pair_names = [
                    [class_names[first], class_names[second]]
                    for first in range(len(class_names))
                    for second in range(first + 1, len(class_names))
                ]
            elif pair_mode == "explicit":
                pair_names = list(apd_config.get("pairs", []))
            else:
                raise ValueError(
                    "apd_aux.pair_mode must be 'explicit' or 'all'; "
                    f"got {pair_mode!r}"
                )
            name_to_index = {
                str(name): index for index, name in enumerate(class_names)
            }
            pair_indices: list[tuple[int, int]] = []
            for pair in pair_names:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Invalid APD pair entry: {pair}")
                missing = [
                    str(name) for name in pair if str(name) not in name_to_index
                ]
                if missing:
                    raise ValueError(
                        f"Unknown classes in APD pair {pair}; missing={missing}"
                    )
                pair_indices.append(
                    (name_to_index[str(pair[0])], name_to_index[str(pair[1])])
                )
            return mobilevit_xxs_apd_aux(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(model_config.get("use_high_frequency", False)),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(model_config.get("use_tail_enhance", False)),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                apd_pair_indices=pair_indices,
                apd_projection_dim=int(apd_config.get("projection_dim", 64)),
                apd_projection_dropout=float(
                    apd_config.get("projection_dropout", 0.0)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_aocg_aux":
            aocg_config = dict(model_config.get("aocg_aux", {}))
            return mobilevit_xxs_aocg_aux(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                aocg_projection_dim=int(
                    aocg_config.get("projection_dim", 64)
                ),
                aocg_projection_dropout=float(
                    aocg_config.get("projection_dropout", 0.0)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_odpm_cgrh_trainonly":
            odpm_config = dict(model_config.get("odpm_cgrh_trainonly", {}))
            return mobilevit_xxs_odpm_cgrh_trainonly(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                odpm_projection_dim=int(
                    odpm_config.get("projection_dim", 64)
                ),
                odpm_projection_dropout=float(
                    odpm_config.get("projection_dropout", 0.0)
                ),
                odpm_margin_threshold=float(
                    odpm_config.get("margin_threshold", 0.20)
                ),
                odpm_refine_scale=float(
                    odpm_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_odpm_cgrh_hard_v2":
            odpm_config = dict(model_config.get("odpm_cgrh_hard_v2", {}))
            return mobilevit_xxs_odpm_cgrh_hard_v2(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                odpm_projection_dim=int(
                    odpm_config.get("projection_dim", 64)
                ),
                odpm_projection_dropout=float(
                    odpm_config.get("projection_dropout", 0.0)
                ),
                odpm_margin_threshold=float(
                    odpm_config.get("probability_margin_threshold", 0.20)
                ),
                odpm_refine_scale=float(
                    odpm_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_odpm_cgrh_coverage_v3":
            odpm_config = dict(
                model_config.get("odpm_cgrh_coverage_v3", {})
            )
            return mobilevit_xxs_odpm_cgrh_coverage_v3(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                odpm_projection_dim=int(
                    odpm_config.get("projection_dim", 64)
                ),
                odpm_projection_dropout=float(
                    odpm_config.get("projection_dropout", 0.0)
                ),
                odpm_margin_threshold=float(
                    odpm_config.get("margin_threshold", 0.20)
                ),
                odpm_refine_scale=float(
                    odpm_config.get("refine_scale", 0.50)
                ),
            )
        if model_name in {
            "mobilevit_xxs_ema_mff_local_eca_continuous_top3_cgrh",
            "mobilevit_xxs_ema_mff_local_eca_continuous_top3_cgrh_swap",
            (
                "mobilevit_xxs_ema_mff_local_eca_"
                "competence_adaptive_top3_cgrh"
            ),
            (
                "mobilevit_xxs_ema_mff_local_eca_"
                "competence_adaptive_top2_cgrh"
            ),
            (
                "mobilevit_xxs_ema_mff_local_eca_"
                "competence_adaptive_top3_cgrh_cdw"
            ),
            (
                "mobilevit_xxs_ema_mff_local_eca_"
                "competence_adaptive_top3_cgrh_refined_only"
            ),
            (
                "mobilevit_xxs_ema_mff_local_eca_"
                "competence_adaptive_top3_cgrh_v2a"
            ),
            "mobilevit_xxs_ema_mff_local_eca_stable_ca_cgrh",
        }:
            stable_ca_cgrh = model_name.endswith("stable_ca_cgrh")
            competence_v2a = model_name.endswith(
                "competence_adaptive_top3_cgrh_v2a"
            )
            competence_cdw = model_name.endswith(
                "competence_adaptive_top3_cgrh_cdw"
            )
            competence_top2 = model_name.endswith(
                "competence_adaptive_top2_cgrh"
            )
            competence_adaptive = (
                "competence_adaptive_top3_cgrh" in model_name
                or competence_top2
                or stable_ca_cgrh
            )
            config_key = (
                "stable_ca_cgrh"
                if stable_ca_cgrh
                else (
                    "competence_adaptive_top3_cgrh_v2a"
                    if competence_v2a
                    else (
                        "competence_adaptive_top3_cgrh_cdw"
                        if competence_cdw
                        else (
                            "competence_adaptive_top2_cgrh"
                            if competence_top2
                            else (
                                "competence_adaptive_top3_cgrh"
                                if competence_adaptive
                                else "continuous_top3_cgrh"
                            )
                        )
                    )
                )
            )
            ct3_config = dict(model_config.get(config_key, {}))
            return mobilevit_xxs_soft_dynamic_cgrh(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                soft_cgrh_projection_dim=int(
                    ct3_config.get("projection_dim", 64)
                ),
                soft_cgrh_projection_dropout=float(
                    ct3_config.get("projection_dropout", 0.0)
                ),
                soft_cgrh_margin_threshold=float(
                    ct3_config.get("margin_threshold", 0.20)
                ),
                soft_cgrh_refine_scale=float(
                    ct3_config.get("refine_scale", 0.50)
                ),
                soft_cgrh_min_route_weight=float(
                    ct3_config.get("min_route_weight", 0.01)
                ),
                soft_cgrh_separate_residual_scales=competence_adaptive,
            )
        if (
            model_name
            == "mobilevit_xxs_ema_mff_local_eca_soft_dynamic_cgrh"
        ):
            soft_config = dict(model_config.get("soft_dynamic_cgrh", {}))
            return mobilevit_xxs_soft_dynamic_cgrh(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                soft_cgrh_projection_dim=int(
                    soft_config.get("projection_dim", 64)
                ),
                soft_cgrh_projection_dropout=float(
                    soft_config.get("projection_dropout", 0.0)
                ),
                soft_cgrh_margin_threshold=float(
                    soft_config.get("margin_threshold", 0.20)
                ),
                soft_cgrh_refine_scale=float(
                    soft_config.get("refine_scale", 0.50)
                ),
                soft_cgrh_min_route_weight=float(
                    soft_config.get("min_route_weight", 0.01)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_auto_cgrh":
            auto_config = dict(model_config.get("auto_cgrh", {}))
            return mobilevit_xxs_auto_cgrh(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                auto_cgrh_projection_dim=int(
                    auto_config.get("projection_dim", 64)
                ),
                auto_cgrh_projection_dropout=float(
                    auto_config.get("projection_dropout", 0.0)
                ),
                auto_cgrh_margin_threshold=float(
                    auto_config.get("margin_threshold", 0.20)
                ),
                auto_cgrh_refine_scale=float(
                    auto_config.get("refine_scale", 0.50)
                ),
            )
        if model_name in {
            "mobilevit_xxs_ema_mff_local_eca_lmc",
            "mobilevit_xxs_ema_mff_local_eca_lmc_nofreeze",
        }:
            lmc_config = dict(model_config.get("lmc", {}))
            return mobilevit_xxs_lmc(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                lmc_projection_dim=int(
                    lmc_config.get("projection_dim", 64)
                ),
                lmc_projection_dropout=float(
                    lmc_config.get("projection_dropout", 0.0)
                ),
                lmc_margin_threshold=float(
                    lmc_config.get("margin_threshold", 0.20)
                ),
                lmc_refine_scale=float(
                    lmc_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_auto_cgrh_ac":
            # A+C variant: same deployable Auto-CGRH model; the A (early
            # freeze) and C (stable-pair gating) behaviour is driven entirely
            # by auto_cgrh loss hyper-parameters in the YAML config.
            auto_config = dict(model_config.get("auto_cgrh", {}))
            return mobilevit_xxs_auto_cgrh(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                auto_cgrh_projection_dim=int(
                    auto_config.get("projection_dim", 64)
                ),
                auto_cgrh_projection_dropout=float(
                    auto_config.get("projection_dropout", 0.0)
                ),
                auto_cgrh_margin_threshold=float(
                    auto_config.get("margin_threshold", 0.20)
                ),
                auto_cgrh_refine_scale=float(
                    auto_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_auto_cgrh_fast":
            auto_config = dict(model_config.get("auto_cgrh", {}))
            configured_max_pairs = auto_config.get("max_active_pairs", "auto")
            if (
                isinstance(configured_max_pairs, str)
                and configured_max_pairs.lower() == "auto"
            ):
                max_pair_cap = int(auto_config.get("max_pair_cap", 8))
                max_active_pairs = min(
                    max_pair_cap,
                    (int(num_classes) + 1) // 2,
                )
            else:
                max_active_pairs = int(configured_max_pairs)
            return mobilevit_xxs_auto_cgrh_fast(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(
                    model_config.get("use_stage2_mff", True)
                ),
                use_stage3_mff=bool(
                    model_config.get("use_stage3_mff", True)
                ),
                use_stage4_mff=bool(
                    model_config.get("use_stage4_mff", True)
                ),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                auto_cgrh_max_pairs=int(
                    max_active_pairs
                ),
                auto_cgrh_projection_dim=int(
                    auto_config.get("projection_dim", 64)
                ),
                auto_cgrh_projection_dropout=float(
                    auto_config.get("projection_dropout", 0.0)
                ),
                auto_cgrh_margin_threshold=float(
                    auto_config.get("margin_threshold", 0.20)
                ),
                auto_cgrh_refine_scale=float(
                    auto_config.get("refine_scale", 0.50)
                ),
            )
        if model_name == "mobilevit_xxs_ema_mff_local_eca_dphg":
            dphg_config = dict(model_config.get("dphg", {}))
            if class_names is None:
                raise ValueError("DPHG requires class_names")
            pair_mode = str(dphg_config.get("pair_mode", "all")).lower()
            if pair_mode == "all":
                pair_names = [
                    [class_names[first], class_names[second]]
                    for first in range(len(class_names))
                    for second in range(first + 1, len(class_names))
                ]
            elif pair_mode == "explicit":
                pair_names = list(dphg_config.get("pairs", []))
            else:
                raise ValueError(
                    "dphg.pair_mode must be 'explicit' or 'all'; "
                    f"got {pair_mode!r}"
                )
            name_to_index = {
                str(name): index for index, name in enumerate(class_names)
            }
            pair_indices: list[tuple[int, int]] = []
            for pair in pair_names:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Invalid DPHG pair entry: {pair}")
                missing = [
                    str(name) for name in pair if str(name) not in name_to_index
                ]
                if missing:
                    raise ValueError(
                        f"Unknown classes in DPHG pair {pair}; missing={missing}"
                    )
                pair_indices.append(
                    (name_to_index[str(pair[0])], name_to_index[str(pair[1])])
                )
            return mobilevit_xxs_dphg(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(model_config.get("use_high_frequency", False)),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(model_config.get("use_tail_enhance", False)),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", True)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
                head_dropout=(
                    float(model_config["head_dropout"])
                    if "head_dropout" in model_config
                    else None
                ),
                dphg_pair_indices=pair_indices,
                dphg_projection_dim=int(
                    dphg_config.get("projection_dim", 64)
                ),
                dphg_projection_dropout=float(
                    dphg_config.get("projection_dropout", 0.0)
                ),
                dphg_margin_threshold=float(
                    dphg_config.get("margin_threshold", 0.20)
                ),
                dphg_refine_scale=float(
                    dphg_config.get("refine_scale", 0.50)
                ),
            )
        if model_name in {
            "mobilevit_xxs_ema_mff_mid_bypass",
            "mobilevit_xxs_ema_mff_mid_sgate",
        }:
            builder = (
                mobilevit_xxs_ema_mff_mid_sgate
                if model_name == "mobilevit_xxs_ema_mff_mid_sgate"
                else mobilevit_xxs_ema_mff_mid_bypass
            )
            return builder(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                bypass_stage=int(model_config.get("bypass_stage", 2)),
                bypass_channels=int(model_config.get("bypass_channels", 64)),
                se_reduction=int(model_config.get("se_reduction", 8)),
                use_se_head=bool(model_config.get("use_se_head", True)),
                use_mid_channel_gate=bool(
                    model_config.get(
                        "use_mid_channel_gate",
                        model_name == "mobilevit_xxs_ema_mff_mid_sgate",
                    )
                ),
                gate_reduction=int(model_config.get("gate_reduction", 4)),
            )
        if model_name in {
            "mobilevit_xxs_hf_ema_cep",
            "mobilevit_xxs_hf_cep",
        }:
            return mobilevit_xxs_hf_cep(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", True)
                ),
                ca_reduction=int(model_config.get("ca_reduction", 16)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                pool_size=int(model_config.get("pool_size", 3)),
                mlp_ratio=float(model_config.get("mlp_ratio", 2.0)),
                use_ca=bool(model_config.get("use_ca", True)),
                use_eca=bool(model_config.get("use_eca", True)),
                use_poolformer=bool(model_config.get("use_poolformer", True)),
                use_stage2_cep=bool(model_config.get("use_stage2_cep", True)),
                use_stage3_cep=bool(model_config.get("use_stage3_cep", True)),
                use_stage4_cep=bool(model_config.get("use_stage4_cep", True)),
                use_final_cep=bool(model_config.get("use_final_cep", True)),
            )
        if model_name in {
            "mobilevit_xxs_dfn",
            "mobilevit_xxs_dfn_ema",
            "mobilevit_xxs_dfn_hf",
            "mobilevit_xxs_dfn_mff",
            "mobilevit_xxs_dfn_tail",
            "mobilevit_xxs_ema_mff",
            "mobilevit_xxs_ema_mff_resfus",
            "mobilevit_xxs_ema_ssg_resfus",
            "mobilevit_xxs_ema_mff_local_eca",
            "mobilevit_xxs_ema_mff_local_eca_headdrop",
            "mobilevit_xxs_ema_mff_local_eca_sam",
            "mobilevit_xxs_ema_mff_local_eca_cdw",
            "mobilevit_xxs_ema_mff_local_eca_ca_lite",
            "mobilevit_xxs_ema_mff_local_eca_dcmg",
            "mobilevit_xxs_ema_mff_local_eca_dcmg_v2",
            "mobilevit_xxs_ema_mff_local_eca_dcmg_cgr_v3",
            "mobilevit_xxs_ema_mff_local_eca_dcmg_cgr_v4",
            "mobilevit_xxs_ema_mff_local_eca_crop085",
            "mobilevit_xxs_ema_mff_local_eca_adak",
            "mobilevit_xxs_ema_mff_local_eca_dualpool",
            "mobilevit_xxs_ema_mff_local_eca_dshead",
            "mobilevit_xxs_ema_mff_local_eca_chgamma",
            "mobilevit_xxs_ema_mff_local_eca_s34",
            "mobilevit_xxs_ema_mff_local_eca_meanstd",
            "mobilevit_xxs_ema_mff_local_eca_meanstd_gate",
            "mobilevit_xxs_ema_mff_local_eca_gapgem_gate",
            "mobilevit_xxs_ema_mff_local_ca",
            "mobilevit_xxs_ema_mff_local_eca_shallow_eca",
            "mobilevit_xxs_ema_mff_local_eca_conf",
            "mobilevit_xxs_ema_mff_local_eca_cdep",
            "mobilevit_xxs_ema_mff_local_rep",
            "mobilevit_xxs_ema_mff_global_aux",
            "mobilevit_xxs_ema_mff_local_eca_global_aux",
            "mobilevit_xxs_ema_mff_local_eca_gem",
            "mobilevit_xxs_ema_local_eca",
            "mobilevit_xxs_ema_local_eca_resfus",
            "mobilevit_xxs_ema_local_eca_gem",
            "mobilevit_xxs_ema_s2mff_s34localeca",
        }:
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
                        "res_concat"
                        if model_name
                        in {
                            "mobilevit_xxs_ema_mff_resfus",
                            "mobilevit_xxs_ema_local_eca_resfus",
                        }
                        else "default",
                    )
                ),
                use_local_eca=bool(
                    model_config.get(
                        "use_local_eca",
                        model_name
                        in {
                            "mobilevit_xxs_ema_mff_local_eca",
                            "mobilevit_xxs_ema_mff_local_eca_headdrop",
                            "mobilevit_xxs_ema_mff_local_eca_sam",
                            "mobilevit_xxs_ema_mff_local_eca_cdw",
                            "mobilevit_xxs_ema_mff_local_eca_ca_lite",
                            "mobilevit_xxs_ema_mff_local_eca_dcmg",
                            "mobilevit_xxs_ema_mff_local_eca_dcmg_v2",
                            "mobilevit_xxs_ema_mff_local_eca_dcmg_cgr_v3",
                            "mobilevit_xxs_ema_mff_local_eca_dcmg_cgr_v4",
                            "mobilevit_xxs_ema_mff_local_eca_crop085",
                            "mobilevit_xxs_ema_mff_local_eca_adak",
                            "mobilevit_xxs_ema_mff_local_eca_dualpool",
                            "mobilevit_xxs_ema_mff_local_eca_dshead",
                            "mobilevit_xxs_ema_mff_local_eca_chgamma",
                            "mobilevit_xxs_ema_mff_local_eca_s34",
                            "mobilevit_xxs_ema_mff_local_eca_meanstd",
                            "mobilevit_xxs_ema_mff_local_eca_meanstd_gate",
                            "mobilevit_xxs_ema_mff_local_eca_gapgem_gate",
                            "mobilevit_xxs_ema_mff_local_eca_shallow_eca",
                            "mobilevit_xxs_ema_mff_local_eca_conf",
                            "mobilevit_xxs_ema_mff_local_eca_global_aux",
                            "mobilevit_xxs_ema_mff_local_eca_gem",
                            "mobilevit_xxs_ema_local_eca",
                            "mobilevit_xxs_ema_local_eca_resfus",
                            "mobilevit_xxs_ema_local_eca_gem",
                            "mobilevit_xxs_ema_s2mff_s34localeca",
                        },
                    )
                ),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_adaptive_eca_kernel=bool(
                    model_config.get(
                        "use_adaptive_eca_kernel",
                        model_name == "mobilevit_xxs_ema_mff_local_eca_adak",
                    )
                ),
                eca_gamma=float(model_config.get("eca_gamma", 2.0)),
                eca_beta=float(model_config.get("eca_beta", 1.0)),
                use_dual_pool_eca=bool(
                    model_config.get(
                        "use_dual_pool_eca",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_dualpool",
                    )
                ),
                use_stage2_local_eca=bool(
                    model_config.get(
                        "use_stage2_local_eca",
                        model_name
                        not in {
                            "mobilevit_xxs_ema_mff_local_eca_s34",
                            "mobilevit_xxs_ema_s2mff_s34localeca",
                        },
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
                        model_name == "mobilevit_xxs_ema_mff_local_ca",
                    )
                ),
                ca_reduction=int(model_config.get("ca_reduction", 32)),
                use_shallow_eca=bool(
                    model_config.get(
                        "use_shallow_eca",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_shallow_eca",
                    )
                ),
                use_stage0_eca=bool(model_config.get("use_stage0_eca", True)),
                use_stage1_eca=bool(model_config.get("use_stage1_eca", True)),
                use_local_rep=bool(
                    model_config.get(
                        "use_local_rep",
                        model_name == "mobilevit_xxs_ema_mff_local_rep",
                    )
                ),
                use_global_aux=bool(
                    model_config.get(
                        "use_global_aux",
                        model_name
                        in {
                            "mobilevit_xxs_ema_mff_global_aux",
                            "mobilevit_xxs_ema_mff_local_eca_global_aux",
                        },
                    )
                ),
                use_gem_pool=bool(
                    model_config.get(
                        "use_gem_pool",
                        model_name
                        in {
                            "mobilevit_xxs_ema_mff_local_eca_gem",
                            "mobilevit_xxs_ema_local_eca_gem",
                        },
                    )
                ),
                gem_p=float(model_config.get("gem_p", 3.0)),
                use_mean_std_pool=bool(
                    model_config.get(
                        "use_mean_std_pool",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_meanstd",
                    )
                ),
                use_mean_std_gate_pool=bool(
                    model_config.get(
                        "use_mean_std_gate_pool",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_meanstd_gate",
                    )
                ),
                use_gap_gem_gate_pool=bool(
                    model_config.get(
                        "use_gap_gem_gate_pool",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_gapgem_gate",
                    )
                ),
                use_class_evidence_pool=bool(
                    model_config.get(
                        "use_class_evidence_pool",
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_cdep",
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
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_dshead",
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
                        model_name
                        == "mobilevit_xxs_ema_mff_local_eca_chgamma",
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
        if model_name in {
            "mobilevit_xxs_ema_mff_local_eca_s34_sddg_s2",
            "mobilevit_xxs_ema_mff_local_eca_s34_sddg_s12",
        }:
            return mobilevit_xxs_sddg(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                sddg_mode=str(
                    model_config.get(
                        "sddg_mode",
                        "s12"
                        if model_name.endswith("_sddg_s12")
                        else "s2",
                    )
                ),
                sddg_channels=int(model_config.get("sddg_channels", 16)),
                use_high_frequency=bool(
                    model_config.get("use_high_frequency", False)
                ),
                use_mff=bool(model_config.get("use_mff", True)),
                use_stage2_mff=bool(model_config.get("use_stage2_mff", True)),
                use_stage3_mff=bool(model_config.get("use_stage3_mff", True)),
                use_stage4_mff=bool(model_config.get("use_stage4_mff", True)),
                use_tail_enhance=bool(
                    model_config.get("use_tail_enhance", False)
                ),
                tail_reduction=int(model_config.get("tail_reduction", 4)),
                mff_variant=str(model_config.get("mff_variant", "default")),
                use_local_eca=bool(model_config.get("use_local_eca", True)),
                eca_kernel=int(model_config.get("eca_kernel", 3)),
                use_stage2_local_eca=bool(
                    model_config.get("use_stage2_local_eca", False)
                ),
                use_stage3_local_eca=bool(
                    model_config.get("use_stage3_local_eca", True)
                ),
                use_stage4_local_eca=bool(
                    model_config.get("use_stage4_local_eca", True)
                ),
            )
        if model_name in {
            "mobilevit_xxs_faf",
            "julitemobileatt_faf",
            "repvit_m0_9_faf",
        }:
            builder = {
                "mobilevit_xxs_faf": mobilevit_xxs_faf,
                "julitemobileatt_faf": julitemobileatt_faf,
                "repvit_m0_9_faf": repvit_m0_9_faf,
            }[model_name]
            return builder(
                num_classes=num_classes,
                pretrained=pretrained,
                dropout=dropout,
                use_freq_branch=bool(
                    model_config.get("use_freq_branch", True)
                ),
                use_aux_head=bool(model_config.get("use_aux_head", True)),
                freq_dim=int(model_config.get("freq_dim", 128)),
                spectrum_size=int(model_config.get("spectrum_size", 112)),
            )
        if model_name == "jujubenet":
            if pretrained:
                raise ValueError(
                    "jujubenet is reproduced under the paper's from-scratch "
                    "protocol; set pretrained=false."
                )
            return jujubenet(
                num_classes=num_classes,
                dropout=dropout,
                drop_path_rate=float(
                    model_config.get("drop_path_rate", 0.1)
                ),
            )
        if model_name == "julitemobileatt":
            if pretrained:
                raise ValueError(
                    "julitemobileatt is reproduced under the paper's "
                    "from-scratch protocol; set pretrained=false."
                )
            return julitemobileatt(
                num_classes=num_classes,
                dropout=dropout,
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
        if model_name == "efficientvim_m1":
            if pretrained:
                raise ValueError(
                    "efficientvim_m1 is currently trained from scratch under the paper comparison protocol. "
                    "Set pretrained to false."
                )
            return efficientvim_m1(
                num_classes=num_classes,
                distillation=False,
            )
        raise ValueError(f"Unsupported custom model: {model_name}")

    if source == "torchvision":
        weights = "DEFAULT" if pretrained else None
        try:
            model = models.get_model(model_name, weights=weights)
        except ValueError as error:
            raise ValueError(
                f"Model '{model_name}' was not found in torchvision. "
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
