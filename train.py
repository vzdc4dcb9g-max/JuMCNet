from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Establish the cuBLAS environment before importing other GPU-related modules.
from src.reproducibility import reproducibility_settings, set_seed

import torch
from torch import nn

from src.config import add_config_arguments, load_runtime_configs, project_path
from src.datasets import create_dataloaders
from src.engine import evaluate, train_one_epoch
from src.losses import build_training_criterion
from src.models import create_model
from src.utils import (
    append_history,
    average_state_dicts,
    build_optimizer,
    build_scheduler,
    clone_state_dict_cpu,
    experiment_directory,
    save_checkpoint,
    write_json,
    write_yaml,
)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train jujube defect classification models")
    add_config_arguments(parser)
    parser.add_argument("--output-dir", default=None, help="Optional training output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config, experiment_config, model_config, args.train_seed = (
        load_runtime_configs(args)
    )
    if args.epochs is not None:
        experiment_config["epochs"] = args.epochs

    set_seed(args.train_seed, deterministic=True)
    reproducibility = reproducibility_settings()
    print(f"reproducibility={reproducibility}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(experiment_config["amp"]) and device.type == "cuda"
    num_classes = int(dataset_config["num_classes"])

    loaders = create_dataloaders(
        dataset_config,
        experiment_config,
        train_seed=args.train_seed,
        include_test=False,
        model_config=model_config,
    )
    model = create_model(
        model_config,
        num_classes=num_classes,
        class_names=[str(name) for name in dataset_config["class_names"]],
    ).to(device)
    criterion = build_training_criterion(
        experiment_config=experiment_config,
        model_config=model_config,
        dataset_config=dataset_config,
    ).to(device)
    sharpness_aware_config = dict(
        model_config.get("sharpness_aware", {}) or {}
    )
    optimizer = build_optimizer(model, experiment_config)
    scheduler = build_scheduler(optimizer, experiment_config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema_config = model_config.get("ema", {})
    ema_enabled = bool(ema_config.get("enabled", False))
    model_ema = None
    if ema_enabled:
        try:
            from timm.utils import ModelEmaV3
        except ImportError as error:
            raise ImportError("Model EMA requires timm>=1.0") from error
        model_ema = ModelEmaV3(
            model,
            decay=float(ema_config.get("decay", 0.9998)),
            min_decay=float(ema_config.get("min_decay", 0.0)),
            update_after_step=int(
                ema_config.get("update_after_step", 0)
            ),
            use_warmup=bool(ema_config.get("use_warmup", True)),
            warmup_gamma=float(ema_config.get("warmup_gamma", 1.0)),
            warmup_power=float(ema_config.get("warmup_power", 0.75)),
            device=device,
            foreach=True,
        )
        model_ema.module.eval()

    output_dir = experiment_directory(
        dataset_config,
        experiment_config,
        model_config,
        args.train_seed,
    )
    if args.output_dir is not None:
        output_dir = project_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_configs = {
        "dataset": dataset_config,
        "experiment": experiment_config,
        "model": model_config,
        "train_seed": args.train_seed,
        "device": str(device),
        "reproducibility": reproducibility,
    }
    write_yaml(output_dir / "resolved_config.yaml", combined_configs)

    monitor = str(experiment_config["monitor"])
    mode = str(experiment_config["monitor_mode"]).lower()
    best_metric = float("-inf") if mode == "max" else float("inf")
    raw_best_metric = float("-inf") if mode == "max" else float("inf")
    epochs = int(experiment_config["epochs"])
    ema_update_step = 0

    print(f"device={device}")
    print(f"model={model_config['model_name']}")
    print(
        f"train={len(loaders['train'].dataset)}, "
        f"val={len(loaders['val'].dataset)}"
    )
    train_loader = loaders["train"]
    print(
        "loader="
        f"num_workers:{train_loader.num_workers}, "
        f"pin_memory:{train_loader.pin_memory}, "
        f"persistent_workers:{train_loader.persistent_workers}"
    )
    if bool(sharpness_aware_config.get("enabled", False)):
        print(
            "sharpness_aware="
            f"{str(sharpness_aware_config.get('variant', 'sam')).upper()} "
            f"rho={float(sharpness_aware_config.get('rho', 0.02))} "
            f"eta={float(sharpness_aware_config.get('eta', 0.01))}"
        )
    print(f"output={output_dir}")
    print(
        "ema="
        f"{'enabled' if ema_enabled else 'disabled'}"
        + (
            f" decay={float(ema_config.get('decay', 0.9998))}"
            if ema_enabled
            else ""
        )
    )
    last_k_ema = int(experiment_config.get("last_k_ema", 0) or 0)
    topk_ema = int(experiment_config.get("topk_ema", last_k_ema) or 0)
    ema_topk: list[tuple[float, int, dict[str, torch.Tensor]]] | None = None
    if topk_ema > 0 and model_ema is not None:
        ema_topk = []
        print(f"topk_ema={topk_ema} (average best val EMA snapshots; best.pth unchanged)")

    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        set_epoch = getattr(criterion, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            num_classes=num_classes,
            amp=amp_enabled,
            max_batches=args.max_train_batches,
            model_ema=model_ema,
            ema_update_step=ema_update_step,
            sharpness_aware=sharpness_aware_config,
        )
        ema_update_step = int(train_metrics.pop("ema_update_step"))
        raw_val_metrics, _ = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            amp=amp_enabled,
            max_batches=args.max_val_batches,
        )
        if model_ema is not None:
            ema_val_metrics, _ = evaluate(
                model=model_ema.module,
                loader=loaders["val"],
                criterion=nn.CrossEntropyLoss(),
                device=device,
                num_classes=num_classes,
                amp=amp_enabled,
                max_batches=args.max_val_batches,
            )
            val_metrics = ema_val_metrics
            selected_model = model_ema.module
            weight_source = "ema"
        else:
            ema_val_metrics = None
            val_metrics = raw_val_metrics
            selected_model = model
            weight_source = "raw"
        scheduler.step()

        current_metric = float(val_metrics[monitor])
        if ema_topk is not None and model_ema is not None:
            snapshot = (
                current_metric,
                int(epoch),
                clone_state_dict_cpu(model_ema.module.state_dict()),
            )
            if len(ema_topk) < topk_ema:
                ema_topk.append(snapshot)
            else:
                worst_i = 0
                for index, (score, _, _) in enumerate(ema_topk):
                    worse = (
                        score < ema_topk[worst_i][0]
                        if mode == "max"
                        else score > ema_topk[worst_i][0]
                    )
                    if worse:
                        worst_i = index
                worst_score = ema_topk[worst_i][0]
                better = (
                    current_metric > worst_score
                    if mode == "max"
                    else current_metric < worst_score
                )
                if better:
                    ema_topk[worst_i] = snapshot
        checkpoint_eligible = getattr(
            criterion, "checkpoint_eligible", None
        )
        eligible = (
            bool(checkpoint_eligible())
            if callable(checkpoint_eligible)
            else True
        )
        improved = eligible and (
            current_metric > best_metric
            if mode == "max"
            else current_metric < best_metric
        )
        if improved:
            best_metric = current_metric
            save_checkpoint(
                checkpoint_dir / "best.pth",
                selected_model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                combined_configs,
                weight_source=weight_source,
                criterion=criterion,
            )
            if model_ema is not None:
                save_checkpoint(
                    checkpoint_dir / "best_ema.pth",
                    selected_model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metric,
                    combined_configs,
                    weight_source="ema",
                    criterion=criterion,
                )

        raw_current_metric = float(raw_val_metrics[monitor])
        raw_improved = eligible and (
            raw_current_metric > raw_best_metric
            if mode == "max"
            else raw_current_metric < raw_best_metric
        )
        if raw_improved:
            raw_best_metric = raw_current_metric
            if model_ema is not None:
                save_checkpoint(
                    checkpoint_dir / "best_raw.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    raw_best_metric,
                    combined_configs,
                    weight_source="raw",
                    criterion=criterion,
                )

        if bool(experiment_config["save_last"]):
            save_checkpoint(
                checkpoint_dir / "last.pth",
                selected_model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                combined_configs,
                weight_source=weight_source,
                criterion=criterion,
            )
            if model_ema is not None:
                save_checkpoint(
                    checkpoint_dir / "last_raw.pth",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    raw_best_metric,
                    combined_configs,
                    weight_source="raw",
                    criterion=criterion,
                )

        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - training_start
        average_epoch_seconds = elapsed_seconds / epoch
        estimated_total_seconds = average_epoch_seconds * epochs
        remaining_seconds = average_epoch_seconds * (epochs - epoch)
        estimated_finish = datetime.now() + timedelta(
            seconds=remaining_seconds
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "raw_val_loss": raw_val_metrics["loss"],
            "raw_val_accuracy": raw_val_metrics["accuracy"],
            "raw_val_macro_f1": raw_val_metrics["macro_f1"],
            "weight_source": weight_source,
            "is_best": int(improved),
            "epoch_seconds": round(epoch_seconds, 3),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "estimated_total_seconds": round(
                estimated_total_seconds, 3
            ),
            "estimated_remaining_seconds": round(remaining_seconds, 3),
            "estimated_finish": estimated_finish.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        if "classification_loss" in train_metrics:
            history_row["train_classification_loss"] = train_metrics[
                "classification_loss"
            ]
        if "confusion_loss" in train_metrics:
            history_row["train_confusion_loss"] = train_metrics[
                "confusion_loss"
            ]
        if "binary_aux_loss" in train_metrics:
            history_row["train_binary_aux_loss"] = train_metrics[
                "binary_aux_loss"
            ]
        if "lgrc_auxiliary_loss" in train_metrics:
            for metric_name in (
                "lgrc_relation_loss",
                "lgrc_position_loss",
                "lgrc_auxiliary_loss",
                "lgrc_effective_weight",
            ):
                history_row[f"train_{metric_name}"] = train_metrics[
                    metric_name
                ]
        if "rcaux_auxiliary_loss" in train_metrics:
            for metric_name, metric_value in train_metrics.items():
                if metric_name.startswith(("rcaux_", "ufb_")):
                    history_row[f"train_{metric_name}"] = metric_value
        if "opt3_auxiliary_loss" in train_metrics:
            for metric_name, metric_value in train_metrics.items():
                if metric_name.startswith("opt3_"):
                    history_row[f"train_{metric_name}"] = metric_value
        if "cgrh_pair_loss" in train_metrics:
            history_row["train_cgrh_pair_loss"] = train_metrics[
                "cgrh_pair_loss"
            ]
            history_row["train_cgrh_route_rate"] = train_metrics[
                "cgrh_route_rate"
            ]
        if "auto_cgrh_pair_loss" in train_metrics:
            for metric_name in (
                "auto_cgrh_pair_loss",
                "auto_cgrh_active_pair_count",
                "auto_cgrh_pairs_frozen",
                "auto_cgrh_supervision_rate",
                "auto_cgrh_route_rate",
                "auto_cgrh_refine_blend",
                "auto_cgrh_effective_pair_weight",
            ):
                history_key = (
                    f"train_{metric_name}"
                    if metric_name.endswith("_loss")
                    else metric_name
                )
                history_row[history_key] = train_metrics[metric_name]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["auto_cgrh_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["auto_cgrh_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "lmc_pair_loss" in train_metrics:
            for metric_name in (
                "lmc_pair_loss",
                "lmc_active_pair_count",
                "lmc_core_hard_pair_count",
                "lmc_eligible_hard_pair_count",
                "lmc_pairs_frozen",
                "lmc_supervision_rate",
                "lmc_route_rate",
                "lmc_refine_blend",
                "lmc_effective_pair_weight",
            ):
                history_key = (
                    f"train_{metric_name}"
                    if metric_name.endswith("_loss")
                    else metric_name
                )
                history_row[history_key] = train_metrics[metric_name]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["lmc_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["lmc_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "soft_cgrh_pair_loss" in train_metrics:
            for metric_name in (
                "soft_cgrh_pair_loss",
                "soft_cgrh_refine_loss",
                "soft_cgrh_effective_pair_weight",
                "soft_cgrh_effective_refine_weight",
                "soft_cgrh_difficulty_scale",
                "soft_cgrh_pair_entropy",
                "soft_cgrh_max_pair_share",
                "soft_cgrh_supervision_rate",
                "soft_cgrh_route_rate",
            ):
                history_row[f"train_{metric_name}"] = train_metrics[
                    metric_name
                ]
            pair_records = getattr(criterion, "pair_records", lambda: [])()
            history_row["soft_cgrh_top_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in pair_records[:5]
            )
            history_row["soft_cgrh_top_weights"] = "|".join(
                f"{float(record['weight']):.6f}"
                for record in pair_records[:5]
            )
        if "ct3_cgrh_pair_loss" in train_metrics:
            for metric_name in (
                "ct3_cgrh_pair_loss",
                "ct3_cgrh_effective_auxiliary_weight",
                "ct3_cgrh_active_pair_count",
                "ct3_cgrh_supervision_weight_sum",
                "ct3_cgrh_handoff_progress",
                "ct3_cgrh_supervision_rate",
                "ct3_cgrh_route_rate",
                "ct3_cgrh_top_pair_weight",
            ):
                history_row[f"train_{metric_name}"] = train_metrics[
                    metric_name
                ]
            pair_records = getattr(criterion, "pair_records", lambda: [])()
            history_row["ct3_cgrh_top_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in pair_records[:5]
            )
            history_row["ct3_cgrh_top_weights"] = "|".join(
                f"{float(record['weight']):.6f}"
                for record in pair_records[:5]
            )
            for metric_name in (
                "ca_cgrh_main_ce",
                "ca_cgrh_refined_ce",
                "ca_cgrh_mean_scale",
                "ca_cgrh_max_scale",
                "ca_cgrh_mean_advantage",
                "ca_cgrh_effective_samples",
            ):
                if metric_name in train_metrics:
                    history_row[f"train_{metric_name}"] = train_metrics[
                        metric_name
                    ]
            for metric_name in (
                "v2a_candidate_supervision_rate",
                "v2a_effective_supervision_rate",
                "v2a_mean_candidate_weight",
                "v2a_competence_gate_mean",
                "v2a_positive_competence_pairs",
            ):
                if metric_name in train_metrics:
                    history_row[f"train_{metric_name}"] = train_metrics[
                        metric_name
                    ]
            competence_records = getattr(
                criterion, "competence_records", lambda: []
            )()
            if competence_records:
                active_competence = [
                    record
                    for record in competence_records
                    if bool(record["active"])
                ]
                history_row["ca_cgrh_pair_scales"] = "|".join(
                    (
                        f"{record['class_i']}~{record['class_j']}:"
                        f"{float(record['residual_scale']):.6f}"
                    )
                    for record in active_competence
                )
        if "cgrh_aux_pair_loss" in train_metrics:
            history_row["train_cgrh_aux_pair_loss"] = train_metrics[
                "cgrh_aux_pair_loss"
            ]
            history_row["cgrh_aux_supervision_rate"] = train_metrics[
                "cgrh_aux_supervision_rate"
            ]
        if "apd_pair_loss" in train_metrics:
            history_row["train_apd_pair_loss"] = train_metrics[
                "apd_pair_loss"
            ]
        if "aocg_pair_loss" in train_metrics:
            history_row["train_aocg_pair_loss"] = train_metrics[
                "aocg_pair_loss"
            ]
            history_row["aocg_active_pair_count"] = train_metrics[
                "aocg_active_pair_count"
            ]
            history_row["aocg_supervision_rate"] = train_metrics[
                "aocg_supervision_rate"
            ]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["aocg_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["aocg_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "odpm_cgrh_pair_loss" in train_metrics:
            history_row["train_odpm_cgrh_pair_loss"] = train_metrics[
                "odpm_cgrh_pair_loss"
            ]
            history_row["train_odpm_cgrh_refine_loss"] = train_metrics[
                "odpm_cgrh_refine_loss"
            ]
            history_row["odpm_cgrh_active_pair_count"] = train_metrics[
                "odpm_cgrh_active_pair_count"
            ]
            history_row["odpm_cgrh_supervision_rate"] = train_metrics[
                "odpm_cgrh_supervision_rate"
            ]
            history_row["odpm_cgrh_route_rate"] = train_metrics[
                "odpm_cgrh_route_rate"
            ]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["odpm_cgrh_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["odpm_cgrh_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "odpm_hard_v2_pair_loss" in train_metrics:
            for metric_name in (
                "odpm_hard_v2_pair_loss",
                "odpm_hard_v2_refine_loss",
                "odpm_hard_v2_active_pair_count",
                "odpm_hard_v2_active_class_rate",
                "odpm_hard_v2_top2_match_rate",
                "odpm_hard_v2_low_margin_rate",
                "odpm_hard_v2_supervision_rate",
                "odpm_hard_v2_selected_samples",
            ):
                history_key = (
                    f"train_{metric_name}"
                    if metric_name.endswith("_loss")
                    else metric_name
                )
                history_row[history_key] = train_metrics[metric_name]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["odpm_hard_v2_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["odpm_hard_v2_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "coverage_v3_pair_loss" in train_metrics:
            for metric_name in (
                "coverage_v3_pair_loss",
                "coverage_v3_effective_pair_weight",
                "coverage_v3_active_pair_count",
                "coverage_v3_supervision_rate",
                "coverage_v3_supervision_instance_rate",
                "coverage_v3_mean_pair_multiplicity",
                "coverage_v3_pairs_frozen",
            ):
                history_key = (
                    f"train_{metric_name}"
                    if metric_name.endswith("_loss")
                    else metric_name
                )
                history_row[history_key] = train_metrics[metric_name]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["coverage_v3_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["coverage_v3_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "dcmg_margin_loss" in train_metrics:
            history_row["train_dcmg_margin_loss"] = train_metrics[
                "dcmg_margin_loss"
            ]
            history_row["dcmg_active_pair_count"] = train_metrics[
                "dcmg_active_pair_count"
            ]
            history_row["dcmg_supervision_rate"] = train_metrics[
                "dcmg_supervision_rate"
            ]
            history_row["dcmg_violation_rate"] = train_metrics[
                "dcmg_violation_rate"
            ]
            history_row["dcmg_mean_logit_gap"] = train_metrics[
                "dcmg_mean_logit_gap"
            ]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["dcmg_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["dcmg_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "dcmg_v2_margin_loss" in train_metrics:
            history_row["train_dcmg_v2_margin_loss"] = train_metrics[
                "dcmg_v2_margin_loss"
            ]
            for metric_name in (
                "active_pair_count",
                "candidate_rate",
                "teacher_reliable_rate",
                "main_close_rate",
                "supervision_rate",
                "violation_rate",
                "mean_logit_gap",
                "effective_pair_weight",
            ):
                history_row[f"dcmg_v2_{metric_name}"] = train_metrics[
                    f"dcmg_v2_{metric_name}"
                ]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["dcmg_v2_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["dcmg_v2_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "dcmg_cgr_v3_pair_loss" in train_metrics:
            history_row["train_dcmg_cgr_v3_pair_loss"] = train_metrics[
                "dcmg_cgr_v3_pair_loss"
            ]
            for metric_name in (
                "active_pair_count",
                "active_class_rate",
                "top2_pair_match_rate",
                "low_margin_rate",
                "supervision_rate",
                "mean_probability_gap",
                "mean_difficulty_weight",
                "effective_pair_weight",
                "auto_frozen",
            ):
                history_row[f"dcmg_cgr_v3_{metric_name}"] = train_metrics[
                    f"dcmg_cgr_v3_{metric_name}"
                ]
            active_pair_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["dcmg_cgr_v3_active_pairs"] = "|".join(
                f"{record['class_i']}~{record['class_j']}"
                for record in active_pair_records
            )
            history_row["dcmg_cgr_v3_pair_roles"] = "|".join(
                str(record.get("role", "active"))
                for record in active_pair_records
            )
            history_row["dcmg_cgr_v3_active_pair_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_pair_records
            )
        if "dcmg_cgr_v4_pair_loss" in train_metrics:
            history_row["train_dcmg_cgr_v4_pair_loss"] = train_metrics[
                "dcmg_cgr_v4_pair_loss"
            ]
            for metric_name in (
                "active_edge_count",
                "coverage_rate",
                "hard_route_rate",
                "mean_sample_weight",
                "effective_supervision_mass",
                "effective_pair_weight",
                "auto_frozen",
            ):
                history_row[f"dcmg_cgr_v4_{metric_name}"] = train_metrics[
                    f"dcmg_cgr_v4_{metric_name}"
                ]
            active_edge_records = getattr(
                criterion, "active_pair_records", lambda: []
            )()
            history_row["dcmg_cgr_v4_active_edges"] = "|".join(
                f"{record['true_class']}->{record['competitor_class']}"
                for record in active_edge_records
            )
            history_row["dcmg_cgr_v4_edge_roles"] = "|".join(
                str(record.get("role", "active"))
                for record in active_edge_records
            )
            history_row["dcmg_cgr_v4_edge_scores"] = "|".join(
                f"{float(record['score']):.6f}"
                for record in active_edge_records
            )
        if "dphg_pair_loss" in train_metrics:
            history_row["train_dphg_pair_loss"] = train_metrics[
                "dphg_pair_loss"
            ]
            history_row["train_dphg_refine_loss"] = train_metrics[
                "dphg_refine_loss"
            ]
            history_row["train_dphg_route_rate"] = train_metrics[
                "dphg_route_rate"
            ]
            history_row["train_dphg_supervised_route_rate"] = train_metrics[
                "dphg_supervised_route_rate"
            ]
        current_weights = getattr(criterion, "current_weights", None)
        if callable(current_weights):
            values = current_weights().detach().cpu().tolist()
            history_row["difficulty_weights"] = "|".join(
                f"{float(value):.4f}" for value in values
            )
        if ema_val_metrics is not None:
            history_row["ema_val_loss"] = ema_val_metrics["loss"]
            history_row["ema_val_accuracy"] = ema_val_metrics["accuracy"]
            history_row["ema_val_macro_f1"] = ema_val_metrics["macro_f1"]
        append_history(output_dir / "train_history.csv", history_row)
        graph_state = getattr(criterion, "graph_state", None)
        if callable(graph_state):
            graph_history_filename = str(
                getattr(
                    criterion,
                    "graph_history_filename",
                    "aocg_pair_history.json",
                )
            )
            write_json(
                output_dir / graph_history_filename,
                graph_state(),
            )
        auxiliary_text = ""
        if "confusion_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" conf_loss={train_metrics['confusion_loss']:.4f}"
            )
        if "binary_aux_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics.get('classification_loss', train_metrics['loss']):.4f}"
                f" aux_loss={train_metrics['binary_aux_loss']:.4f}"
            )
        if "lgrc_auxiliary_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" lgrc_rel={train_metrics['lgrc_relation_loss']:.4f}"
                f" lgrc_pos={train_metrics['lgrc_position_loss']:.4f}"
                f" lgrc_w={train_metrics['lgrc_effective_weight']:.4f}"
            )
        if "rcaux_auxiliary_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" rcaux_loss={train_metrics['rcaux_auxiliary_loss']:.4f}"
                f" rcaux_w={train_metrics['rcaux_effective_weight']:.4f}"
                f" code_entropy={train_metrics['rcaux_assignment_entropy']:.3f}"
                f" usage_min={train_metrics['rcaux_usage_min']:.3f}"
            )
        if "opt3_auxiliary_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" opt3_loss={train_metrics['opt3_auxiliary_loss']:.4f}"
                f" opt3_w={train_metrics['opt3_effective_weight']:.4f}"
                f" component_entropy="
                f"{train_metrics['opt3_component_entropy']:.3f}"
                f" usage_min={train_metrics['opt3_usage_min']:.3f}"
            )
        if "cgrh_pair_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" pair_loss={train_metrics['cgrh_pair_loss']:.4f}"
                f" route={train_metrics['cgrh_route_rate']:.3f}"
            )
        if "auto_cgrh_pair_loss" in train_metrics:
            pair_names = history_row.get(
                "auto_cgrh_active_pairs", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" auto_pair={train_metrics['auto_cgrh_pair_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" frozen={int(train_metrics['auto_cgrh_pairs_frozen'])}"
                f" supervised={train_metrics['auto_cgrh_supervision_rate']:.3f}"
                f" route={train_metrics['auto_cgrh_route_rate']:.3f}"
            )
        if "lmc_pair_loss" in train_metrics:
            pair_names = history_row.get("lmc_active_pairs", "") or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" lmc_pair={train_metrics['lmc_pair_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" frozen={int(train_metrics['lmc_pairs_frozen'])}"
                f" supervised={train_metrics['lmc_supervision_rate']:.3f}"
                f" route={train_metrics['lmc_route_rate']:.3f}"
            )
        if "soft_cgrh_pair_loss" in train_metrics:
            top_pairs = history_row.get("soft_cgrh_top_pairs", "") or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" soft_pair={train_metrics['soft_cgrh_pair_loss']:.4f}"
                f" pair_w={train_metrics['soft_cgrh_effective_pair_weight']:.3f}"
                f" diff={train_metrics['soft_cgrh_difficulty_scale']:.3f}"
                f" top_pairs={top_pairs}"
                f" route={train_metrics['soft_cgrh_route_rate']:.3f}"
            )
        if "ct3_cgrh_pair_loss" in train_metrics:
            top_pairs = history_row.get("ct3_cgrh_top_pairs", "") or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" ct3_pair={train_metrics['ct3_cgrh_pair_loss']:.4f}"
                f" aux_w={train_metrics['ct3_cgrh_effective_auxiliary_weight']:.3f}"
                f" active={int(train_metrics['ct3_cgrh_active_pair_count'])}"
                f" top_pairs={top_pairs}"
                f" route={train_metrics['ct3_cgrh_route_rate']:.3f}"
            )
            if "ca_cgrh_mean_scale" in train_metrics:
                auxiliary_text += (
                    f" scale={train_metrics['ca_cgrh_mean_scale']:.3f}"
                    f"/{train_metrics['ca_cgrh_max_scale']:.3f}"
                    f" advantage={train_metrics['ca_cgrh_mean_advantage']:.3f}"
                    f" evidence={train_metrics['ca_cgrh_effective_samples']:.1f}"
                )
        if "cgrh_aux_pair_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" cgrh_aux_pair={train_metrics['cgrh_aux_pair_loss']:.4f}"
                f" supervised={train_metrics['cgrh_aux_supervision_rate']:.3f}"
            )
        if "apd_pair_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" apd_pair_loss={train_metrics['apd_pair_loss']:.4f}"
            )
        if "aocg_pair_loss" in train_metrics:
            pair_names = history_row.get("aocg_active_pairs", "") or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" aocg_pair={train_metrics['aocg_pair_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" supervised={train_metrics['aocg_supervision_rate']:.3f}"
            )
        if "odpm_cgrh_pair_loss" in train_metrics:
            pair_names = history_row.get(
                "odpm_cgrh_active_pairs", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" odpm_pair={train_metrics['odpm_cgrh_pair_loss']:.4f}"
                f" odpm_refine={train_metrics['odpm_cgrh_refine_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" supervised={train_metrics['odpm_cgrh_supervision_rate']:.3f}"
                f" route={train_metrics['odpm_cgrh_route_rate']:.3f}"
            )
        if "odpm_hard_v2_pair_loss" in train_metrics:
            pair_names = history_row.get(
                "odpm_hard_v2_active_pairs", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" hard_pair={train_metrics['odpm_hard_v2_pair_loss']:.4f}"
                f" hard_refine={train_metrics['odpm_hard_v2_refine_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" active_cls={train_metrics['odpm_hard_v2_active_class_rate']:.3f}"
                f" top2={train_metrics['odpm_hard_v2_top2_match_rate']:.3f}"
                f" low_margin={train_metrics['odpm_hard_v2_low_margin_rate']:.3f}"
                f" supervised={train_metrics['odpm_hard_v2_supervision_rate']:.3f}"
            )
        if "coverage_v3_pair_loss" in train_metrics:
            pair_names = history_row.get(
                "coverage_v3_active_pairs", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" coverage_pair={train_metrics['coverage_v3_pair_loss']:.4f}"
                f" weight={train_metrics['coverage_v3_effective_pair_weight']:.4f}"
                f" active_pairs={pair_names}"
                f" supervised={train_metrics['coverage_v3_supervision_rate']:.3f}"
                f" instances={train_metrics['coverage_v3_supervision_instance_rate']:.3f}"
                f" multiplicity={train_metrics['coverage_v3_mean_pair_multiplicity']:.3f}"
                f" frozen={int(train_metrics['coverage_v3_pairs_frozen'])}"
            )
        if "dcmg_margin_loss" in train_metrics:
            pair_names = history_row.get("dcmg_active_pairs", "") or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" dcmg_margin={train_metrics['dcmg_margin_loss']:.4f}"
                f" active_pairs={pair_names}"
                f" supervised={train_metrics['dcmg_supervision_rate']:.3f}"
                f" violation={train_metrics['dcmg_violation_rate']:.3f}"
                f" gap={train_metrics['dcmg_mean_logit_gap']:.3f}"
            )
        if "dcmg_v2_margin_loss" in train_metrics:
            pair_names = history_row.get(
                "dcmg_v2_active_pairs", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" dcmg_v2={train_metrics['dcmg_v2_margin_loss']:.4f}"
                f" weight={train_metrics['dcmg_v2_effective_pair_weight']:.4f}"
                f" active_pairs={pair_names}"
                f" candidate={train_metrics['dcmg_v2_candidate_rate']:.3f}"
                f" reliable={train_metrics['dcmg_v2_teacher_reliable_rate']:.3f}"
                f" close={train_metrics['dcmg_v2_main_close_rate']:.3f}"
                f" supervised={train_metrics['dcmg_v2_supervision_rate']:.3f}"
                f" violation={train_metrics['dcmg_v2_violation_rate']:.3f}"
                f" gap={train_metrics['dcmg_v2_mean_logit_gap']:.3f}"
            )
        if "dcmg_cgr_v3_pair_loss" in train_metrics:
            pair_names = history_row.get(
                "dcmg_cgr_v3_active_pairs", ""
            ) or "none"
            pair_roles = history_row.get(
                "dcmg_cgr_v3_pair_roles", ""
            ) or "none"
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" dcmg_cgr={train_metrics['dcmg_cgr_v3_pair_loss']:.4f}"
                f" weight={train_metrics['dcmg_cgr_v3_effective_pair_weight']:.4f}"
                f" active_pairs={pair_names}"
                f" roles={pair_roles}"
                f" top2_match={train_metrics['dcmg_cgr_v3_top2_pair_match_rate']:.3f}"
                f" supervised={train_metrics['dcmg_cgr_v3_supervision_rate']:.3f}"
                f" prob_gap={train_metrics['dcmg_cgr_v3_mean_probability_gap']:.3f}"
                f" frozen={int(train_metrics['dcmg_cgr_v3_auto_frozen'])}"
            )
        if "dcmg_cgr_v4_pair_loss" in train_metrics:
            active_edges_text = history_row.get(
                "dcmg_cgr_v4_active_edges", ""
            )
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" dcmg_cgr_v4={train_metrics['dcmg_cgr_v4_pair_loss']:.4f}"
                f" weight={train_metrics['dcmg_cgr_v4_effective_pair_weight']:.4f}"
                f" coverage={train_metrics['dcmg_cgr_v4_coverage_rate']:.3f}"
                f" hard={train_metrics['dcmg_cgr_v4_hard_route_rate']:.3f}"
                f" mass={train_metrics['dcmg_cgr_v4_effective_supervision_mass']:.3f}"
                f" frozen={int(train_metrics['dcmg_cgr_v4_auto_frozen'])}"
                f" edges={active_edges_text or '-'}"
            )
        if "dphg_pair_loss" in train_metrics:
            auxiliary_text = (
                f" cls_loss={train_metrics['classification_loss']:.4f}"
                f" dphg_pair={train_metrics['dphg_pair_loss']:.4f}"
                f" dphg_refine={train_metrics['dphg_refine_loss']:.4f}"
                f" route={train_metrics['dphg_route_rate']:.3f}"
                f" supervised={train_metrics['dphg_supervised_route_rate']:.3f}"
            )
        if "difficulty_weight_max" in train_metrics:
            auxiliary_text += (
                f" dw=[{train_metrics['difficulty_weight_min']:.3f},"
                f"{train_metrics['difficulty_weight_max']:.3f}]"
            )
        ema_text = ""
        if ema_val_metrics is not None:
            ema_text = (
                f" raw_val_f1={raw_val_metrics['macro_f1']:.4f}"
                f" ema_val_f1={ema_val_metrics['macro_f1']:.4f}"
            )
        print(
            f"[{epoch:03d}/{epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"time={format_duration(epoch_seconds)} "
            f"elapsed={format_duration(elapsed_seconds)} "
            f"total_est={format_duration(estimated_total_seconds)} "
            f"remaining={format_duration(remaining_seconds)} "
            f"finish={estimated_finish:%Y-%m-%d %H:%M:%S}"
            f"{auxiliary_text}"
            f"{ema_text}"
            f"{' *best' if improved else ''}"
        )

    total_seconds = time.perf_counter() - training_start
    print(f"total_time={format_duration(total_seconds)}")
    print(f"best_weight_source={weight_source}")
    print(f"Training complete. Best validation {monitor}={best_metric:.6f}")
    print(f"Best checkpoint: {checkpoint_dir / 'best.pth'}")

    if ema_topk and model_ema is not None:
        def _val_summary(metrics: dict) -> dict[str, float]:
            return {
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
            }

        ranked = sorted(
            ema_topk,
            key=lambda item: item[0],
            reverse=(mode == "max"),
        )
        averaged = average_state_dicts([item[2] for item in ranked])
        topk_epochs = [int(item[1]) for item in ranked]
        topk_scores = [float(item[0]) for item in ranked]
        model_ema.module.load_state_dict(averaged)
        save_checkpoint(
            checkpoint_dir / "topk_ema.pth",
            model_ema.module,
            optimizer,
            scheduler,
            int(topk_epochs[0]),
            best_metric,
            combined_configs,
            weight_source="ema_topk",
            criterion=criterion,
        )
        topk_metrics, _ = evaluate(
            model=model_ema.module,
            loader=loaders["val"],
            criterion=nn.CrossEntropyLoss(),
            device=device,
            num_classes=num_classes,
            amp=amp_enabled,
            max_batches=args.max_val_batches,
        )
        comparison: dict = {
            "split": "val",
            "note": (
                "best.pth is selected on this same val split, so it is expected "
                "to look equal or better here; compare test.py for generalization. "
                "topk_ema averages the k highest-val EMA snapshots, not the final k epochs."
            ),
            "topk": topk_ema,
            "topk_epochs": topk_epochs,
            "topk_val_scores": topk_scores,
            "topk_ema": _val_summary(topk_metrics),
            "topk_path": str(checkpoint_dir / "topk_ema.pth"),
            "best_path": str(checkpoint_dir / "best.pth"),
        }
        best_path = checkpoint_dir / "best.pth"
        if best_path.is_file():
            best_ckpt = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            model_ema.module.load_state_dict(best_ckpt["model_state"])
            best_metrics, _ = evaluate(
                model=model_ema.module,
                loader=loaders["val"],
                criterion=nn.CrossEntropyLoss(),
                device=device,
                num_classes=num_classes,
                amp=amp_enabled,
                max_batches=args.max_val_batches,
            )
            comparison["best"] = _val_summary(best_metrics)
            comparison["best_epoch"] = int(best_ckpt.get("epoch", -1))
            comparison["val_macro_f1_winner"] = (
                "best"
                if comparison["best"]["macro_f1"]
                >= comparison["topk_ema"]["macro_f1"]
                else "topk_ema"
            )
            print(
                "val_compare "
                f"best_f1={comparison['best']['macro_f1']:.6f} "
                f"(ep{comparison['best_epoch']}) "
                f"topk_f1={comparison['topk_ema']['macro_f1']:.6f} "
                f"(ep{topk_epochs}) "
                f"winner={comparison['val_macro_f1_winner']}"
            )
        write_json(output_dir / "val_checkpoint_compare.json", comparison)
        print(f"Top-k EMA checkpoint: {checkpoint_dir / 'topk_ema.pth'}")
    print("Run test.py separately for test-set evaluation.")


if __name__ == "__main__":
    main()
