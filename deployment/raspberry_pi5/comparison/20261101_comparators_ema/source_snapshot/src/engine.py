from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from src.metrics import ClassificationMeter
from src.sharpness import (
    SharpnessAwarePerturbation,
    disable_batchnorm_running_stats,
    parse_sharpness_config,
    unscale_first_pass_gradients,
    update_scaler_after_skipped_first_pass,
)


def _autocast_context(device: torch.device, amp: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_classes: int,
    amp: bool,
    max_batches: int | None = None,
    model_ema: Any | None = None,
    ema_update_step: int = 0,
    sharpness_aware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.train()
    meter = ClassificationMeter(num_classes)
    component_sums: dict[str, float] = {}
    component_samples = 0
    sharpness_config = parse_sharpness_config(sharpness_aware)
    perturbation = (
        SharpnessAwarePerturbation(optimizer, sharpness_config)
        if sharpness_config.enabled
        else None
    )
    if perturbation is not None and not isinstance(criterion, nn.CrossEntropyLoss):
        raise TypeError(
            "SAM/ASAM currently supports the plain CrossEntropyLoss LocalECA "
            "path only; disable auxiliary/confusion losses."
        )
    sharpness_grad_norm_sum = 0.0
    sharpness_second_loss_sum = 0.0
    sharpness_batches = 0

    for batch_index, (images, targets, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            if bool(
                getattr(
                    criterion,
                    "requires_soft_dynamic_cgrh",
                    False,
                )
            ) or bool(
                getattr(
                    criterion,
                    "requires_continuous_top3_cgrh",
                    False,
                )
            ) or bool(
                getattr(
                    criterion,
                    "requires_competence_adaptive_top3_cgrh",
                    False,
                )
            ):
                pair_weights = criterion.pair_weight_vector()
                set_weights = getattr(
                    model,
                    "set_soft_cgrh_pair_weights",
                    None,
                )
                if not callable(set_weights):
                    raise RuntimeError(
                        "Soft Dynamic CGRH criterion requires a "
                        "compatible model"
                    )
                set_weights(pair_weights)
                if model_ema is not None:
                    ema_set_weights = getattr(
                        model_ema.module,
                        "set_soft_cgrh_pair_weights",
                        None,
                    )
                    if callable(ema_set_weights):
                        ema_set_weights(pair_weights)

                residual_scale_vector = getattr(
                    criterion,
                    "residual_scale_vector",
                    None,
                )
                if callable(residual_scale_vector):
                    residual_scales = residual_scale_vector()
                    set_scales = getattr(
                        model,
                        "set_soft_cgrh_residual_scales",
                        None,
                    )
                    if not callable(set_scales):
                        raise RuntimeError(
                            "Competence-adaptive CGRH requires residual scales"
                        )
                    set_scales(residual_scales)
                    if model_ema is not None:
                        ema_set_scales = getattr(
                            model_ema.module,
                            "set_soft_cgrh_residual_scales",
                            None,
                        )
                        if callable(ema_set_scales):
                            ema_set_scales(residual_scales)

                logits, cgrh_state = model(
                    images,
                    return_soft_cgrh=True,
                )
                with torch.no_grad():
                    teacher_logits = (
                        model_ema.module(images)
                        if model_ema is not None
                        else cgrh_state["main_logits"].detach()
                    )
                criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, cgrh_state)
            elif bool(getattr(criterion, "requires_auto_cgrh", False)):
                pairs_frozen = bool(
                    getattr(criterion, "_pairs_frozen", False)
                )
                never_freeze = bool(
                    getattr(criterion, "never_freeze", False)
                )
                active_pairs = (
                    list(getattr(criterion, "_active_pairs", []))
                    if (pairs_frozen or never_freeze)
                    else []
                )
                set_pairs = getattr(model, "set_auto_cgrh_pairs", None)
                if not callable(set_pairs):
                    raise RuntimeError(
                        "Auto-CGRH criterion requires a compatible model"
                    )
                set_pairs(active_pairs)
                update_counts = getattr(model, "update_hard_pair_counts", None)
                if callable(update_counts):
                    update_counts(
                        active_count=len(active_pairs),
                        core_count=len(
                            list(getattr(criterion, "_core_pairs", []))
                        ),
                        eligible_count=int(
                            getattr(
                                criterion,
                                "_eligible_hard_pair_count",
                                len(active_pairs),
                            )
                        ),
                    )
                if model_ema is not None:
                    ema_set_pairs = getattr(
                        model_ema.module,
                        "set_auto_cgrh_pairs",
                        None,
                    )
                    if callable(ema_set_pairs):
                        ema_set_pairs(active_pairs)
                    ema_update = getattr(
                        model_ema.module,
                        "update_hard_pair_counts",
                        None,
                    )
                    if callable(ema_update):
                        ema_update(
                            active_count=len(active_pairs),
                            core_count=len(
                                list(getattr(criterion, "_core_pairs", []))
                            ),
                            eligible_count=int(
                                getattr(
                                    criterion,
                                    "_eligible_hard_pair_count",
                                    len(active_pairs),
                                )
                            ),
                        )

                logits, cgrh_state = model(
                    images,
                    return_cgrh=True,
                )
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        teacher_logits = (
                            model_ema.module(
                                images,
                                return_main_logits=True,
                            )
                            if model_ema is not None
                            else cgrh_state["main_logits"].detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, cgrh_state)
            elif bool(
                getattr(
                    criterion,
                    "requires_odpm_cgrh_coverage_v3",
                    False,
                )
            ):
                active_pairs = list(getattr(criterion, "_active_pairs", []))
                logits, odpm_state = model(
                    images,
                    return_odpm_cgrh_coverage_v3=True,
                    odpm_active_pairs=active_pairs,
                )
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        teacher_logits = (
                            model_ema.module(images)
                            if model_ema is not None
                            else logits.detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, odpm_state)
            elif bool(
                getattr(criterion, "requires_odpm_cgrh_hard_v2", False)
            ):
                active_pairs = list(getattr(criterion, "_active_pairs", []))
                logits, odpm_state = model(
                    images,
                    return_odpm_cgrh_hard_v2=True,
                    odpm_active_pairs=active_pairs,
                )
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        # Pair discovery always observes the deployable main
                        # head (EMA when enabled), never expert-refined logits.
                        teacher_logits = (
                            model_ema.module(images)
                            if model_ema is not None
                            else odpm_state["main_logits"].detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, odpm_state)
            elif bool(getattr(criterion, "requires_odpm_cgrh", False)):
                active_pairs = list(getattr(criterion, "_active_pairs", []))
                logits, odpm_state = model(
                    images,
                    return_odpm_cgrh=True,
                    odpm_active_pairs=active_pairs,
                )
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        # Pair discovery observes deployable main-head logits,
                        # preferably from EMA, never expert-refined logits.
                        teacher_logits = (
                            model_ema.module(images)
                            if model_ema is not None
                            else odpm_state["main_logits"].detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, odpm_state)
            elif bool(getattr(criterion, "requires_dcmg_v2", False)):
                logits = model(images)
                with torch.no_grad():
                    teacher_logits = (
                        model_ema.module(images)
                        if model_ema is not None
                        else logits.detach()
                    )
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, teacher_logits)
            elif bool(getattr(criterion, "requires_dcmg", False)):
                logits = model(images)
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        teacher_logits = (
                            model_ema.module(images)
                            if model_ema is not None
                            else logits.detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets)
            elif bool(getattr(criterion, "requires_aocg", False)):
                logits, aocg_state = model(images, return_aocg_aux=True)
                should_observe = getattr(
                    criterion, "should_observe_teacher", None
                )
                if not callable(should_observe) or bool(should_observe()):
                    with torch.no_grad():
                        teacher_logits = (
                            model_ema.module(images)
                            if model_ema is not None
                            else logits.detach()
                        )
                    criterion.observe_teacher(teacher_logits, targets)
                loss = criterion(logits, targets, aocg_state)
            elif bool(getattr(criterion, "requires_dphg", False)):
                logits, dphg_state = model(images, return_dphg=True)
                loss = criterion(logits, targets, dphg_state)
            elif bool(getattr(criterion, "requires_cgrh_aux", False)):
                logits, cgrh_aux_state = model(
                    images, return_cgrh_aux=True
                )
                loss = criterion(logits, targets, cgrh_aux_state)
            elif bool(getattr(criterion, "requires_apd_aux", False)):
                logits, apd_state = model(images, return_apd_aux=True)
                loss = criterion(logits, targets, apd_state)
            elif bool(getattr(criterion, "requires_cgrh", False)):
                logits, cgrh_state = model(images, return_cgrh=True)
                loss = criterion(logits, targets, cgrh_state)
            elif bool(getattr(criterion, "requires_lgrc", False)):
                auxiliary_active = getattr(
                    criterion, "auxiliary_active", lambda: True
                )
                if bool(auxiliary_active()):
                    logits, lgrc_state = model(images, return_lgrc=True)
                else:
                    logits = model(images)
                    lgrc_state = None
                loss = criterion(logits, targets, lgrc_state)
            elif bool(getattr(criterion, "requires_rcaux", False)):
                auxiliary_active = getattr(
                    criterion, "auxiliary_active", lambda: True
                )
                if bool(auxiliary_active()):
                    logits, rcaux_state = model(images, return_rcaux=True)
                else:
                    logits = model(images)
                    rcaux_state = None
                loss = criterion(logits, targets, rcaux_state)
            elif bool(getattr(criterion, "requires_features", False)):
                logits, features = model(images, return_features=True)
                loss = criterion(logits, targets, features)
            elif bool(getattr(criterion, "requires_aux", False)):
                logits, aux_logits = model(images, return_aux=True)
                loss = criterion(logits, targets, aux_logits)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if perturbation is None:
            scaler.step(optimizer)
            scaler.update()
        else:
            if not unscale_first_pass_gradients(scaler, optimizer):
                continue
            grad_norm = perturbation.first_step()
            if grad_norm is None:
                update_scaler_after_skipped_first_pass(scaler, optimizer)
                continue
            try:
                with disable_batchnorm_running_stats(model):
                    with _autocast_context(device, amp):
                        second_logits = model(images)
                        second_loss = criterion(second_logits, targets)
                    scaler.scale(second_loss).backward()
            finally:
                # The base optimizer must always see the original parameters,
                # including when the second forward/backward raises.
                perturbation.restore()
            scaler.step(optimizer)
            scaler.update()
            sharpness_grad_norm_sum += grad_norm
            sharpness_second_loss_sum += float(second_loss.detach().item())
            sharpness_batches += 1
        if model_ema is not None:
            ema_update_step += 1
            model_ema.update(model, step=ema_update_step)

        predictions = logits.argmax(dim=1)
        meter.update(loss.item(), targets, predictions)
        components = getattr(criterion, "last_components", None)
        if isinstance(components, dict):
            batch_size = int(targets.shape[0])
            component_samples += batch_size
            for name, value in components.items():
                component_sums[name] = (
                    component_sums.get(name, 0.0) + float(value) * batch_size
                )

    metrics = meter.compute()
    if component_samples:
        for name, total in component_sums.items():
            metrics[name] = total / component_samples
    if sharpness_batches:
        metrics["sharpness_grad_norm"] = (
            sharpness_grad_norm_sum / sharpness_batches
        )
        metrics["sharpness_second_loss"] = (
            sharpness_second_loss_sum / sharpness_batches
        )
    metrics["ema_update_step"] = ema_update_step
    return metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    amp: bool,
    max_batches: int | None = None,
    collect_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    meter = ClassificationMeter(num_classes)
    supports_cgrh = hasattr(model, "cgrh")
    main_meter = ClassificationMeter(num_classes) if supports_cgrh else None
    cgrh_routed = 0
    cgrh_changed = 0
    cgrh_corrected = 0
    cgrh_harmed = 0
    cgrh_pair_route_counts: torch.Tensor | None = None
    rows: list[dict[str, Any]] = []

    for batch_index, (images, targets, paths) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(device, amp):
            cgrh_state: dict[str, torch.Tensor] | None = None
            if supports_cgrh:
                logits, cgrh_state = model(images, return_cgrh=True)
            else:
                logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        meter.update(loss.item(), targets, predictions)

        main_predictions: torch.Tensor | None = None
        main_confidence: torch.Tensor | None = None
        routed: torch.Tensor | None = None
        if cgrh_state is not None:
            main_logits = cgrh_state["main_logits"]
            main_probabilities = torch.softmax(main_logits, dim=1)
            main_confidence, main_predictions = main_probabilities.max(dim=1)
            assert main_meter is not None
            main_loss = torch.nn.functional.cross_entropy(
                main_logits, targets
            )
            main_meter.update(main_loss.item(), targets, main_predictions)
            route_mask = cgrh_state["route_mask"]
            routed = route_mask.any(dim=1)
            route_counts = route_mask.sum(dim=0).to(torch.int64).cpu()
            if cgrh_pair_route_counts is None:
                cgrh_pair_route_counts = torch.zeros_like(route_counts)
            cgrh_pair_route_counts += route_counts

            changed = predictions != main_predictions
            cgrh_routed += int(routed.sum().item())
            cgrh_changed += int(changed.sum().item())
            cgrh_corrected += int(
                ((main_predictions != targets) & (predictions == targets))
                .sum()
                .item()
            )
            cgrh_harmed += int(
                ((main_predictions == targets) & (predictions != targets))
                .sum()
                .item()
            )

        if collect_predictions:
            target_values = targets.cpu().tolist()
            prediction_values = predictions.cpu().tolist()
            confidence_values = confidence.cpu().tolist()
            main_prediction_values = (
                main_predictions.cpu().tolist()
                if main_predictions is not None
                else None
            )
            main_confidence_values = (
                main_confidence.cpu().tolist()
                if main_confidence is not None
                else None
            )
            routed_values = routed.cpu().tolist() if routed is not None else None
            for index, path in enumerate(paths):
                row = {
                    "path": path,
                    "target": int(target_values[index]),
                    "prediction": int(prediction_values[index]),
                    "confidence": float(confidence_values[index]),
                }
                if main_prediction_values is not None:
                    row.update(
                        {
                            "main_prediction": int(
                                main_prediction_values[index]
                            ),
                            "main_confidence": float(
                                main_confidence_values[index]
                            ),
                            "cgrh_routed": int(bool(routed_values[index])),
                            "cgrh_changed": int(
                                main_prediction_values[index]
                                != prediction_values[index]
                            ),
                        }
                    )
                rows.append(row)

    metrics = meter.compute()
    if main_meter is not None:
        main_metrics = main_meter.compute()
        metrics["cgrh_main_accuracy"] = main_metrics["accuracy"]
        metrics["cgrh_main_macro_f1"] = main_metrics["macro_f1"]
        metrics["cgrh_routed_samples"] = cgrh_routed
        metrics["cgrh_route_rate"] = cgrh_routed / max(
            main_meter.sample_count, 1
        )
        metrics["cgrh_changed_predictions"] = cgrh_changed
        metrics["cgrh_corrected_errors"] = cgrh_corrected
        metrics["cgrh_harmed_correct"] = cgrh_harmed
        metrics["cgrh_pair_route_counts"] = (
            cgrh_pair_route_counts.tolist()
            if cgrh_pair_route_counts is not None
            else []
        )
    return metrics, rows
