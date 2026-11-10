from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn


@dataclass(frozen=True)
class SharpnessAwareConfig:
    enabled: bool = False
    variant: str = "sam"
    rho: float = 0.02
    eta: float = 0.01

    @property
    def adaptive(self) -> bool:
        return self.variant == "asam"


def parse_sharpness_config(config: dict[str, Any] | None) -> SharpnessAwareConfig:
    values = dict(config or {})
    enabled = bool(values.get("enabled", False))
    variant = str(values.get("variant", "sam")).strip().lower()
    if variant not in {"sam", "asam"}:
        raise ValueError("sharpness_aware.variant must be 'sam' or 'asam'")
    rho = float(values.get("rho", 0.02 if variant == "sam" else 0.2))
    eta = float(values.get("eta", 0.01))
    if rho <= 0.0:
        raise ValueError("sharpness_aware.rho must be positive")
    if eta < 0.0:
        raise ValueError("sharpness_aware.eta must be non-negative")
    return SharpnessAwareConfig(
        enabled=enabled,
        variant=variant,
        rho=rho,
        eta=eta,
    )


class SharpnessAwarePerturbation:
    """Apply and restore one SAM/ASAM parameter perturbation.

    The wrapped optimizer remains the ordinary base optimizer. This keeps the
    existing scheduler and checkpoint format unchanged and ensures that EMA is
    updated only after the real optimizer step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: SharpnessAwareConfig,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self._perturbations: dict[torch.nn.Parameter, torch.Tensor] = {}

    def _parameters_with_grad(self) -> Iterator[torch.nn.Parameter]:
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    yield parameter

    @torch.no_grad()
    def first_step(self) -> float | None:
        parameters = list(self._parameters_with_grad())
        if not parameters:
            return None

        norm_terms: list[torch.Tensor] = []
        for parameter in parameters:
            gradient = parameter.grad.detach().float()
            if self.config.adaptive:
                gradient = gradient * (
                    parameter.detach().float().abs() + self.config.eta
                )
            norm_terms.append(torch.linalg.vector_norm(gradient, ord=2))
        grad_norm = torch.linalg.vector_norm(torch.stack(norm_terms), ord=2)
        if not bool(torch.isfinite(grad_norm)) or float(grad_norm.item()) == 0.0:
            return None

        scale = self.config.rho / (grad_norm + 1e-12)
        self._perturbations.clear()
        for parameter in parameters:
            gradient = parameter.grad.detach().float()
            if self.config.adaptive:
                adaptive_scale = parameter.detach().float().abs() + self.config.eta
                perturbation = gradient * adaptive_scale.square() * scale
            else:
                perturbation = gradient * scale
            perturbation = perturbation.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.add_(perturbation)
            self._perturbations[parameter] = perturbation
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())

    @torch.no_grad()
    def restore(self) -> None:
        for parameter, perturbation in self._perturbations.items():
            parameter.sub_(perturbation)
        self._perturbations.clear()


def unscale_first_pass_gradients(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
) -> bool:
    """Unscale first-pass gradients without consuming GradScaler step state.

    The second SAM backward is the gradient that the base optimizer consumes.
    Keeping GradScaler in READY state lets ``scaler.step(optimizer)`` perform
    its normal finite check and unscale for that second backward.
    """

    gradients = [
        parameter.grad
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.grad is not None
    ]
    if not gradients:
        return False
    inverse_scale = 1.0 / float(scaler.get_scale())
    # A foreach operation avoids one host synchronization per parameter. The
    # single combined norm check in ``first_step`` handles non-finite values.
    torch._foreach_mul_(gradients, inverse_scale)
    return True


def update_scaler_after_skipped_first_pass(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Record a non-finite/zero first pass and reset GradScaler batch state."""

    # Gradients were already manually unscaled, but a second unscale is harmless
    # for the only two skip cases (all-zero or non-finite) and lets GradScaler
    # record overflow before ``update``.
    scaler.unscale_(optimizer)
    optimizer.zero_grad(set_to_none=True)
    scaler.update()


@contextmanager
def disable_batchnorm_running_stats(model: nn.Module):
    """Use batch statistics without updating running buffers on SAM pass two."""

    states: list[tuple[nn.modules.batchnorm._BatchNorm, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            states.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in states:
            module.track_running_stats = track_running_stats
