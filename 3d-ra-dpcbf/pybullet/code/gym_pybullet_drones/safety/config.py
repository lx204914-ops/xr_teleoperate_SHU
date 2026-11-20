"""Configuration for smooth risk-aware parabolic barriers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    weight_distance: float = 0.45
    weight_closing: float = 0.40
    weight_ttc: float = 0.15
    distance_scale: float = 2.0
    closing_scale: float = 1.5
    ttc_scale: float = 3.0
    clearance_beta: float = 10.0
    closing_beta: float = 10.0
    ttc_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        total = self.weight_distance + self.weight_closing + self.weight_ttc
        if abs(total - 1.0) > 1e-12:
            raise ValueError("Risk weights must sum to one.")
        if min(
            self.weight_distance,
            self.weight_closing,
            self.weight_ttc,
        ) < 0.0:
            raise ValueError("Risk weights cannot be negative.")
        if min(
            self.distance_scale,
            self.closing_scale,
            self.ttc_scale,
            self.clearance_beta,
            self.closing_beta,
            self.ttc_epsilon,
        ) <= 0.0:
            raise ValueError("Risk scales and smoothing parameters must be positive.")


@dataclass(frozen=True)
class BarrierConfig:
    k_lambda: float = 0.1
    k_mu: float = 0.5
    gamma_radius: float = 0.5
    safety_scale: float = 1.05
    gap_beta: float = 10.0
    gap_epsilon: float = 1e-3
    speed_epsilon: float = 1e-3
    distance_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.gamma_radius < 0.0:
            raise ValueError("gamma_radius cannot be negative.")
        if min(
            self.k_lambda,
            self.k_mu,
            self.safety_scale,
            self.gap_beta,
            self.gap_epsilon,
            self.speed_epsilon,
            self.distance_epsilon,
        ) <= 0.0:
            raise ValueError("Barrier parameters must be positive.")
        if self.safety_scale <= 1.0:
            raise ValueError("safety_scale must be greater than one.")

