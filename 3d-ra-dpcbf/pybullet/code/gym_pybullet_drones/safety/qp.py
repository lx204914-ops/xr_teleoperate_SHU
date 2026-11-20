"""Quadratic-program safety filters for drone outer-loop acceleration."""

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import cvxpy as cp
import numpy as np

from .barriers import (
    BarrierResult,
    base_hocbf_affine_terms,
    ra_barrier_affine_terms,
)
from .config import BarrierConfig, RiskConfig


@dataclass(frozen=True)
class HOCBFSolution:
    control: np.ndarray
    status: str
    margins: tuple[float, ...]
    solve_time: float


@dataclass(frozen=True)
class RADPCBFSolution:
    control: np.ndarray
    status: str
    slacks: tuple[float, ...]
    barriers: tuple[BarrierResult, ...]
    hard_margins: tuple[float, ...]
    ra_margins: tuple[float, ...]
    solve_time: float


class HOCBFSafetyFilter:
    def __init__(
        self,
        robot_radius: float,
        max_acceleration: float,
        margin: float = 0.15,
        gain_1: float = 2.0,
        gain_2: float = 2.0,
    ) -> None:
        if min(robot_radius, max_acceleration, gain_1, gain_2) <= 0.0 or margin < 0.0:
            raise ValueError("HOCBF filter parameters are invalid.")
        self.robot_radius = float(robot_radius)
        self.max_acceleration = float(max_acceleration)
        self.margin = float(margin)
        self.gain_1 = float(gain_1)
        self.gain_2 = float(gain_2)

    def solve(
        self,
        state: np.ndarray,
        nominal_control: np.ndarray,
        obstacles: Sequence[np.ndarray],
    ) -> HOCBFSolution:
        nominal = np.asarray(nominal_control, dtype=float).reshape(-1)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_control must contain three finite values.")
        nominal = np.clip(nominal, -self.max_acceleration, self.max_acceleration)
        if not obstacles:
            return HOCBFSolution(nominal, "optimal", (), 0.0)

        control = cp.Variable(3)
        constraints = [
            control <= self.max_acceleration,
            control >= -self.max_acceleration,
        ]
        terms: list[tuple[float, np.ndarray]] = []
        for obstacle in obstacles:
            offset, coefficient, _ = base_hocbf_affine_terms(
                state,
                obstacle,
                self.robot_radius,
                self.margin,
                self.gain_1,
                self.gain_2,
            )
            terms.append((offset, coefficient))
            constraints.append(offset + coefficient @ control >= 0.0)
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(control - nominal)),
            constraints,
        )
        started = perf_counter()
        status = "solver_error"
        for solver in ("CLARABEL", "OSQP", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            status = str(problem.status)
            if status in {"optimal", "optimal_inaccurate"}:
                break
        elapsed = perf_counter() - started
        if status not in {"optimal", "optimal_inaccurate"} or control.value is None:
            return HOCBFSolution(nominal, status, (), elapsed)
        result = np.asarray(control.value, dtype=float).reshape(3)
        margins = tuple(float(offset + coefficient @ result) for offset, coefficient in terms)
        return HOCBFSolution(result, status, margins, elapsed)


class RADPCBFSafetyFilter:
    """Hard distance HOCBF plus soft analytic 3D RA-DPCBF."""

    def __init__(
        self,
        robot_radius: float,
        max_acceleration: float,
        base_margin: float = 0.15,
        base_gain_1: float = 2.0,
        base_gain_2: float = 2.0,
        ra_alpha: float = 1.5,
        slack_penalty: float = 1e4,
        risk_config: RiskConfig = RiskConfig(),
        barrier_config: BarrierConfig = BarrierConfig(),
    ) -> None:
        if min(
            robot_radius,
            max_acceleration,
            base_gain_1,
            base_gain_2,
            ra_alpha,
            slack_penalty,
        ) <= 0.0 or base_margin < 0.0:
            raise ValueError("RA-DPCBF filter parameters are invalid.")
        self.robot_radius = float(robot_radius)
        self.max_acceleration = float(max_acceleration)
        self.base_margin = float(base_margin)
        self.base_gain_1 = float(base_gain_1)
        self.base_gain_2 = float(base_gain_2)
        self.ra_alpha = float(ra_alpha)
        self.slack_penalty = float(slack_penalty)
        self.risk_config = risk_config
        self.barrier_config = barrier_config

    def solve(
        self,
        state: np.ndarray,
        nominal_control: np.ndarray,
        obstacles: Sequence[np.ndarray],
    ) -> RADPCBFSolution:
        nominal = np.asarray(nominal_control, dtype=float).reshape(-1)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_control must contain three finite values.")
        nominal = np.clip(nominal, -self.max_acceleration, self.max_acceleration)
        if not obstacles:
            return RADPCBFSolution(nominal, "optimal", (), (), (), (), 0.0)

        control = cp.Variable(3)
        slack = cp.Variable(len(obstacles), nonneg=True)
        constraints = [
            control <= self.max_acceleration,
            control >= -self.max_acceleration,
        ]
        hard_terms: list[tuple[float, np.ndarray]] = []
        ra_terms: list[tuple[float, np.ndarray, BarrierResult]] = []
        for index, obstacle in enumerate(obstacles):
            hard_offset, hard_coefficient, _ = base_hocbf_affine_terms(
                state,
                obstacle,
                self.robot_radius,
                self.base_margin,
                self.base_gain_1,
                self.base_gain_2,
            )
            ra_offset, ra_coefficient, barrier = ra_barrier_affine_terms(
                state,
                obstacle,
                self.robot_radius,
                self.risk_config,
                self.barrier_config,
            )
            hard_terms.append((hard_offset, hard_coefficient))
            ra_terms.append((ra_offset, ra_coefficient, barrier))
            constraints.append(hard_offset + hard_coefficient @ control >= 0.0)
            constraints.append(
                ra_offset
                + ra_coefficient @ control
                + self.ra_alpha * barrier.h
                + slack[index]
                >= 0.0
            )
        objective = cp.Minimize(
            0.5 * cp.sum_squares(control - nominal)
            + self.slack_penalty * cp.sum_squares(slack)
        )
        problem = cp.Problem(objective, constraints)
        started = perf_counter()
        status = "solver_error"
        for solver in ("CLARABEL", "OSQP", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError:
                continue
            status = str(problem.status)
            if status in {"optimal", "optimal_inaccurate"}:
                break
        elapsed = perf_counter() - started
        barriers = tuple(item[2] for item in ra_terms)
        if (
            status not in {"optimal", "optimal_inaccurate"}
            or control.value is None
            or slack.value is None
        ):
            return RADPCBFSolution(nominal, status, (), barriers, (), (), elapsed)
        result = np.asarray(control.value, dtype=float).reshape(3)
        slack_values = tuple(
            float(max(0.0, item)) for item in np.asarray(slack.value).reshape(-1)
        )
        hard_margins = tuple(
            float(offset + coefficient @ result)
            for offset, coefficient in hard_terms
        )
        ra_margins = tuple(
            float(offset + coefficient @ result + self.ra_alpha * barrier.h)
            for offset, coefficient, barrier in ra_terms
        )
        return RADPCBFSolution(
            result,
            status,
            slack_values,
            barriers,
            hard_margins,
            ra_margins,
            elapsed,
        )
