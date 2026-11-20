"""Minimal physical scenes used before adding safety filters."""

from dataclasses import dataclass

import numpy as np

from gym_pybullet_drones.safety.obstacles import SphereObstacle


@dataclass(frozen=True)
class FlightScene:
    name: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: tuple[SphereObstacle, ...]
    duration: float


def build_scene(name: str, seed: int = 7) -> FlightScene:
    del seed
    start = np.array([0.0, 0.0, 1.0], dtype=float)
    goal = np.array([3.0, 0.0, 1.0], dtype=float)
    if name == "goal_only":
        return FlightScene(name, start, goal, (), 8.0)
    if name in {"static_scene", "static_hocbf", "static_ra_dpcbf"}:
        obstacle = SphereObstacle(
            obstacle_id=0,
            initial_position=np.array([1.5, 0.15, 1.0]),
            velocity=np.zeros(3),
            radius=0.35,
        )
        return FlightScene(name, start, goal, (obstacle,), 8.0)
    dynamic_specs = {
        "dynamic_crossing": (
            np.array([1.55, -0.65, 1.0]),
            np.array([0.0, 0.22, 0.0]),
        ),
        "dynamic_vertical": (
            np.array([1.55, 0.10, 1.65]),
            np.array([0.0, 0.0, -0.20]),
        ),
        "dynamic_head_on": (
            np.array([2.40, 0.15, 1.0]),
            np.array([-0.20, 0.0, 0.0]),
        ),
        "dynamic_diagonal_3d": (
            np.array([1.70, -0.65, 1.55]),
            np.array([-0.05, 0.20, -0.14]),
        ),
    }
    if name in dynamic_specs:
        position, velocity = dynamic_specs[name]
        obstacle = SphereObstacle(
            obstacle_id=0,
            initial_position=position,
            velocity=velocity,
            radius=0.35,
        )
        return FlightScene(name, start, goal, (obstacle,), 10.0)
    raise ValueError(f"Unknown scene: {name}")
