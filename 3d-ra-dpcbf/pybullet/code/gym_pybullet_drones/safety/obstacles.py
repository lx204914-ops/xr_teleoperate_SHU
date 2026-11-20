"""Deterministic spherical obstacles shared by safety and PyBullet scenes."""

from dataclasses import dataclass

import numpy as np
import pybullet as p


def _vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    return vector.copy()


@dataclass(frozen=True)
class ObstacleState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    radius: float


@dataclass
class SphereObstacle:
    obstacle_id: int
    initial_position: np.ndarray
    velocity: np.ndarray
    radius: float
    color: tuple[float, float, float, float] = (0.9, 0.12, 0.08, 0.9)
    body_id: int | None = None

    def __post_init__(self) -> None:
        self.initial_position = _vector3(self.initial_position, "initial_position")
        self.velocity = _vector3(self.velocity, "velocity")
        if self.radius <= 0.0:
            raise ValueError("Obstacle radius must be positive.")

    def state_at(self, time: float) -> ObstacleState:
        if time < 0.0:
            raise ValueError("time cannot be negative.")
        return ObstacleState(
            position=self.initial_position + time * self.velocity,
            velocity=self.velocity.copy(),
            acceleration=np.zeros(3, dtype=float),
            radius=float(self.radius),
        )

    def create_body(self, client_id: int) -> int:
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=self.color,
            physicsClientId=client_id,
        )
        collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.radius,
            physicsClientId=client_id,
        )
        self.body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.initial_position,
            physicsClientId=client_id,
        )
        return self.body_id

    def update_body(self, time: float, client_id: int) -> ObstacleState:
        state = self.state_at(time)
        if self.body_id is None:
            raise RuntimeError("Obstacle body has not been created.")
        p.resetBasePositionAndOrientation(
            self.body_id,
            state.position,
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client_id,
        )
        return state


def swept_sphere_collision(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_position: np.ndarray,
    combined_radius: float,
) -> bool:
    if combined_radius <= 0.0:
        raise ValueError("combined_radius must be positive.")
    start = _vector3(start, "start")
    end = _vector3(end, "end")
    center = _vector3(obstacle_position, "obstacle_position")
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 1e-15:
        closest = start
    else:
        fraction = float(np.clip((center - start) @ segment / denominator, 0.0, 1.0))
        closest = start + fraction * segment
    return float(np.linalg.norm(closest - center)) <= combined_radius

