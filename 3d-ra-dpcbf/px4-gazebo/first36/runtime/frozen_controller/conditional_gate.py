"""Stateful exceptional-condition gate for the copied RA-DPCBF controller.

The primary RA-DPCBF barrier and sampling-tightened safety QP are unchanged.
This module only decides when the existing chi--Gamma auxiliary detour may be
started, and when that auxiliary row should yield back to the base safe QP.
"""
from collections import deque
import numpy as np

from detour import blocked
from ra_dpcbf.certificates import current_common_set


class ConditionalDetourGate:
    """Enable the original detour only for persistent exceptional conditions."""

    def __init__(self, config):
        self.cfg = dict(config)
        self.elapsed = {"feasibility": 0.0, "radial_braking": 0.0, "stagnation": 0.0}
        self.previous = {key: False for key in self.elapsed}
        self.progress_release_elapsed = 0.0
        self.progress_release_previous = False
        self.rearm_elapsed = 0.0
        self.rearm_previous = False
        self.safe_progress = False
        self.distance_history = deque()
        self.activation_counts = {key: 0 for key in self.elapsed}
        self.release_count = 0
        self.rearm_count = 0
        self.last_common_check_time = -np.inf
        self.cached_common_margin = np.inf
        self.cached_common_status = "not_checked"

    @staticmethod
    def _goal_kinematics(x, goal, control):
        line = np.asarray(goal, dtype=float) - np.asarray(x[:3], dtype=float)
        distance = float(np.linalg.norm(line))
        if distance <= 1e-12:
            return distance, 0.0, 0.0
        direction = line / distance
        velocity = np.asarray(x[3:], dtype=float)
        toward_speed = float(direction @ velocity)
        turn_term = float((velocity @ velocity - toward_speed * toward_speed) / distance)
        toward_speed_rate = float(direction @ np.asarray(control, dtype=float) - turn_term)
        return distance, toward_speed, toward_speed_rate

    def _dwell(self, name, predicate, dt):
        if predicate:
            self.elapsed[name] = self.elapsed[name] + dt if self.previous[name] else 0.0
            self.previous[name] = True
        else:
            self.elapsed[name] = 0.0
            self.previous[name] = False
        return self.elapsed[name]

    @staticmethod
    def _candidate(x, goal, obs, log, s_on):
        mask = blocked(np.asarray(x[:3]), np.asarray(goal), np.asarray(obs), np.asarray(log[:, 3]))
        ids = np.flatnonzero(mask & (log[:, 6] >= 0.0) & (log[:, 6] <= s_on))
        if not len(ids):
            return mask, -1
        return mask, int(ids[np.lexsort((ids, log[ids, 6]))[0]])

    def pre_update(self, x, goal, obs, log, B, Cexec, target, base, dt, mechanism_active):
        """Return a decision before the original detour direction search.

        The feasibility trigger periodically evaluates the sampling-tightened
        common-set max-min margin.  Per-row box reserve is logged only as a
        diagnostic and is never substituted for the common-set decision.
        """
        cfg = self.cfg
        # Release dwell cannot span inactive observations.
        if not mechanism_active:
            self.progress_release_elapsed = 0.0
            self.progress_release_previous = False
        distance, toward_speed, base_rate = self._goal_kinematics(x, goal, base)
        now = float(self.distance_history[-1][0] + dt) if self.distance_history else 0.0
        self.distance_history.append((now, distance))
        window = float(cfg["stagnation_window_s"])
        while self.distance_history and now - self.distance_history[0][0] > window + dt:
            self.distance_history.popleft()

        mask, j = self._candidate(x, goal, obs, log, cfg["s_on"])
        blocking = j >= 0
        B = np.asarray(B, dtype=float); Cexec = np.asarray(Cexec, dtype=float)
        limits = np.asarray(cfg["acceleration_component_limits"], dtype=float)
        row_norm = np.maximum(np.linalg.norm(B, axis=1), 1e-12)
        row_reserve = (Cexec + np.abs(B) @ limits) / row_norm
        minimum_row_reserve = float(np.min(row_reserve)) if len(row_reserve) else np.inf
        if blocking and now-self.last_common_check_time >= cfg["feasibility_check_period_s"]-dt/2:
            report=current_common_set(B,Cexec,limits)
            self.cached_common_status=str(report["status"])
            self.cached_common_margin=(float(report["margin"])
                                       if report.get("margin") is not None else np.inf)
            self.last_common_check_time=now
        minimum_reserve=self.cached_common_margin if blocking else np.inf

        radial_brake = 0.0; transverse_correction = 0.0; q = np.nan
        if blocking:
            r = np.asarray(obs[j, :3]) - np.asarray(x[:3]); D = max(float(np.linalg.norm(r)), 1e-12)
            e = r / D; correction = np.asarray(base) - np.asarray(target)
            radial_brake = max(0.0, float(-e @ correction))
            transverse_correction = float(np.linalg.norm(correction - e * (e @ correction)))
            q = float(log[j, 1])

        enough_window = bool(self.distance_history and now - self.distance_history[0][0] >= window - dt)
        distance_drop = (float(self.distance_history[0][1] - distance) if enough_window else np.inf)
        predicates = {
            "feasibility": bool(blocking and minimum_reserve <= cfg["feasibility_reserve_on_mps2"]),
            "radial_braking": bool(blocking and q < 0.0 and radial_brake >= cfg["radial_brake_on_mps2"] and
                                    radial_brake >= cfg["radial_to_transverse_ratio"] * transverse_correction),
            "stagnation": bool(blocking and distance > cfg["stagnation_ignore_goal_radius_m"] and enough_window and
                               distance_drop <= cfg["stagnation_max_progress_m"]),
        }

        # SAFE_PROGRESS is a hysteresis state: the base safe QP keeps authority
        # while it is moving toward the goal or correcting retreat.
        rearmed = False
        if self.safe_progress:
            stalled = bool(toward_speed <= 0.0 and base_rate <= 0.0)
            if stalled:
                self.rearm_elapsed = self.rearm_elapsed + dt if self.rearm_previous else 0.0
                self.rearm_previous = True
            else:
                self.rearm_elapsed = 0.0; self.rearm_previous = False
            if self.rearm_elapsed >= cfg["rearm_dwell_s"] - 1e-12:
                self.safe_progress = False; self.rearm_elapsed = 0.0; self.rearm_previous = False
                self.rearm_count += 1; rearmed = True
            for key in self.elapsed:
                self._dwell(key, False, dt)
            allow = False
            reason = "safe_progress_rearmed_next_step" if rearmed else "safe_progress_base_only"
        elif mechanism_active:
            for key in self.elapsed:
                self._dwell(key, False, dt)
            allow = True; reason = "detour_already_active"
        else:
            reached = []
            for key, predicate in predicates.items():
                value = self._dwell(key, predicate, dt)
                if value >= cfg[f"{key}_dwell_s"] - 1e-12:
                    reached.append(key)
            priority = ("feasibility", "stagnation", "radial_braking")
            chosen = next((key for key in priority if key in reached), None)
            allow = chosen is not None
            reason = f"conditional_{chosen}" if chosen else "base_safe_qp_default"
            if allow:
                self.activation_counts[chosen] += 1
                for key in self.elapsed:
                    self._dwell(key, False, dt)

        return {
            "allow_activation": bool(allow), "gate_reason": reason, "candidate": int(j),
            "blocking_count": int(np.count_nonzero(mask)), "common_set_status": self.cached_common_status,
            "minimum_common_set_margin_mps2": minimum_reserve,
            "minimum_individual_row_reserve_mps2": minimum_row_reserve,
            "radial_brake_mps2": radial_brake, "transverse_correction_mps2": transverse_correction,
            "goal_distance_m": distance, "toward_goal_speed_mps": toward_speed,
            "base_toward_speed_rate_mps2": base_rate, "window_progress_m": distance_drop,
            "feasibility_elapsed_s": self.elapsed["feasibility"],
            "radial_braking_elapsed_s": self.elapsed["radial_braking"],
            "stagnation_elapsed_s": self.elapsed["stagnation"],
            "safe_progress": bool(self.safe_progress), "rearmed": bool(rearmed),
        }

    def post_augmented(self, x, goal, base, augmented, detour, dt):
        """Check the audited early-release predicate after both QPs are known."""
        _, toward_speed, base_rate = self._goal_kinematics(x, goal, base)
        _, _, augmented_rate = self._goal_kinematics(x, goal, augmented)
        distance, _, _ = self._goal_kinematics(x, goal, base)
        base_eval=detour.get("goal_base_eval",{})
        aug_eval=detour.get("goal_augmented_eval",{})
        goal_near=distance<=self.cfg["stagnation_ignore_goal_radius_m"]
        progress_better=(base_eval.get("cost",float("inf"))<aug_eval.get("cost",-float("inf"))-1e-8 and
                         base_eval.get("goal_distance_T",float("inf"))<aug_eval.get("goal_distance_T",-float("inf"))-1e-8)
        predicate=bool(detour.get("active") and (goal_near or
                       (detour.get("primary_clear") and progress_better and (toward_speed>0 or base_rate>0))))
        if predicate:
            self.progress_release_elapsed = (self.progress_release_elapsed + dt
                                             if self.progress_release_previous else 0.0)
            self.progress_release_previous = True
        else:
            self.progress_release_elapsed = 0.0; self.progress_release_previous = False
        release = self.progress_release_elapsed >= self.cfg["release_dwell_s"] - 1e-12
        if release:
            self.safe_progress = True; self.release_count += 1
            self.progress_release_elapsed = 0.0; self.progress_release_previous = False
        return {
            "release": bool(release), "release_predicate": predicate,
            "release_elapsed_s": self.progress_release_elapsed,
            "toward_goal_speed_mps": toward_speed,
            "base_toward_speed_rate_mps2": base_rate,
            "augmented_toward_speed_rate_mps2": augmented_rate,
        }

    def summary(self):
        return {"activation_counts": dict(self.activation_counts), "release_count": self.release_count,
                "rearm_count": self.rearm_count, "safe_progress_at_end": bool(self.safe_progress)}
