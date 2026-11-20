import numpy as np

from conditional_gate import ConditionalDetourGate
from configuration import load_config
from safety_filter import SafetyFilter
from test_integration import state


def gate_fixture():
    cfg = load_config()["conditional_detour"]
    gate = ConditionalDetourGate(cfg)
    x = np.zeros(6); goal = np.array([10., 0., 0.])
    obs = np.array([[2., 0., 0., 0., 0., 0., 0., 0., 0., .5]])
    log = np.zeros((1, 16)); log[0, 1] = -1.; log[0, 3] = 1.; log[0, 6] = 1.
    return cfg, gate, x, goal, obs, log


def test_default_conditional_filter_matches_base_safe_qp_before_any_trigger():
    cfg, _, x, obs, _, _, _, B, C, log = state()
    goal = np.array([10., 0., 3.])
    conditional = SafetyFilter(cfg, "conditional_detour").solve(
        x, goal, obs, log, B, C, np.zeros(3), np.zeros(3), cfg["control_dt"], certified=False)
    base = SafetyFilter(cfg, "no_detour").solve(
        x, goal, obs, log, B, C, np.zeros(3), np.zeros(3), cfg["control_dt"], certified=False)
    assert conditional.status == base.status == "ok"
    assert not conditional.detour["active"]
    assert conditional.detour["gate_reason"] == "base_safe_qp_default"
    np.testing.assert_allclose(conditional.control, base.control, atol=1e-12)


def test_persistent_input_box_reserve_warning_opens_gate():
    cfg, gate, x, goal, obs, log = gate_fixture()
    B = np.array([[1., 0., 0.]]); C = np.array([-1.9])
    out = None
    for _ in range(21):
        out = gate.pre_update(x, goal, obs, log, B, C, np.zeros(3), np.array([1.9, 0., 0.]), .0025, False)
    assert out["allow_activation"] and out["gate_reason"] == "conditional_feasibility"
    assert abs(out["minimum_common_set_margin_mps2"] - .1) < 1e-12


def test_joint_common_set_boundary_is_detected_when_each_row_alone_has_large_box_reserve():
    cfg, gate, x, goal, obs, log = gate_fixture()
    B = np.array([[1., 0., 0.], [-1., 0., 0.]])
    C = np.zeros(2)
    out = None
    for _ in range(21):
        out = gate.pre_update(x, goal, obs, log, B, C, np.zeros(3), np.zeros(3), .0025, False)
    assert out["allow_activation"] and out["gate_reason"] == "conditional_feasibility"
    assert abs(out["minimum_common_set_margin_mps2"]) < 1e-12
    assert out["minimum_individual_row_reserve_mps2"] == 2.0


def test_persistent_radial_braking_opens_gate_without_near_infeasibility():
    cfg, gate, x, goal, obs, log = gate_fixture()
    B = np.array([[-1., 0., 0.]]); C = np.array([2.])
    out = None
    for _ in range(141):
        out = gate.pre_update(x, goal, obs, log, B, C, np.zeros(3), np.array([-1., 0., 0.]), .0025, False)
    assert out["allow_activation"] and out["gate_reason"] == "conditional_radial_braking"
    assert out["radial_brake_mps2"] == 1.0 and out["transverse_correction_mps2"] == 0.0


def test_stagnation_window_opens_gate():
    cfg, gate, x, goal, obs, log = gate_fixture()
    B = np.array([[1., 0., 0.]]); C = np.array([2.])
    out = None
    for _ in range(11):
        out = gate.pre_update(x, goal, obs, log, B, C, np.zeros(3), np.zeros(3), .1, False)
        if out["allow_activation"]: break
    assert out["allow_activation"] and out["gate_reason"] == "conditional_stagnation"


def test_goal_recovery_release_and_hysteretic_rearm():
    cfg, gate, x, goal, obs, log = gate_fixture(); x[3] = -1.
    detour = {"active": True, "primary_clear": True, "primary_receding": True}
    out = None
    for _ in range(5):
        out = gate.post_augmented(x, goal, np.array([1., 0., 0.]), np.zeros(3), detour, .1)
    assert out["release"] and gate.safe_progress
    for _ in range(5):
        decision = gate.pre_update(x, goal, obs, log, np.array([[1., 0., 0.]]), np.array([2.]),
                                   np.zeros(3), np.zeros(3), .1, False)
    assert not gate.safe_progress and decision["rearmed"]
    assert not decision["allow_activation"]
