"""The copied conditional-detour RA-DPCBF adapter."""
from contextlib import contextmanager
from pathlib import Path

from bootstrap import LABELS


@contextmanager
def _native_scene(scene):
    import run_suite
    original = run_suite.load_scene
    def provider(name, mode="bounded", seed=17, duration=24.0):
        if abs(float(duration) - 24.0) > 1e-12: raise ValueError("Registered horizon is 24 s")
        return scene
    run_suite.load_scene = provider
    try: yield
    finally: run_suite.load_scene = original


def run_method(scene: dict, destination: Path, save_history: bool = True):
    import run_suite
    destination = Path(destination)
    with _native_scene(scene):
        row = run_suite.run(scene["scene_id"], "bounded", "conditional_detour",
                            seed=int(scene["metadata"]["seed"]), duration=24.0,
                            out=str(destination), save_history=save_history)
    row["variant"] = "ra_conditional_detour"
    row["method_label"] = LABELS["ra_conditional_detour"]
    return row
