"""Bindings for the independent conditional-detour copy."""
from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, sys, sysconfig

HERE = Path(__file__).resolve().parents[1]
MAIN = HERE / "current_method_conditional"
CODE = HERE / "code"
sys.dont_write_bytecode = True
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "runtime_cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "runtime_cache" / "matplotlib"))
for item in (str(CODE), str(MAIN)):
    if item not in sys.path: sys.path.insert(0, item)

METHODS = ("ra_conditional_detour",)
LABELS = {"ra_conditional_detour": "RA-DPCBF with conditional chi-Gamma assistance"}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return json.loads((HERE / "inputs" / "experiment_contract.json").read_text(encoding="utf-8"))


def source_manifest():
    result = {}
    for root in (CODE, MAIN):
        for path in sorted(root.rglob("*")):
            if path.is_file() and not any(part in ("__pycache__", ".pytest_cache") for part in path.parts):
                result[str(path.relative_to(HERE)).replace("\\", "/")] = sha256(path)
    for path in sorted((HERE / "inputs").glob("*")):
        if path.is_file() and path.name != "code_manifest.json":
            result[str(path.relative_to(HERE)).replace("\\", "/")] = sha256(path)
    return result


def runtime_environment():
    packages = ("numpy", "scipy", "numba", "osqp", "clarabel", "matplotlib")
    return {"python": sys.version, "python_cache_tag": sys.implementation.cache_tag,
            "python_soabi": sysconfig.get_config_var("SOABI"), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def verify_frozen_sources():
    manifest = HERE / "inputs" / "code_manifest.json"
    if not manifest.is_file(): raise FileNotFoundError("Source manifest is not frozen")
    expected = json.loads(manifest.read_text(encoding="utf-8")); current = source_manifest()
    if expected != current:
        missing = sorted(set(expected) - set(current)); added = sorted(set(current) - set(expected))
        changed = sorted(k for k in set(expected) & set(current) if expected[k] != current[k])
        raise RuntimeError(f"Frozen source mismatch: missing={missing}, added={added}, changed={changed}")
    locked = json.loads((HERE / "inputs" / "dependency_versions.json").read_text(encoding="utf-8"))
    actual = runtime_environment()
    if actual != locked: raise RuntimeError(f"Frozen runtime mismatch: locked={locked}, actual={actual}")
    return {"files": len(current), "manifest_sha256": sha256(manifest), "runtime": actual}
