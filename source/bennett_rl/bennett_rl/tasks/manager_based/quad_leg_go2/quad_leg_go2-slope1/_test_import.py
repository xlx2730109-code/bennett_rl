"""Test if the slope1 config loads correctly."""
import importlib
import sys

sys.path.insert(0, "source/bennett_rl")

# Try importing via the gym registration string path
module_path = "bennett_rl.tasks.manager_based.quad_leg_go2.quad_leg_go2-slope1.rough_env_cfg"

try:
    mod = importlib.import_module(module_path)
    print(f"IMPORT SUCCEEDED: {mod}")
    print(f"  __file__ = {mod.__file__}")
    cfg = mod.QuadLegGo2Slope1EnvCfg()
    print(f"  MIN_BASE_HEIGHT = {mod.MIN_BASE_HEIGHT}")
    print(f"  TARGET_BASE_HEIGHT = {mod.TARGET_BASE_HEIGHT}")
    print(f"  SLOPE_ANGLES count = {len(mod.SLOPE_ANGLES)}")
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}")

    # Try using importlib.util.spec_from_file_location
    import importlib.util
    from pathlib import Path

    p = Path("source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_go2/quad_leg_go2-slope1/rough_env_cfg.py")
    if p.exists():
        print(f"\nFile exists at {p.resolve()}")
        spec = importlib.util.spec_from_file_location("rough_env_cfg", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.QuadLegGo2Slope1EnvCfg()
        print(f"  ALTERNATE LOAD SUCCEEDED")
        print(f"  MIN_BASE_HEIGHT = {mod.MIN_BASE_HEIGHT}")
    else:
        print(f"\nFile NOT FOUND at {p.resolve()}")
