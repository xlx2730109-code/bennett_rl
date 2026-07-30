"""
Quick debug: print config values and check termination threshold.
Run with: D:\Conda\envs\env_isaaclab\python.exe source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_go2/quad_leg_go2-slope1/_debug_reset.py
"""
import os
import sys

# Find the real package path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_root = os.path.abspath(os.path.join(_this_dir, "../../../../../../.."))
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

# Import the package chain - this triggers all __init__.py files
import bennett_rl.tasks  # noqa: F401

# Now load the config
from isaaclab_tasks.utils import load_cfg_from_registry

cfg = load_cfg_from_registry("Isaac-BennettRL-Go2-Slope1-v0", "env_cfg_entry_point")

print(f"=== SLOPE1 CONFIG ===")
print(f"MIN_BASE_HEIGHT = {cfg.__class__.MIN_BASE_HEIGHT}")
print(f"init_state.pos = {cfg.scene.robot.init_state.pos}")
print(f"num_envs = {cfg.scene.num_envs}")
print(f"use_terrain_origins = {cfg.scene.terrain.use_terrain_origins}")
print(f"terrain rows = {cfg.scene.terrain.terrain_generator.num_rows}")
print(f"terrain cols = {cfg.scene.terrain.terrain_generator.num_cols}")
print()

# Check terminations
print("=== TERMINATIONS ===")
for name in dir(cfg.terminations):
    if name.startswith("_"):
        continue
    term = getattr(cfg.terminations, name)
    if term is None:
        print(f"  {name}: None")
        continue
    f = getattr(term, "func", "N/A")
    p = getattr(term, "params", {})
    print(f"  {name}: func={f.__name__ if hasattr(f, '__name__') else f}")
    print(f"         params={p}")

print()
print("=== CHECK root_height threshold ===")
rt = cfg.terminations.root_height
print(f"minimum_height = {rt.params.get('minimum_height', 'NOT SET')}")

print()
print("=== All good ===")
