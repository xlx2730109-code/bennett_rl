"""Quick config validation for slope1."""
import importlib.util
import sys

sys.path.insert(0, "source/bennett_rl")

spec = importlib.util.spec_from_file_location(
    "rough_env_cfg",
    "source/bennett_rl/bennett_rl/tasks/manager_based/quad_leg_go2/quad_leg_go2-slope1/rough_env_cfg.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = mod.QuadLegGo2Slope1EnvCfg()

print("Config loaded OK")
print(f"Robot init pos: {cfg.scene.robot.init_state.pos}")
print(f"num_envs: {cfg.scene.num_envs}")
print(f"terrain size: {cfg.scene.terrain.terrain_generator.size}")
print(f"terrain rows: {cfg.scene.terrain.terrain_generator.num_rows}")
print(f"terrain cols: {cfg.scene.terrain.terrain_generator.num_cols}")
print(f"use_terrain_origins: {cfg.scene.terrain.use_terrain_origins}")
print(f"max_init_terrain_level: {cfg.scene.terrain.max_init_terrain_level}")
print(f"height_scanner: {cfg.scene.height_scanner}")
print()

# Check terminations
print("=== TERMINATIONS ===")
for name in sorted(dir(cfg.terminations)):
    if name.startswith("_"):
        continue
    term = getattr(cfg.terminations, name)
    f = getattr(term, "func", "N/A")
    p = getattr(term, "params", "N/A")
    print(f"  {name}: func={f}  params={p}")

# Check replay buffer size
print()
print(f"num_steps_per_env: {cfg.__dict__.get('num_steps_per_env', 'N/A')}")
