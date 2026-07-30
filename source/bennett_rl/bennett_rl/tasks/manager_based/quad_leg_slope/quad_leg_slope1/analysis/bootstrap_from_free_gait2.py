"""Create a fresh-optimizer Slope1 checkpoint from a mature FreeGait2 policy."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_SOURCE = Path(
    "logs/rsl_rl/quad_leg_free_gait/quad_leg_free_gait2/flat/"
    "2026-07-29_04-57-26/model_2100.pt"
)
DEFAULT_OUTPUT = Path(
    "logs/rsl_rl/quad_leg_slope/quad_leg_slope1/"
    "bootstrap_free_gait2_model2100/model_0.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-BennettRL-QuadLeg-Slope1-v0")
parser.add_argument("--source_checkpoint", type=Path, default=DEFAULT_SOURCE)
parser.add_argument("--output_checkpoint", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import bennett_rl.tasks  # noqa: F401


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapped_policy_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map only the stochastic-parameter representation; require all else to match."""
    source_only = set(source_state) - set(target_state)
    target_only = set(target_state) - set(source_state)
    if source_only != {"std"} or target_only != {"log_std"}:
        raise RuntimeError(
            "Unexpected policy-state difference; refusing partial transfer: "
            f"source_only={sorted(source_only)}, target_only={sorted(target_only)}"
        )

    mapped: dict[str, torch.Tensor] = {}
    for key, target_value in target_state.items():
        if key == "log_std":
            source_value = torch.log(source_state["std"].clamp_min(1.0e-6))
        else:
            source_value = source_state[key]
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise RuntimeError(
                f"Shape mismatch for {key}: source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        mapped[key] = source_value.detach().to(
            device=target_value.device,
            dtype=target_value.dtype,
        )
    return mapped


def main() -> None:
    source = args_cli.source_checkpoint.resolve()
    output = args_cli.output_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output}")
    if source == output:
        raise ValueError("Source and output checkpoint must differ")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.log_dir = str(output.parent)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    agent_cfg.resume = False
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        source_data = torch.load(source, map_location="cpu", weights_only=False)
        source_state = source_data.get("model_state_dict")
        if not isinstance(source_state, dict):
            raise KeyError(f"{source} has no model_state_dict")

        policy = getattr(runner.alg, "policy", None)
        if policy is None:
            raise RuntimeError("RSL-RL runner has no policy module")
        mapped = _mapped_policy_state(source_state, policy.state_dict())
        policy.load_state_dict(mapped, strict=True)

        # Deliberately keep the newly constructed Slope1 optimizer.  Loading the
        # FreeGait2 optimizer would mix an incompatible std parameterization and
        # stale moment estimates with the new task.
        runner.current_learning_iteration = 0
        output.parent.mkdir(parents=True, exist_ok=False)
        runner.save(
            str(output),
            infos={
                "bootstrap_source": str(source),
                "bootstrap_source_sha256": _sha256(source),
                "state_transform": "std -> log_std; actor/critic copied exactly",
                "optimizer": "fresh Slope1 optimizer",
            },
        )

        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("iter") != 0:
            raise RuntimeError(f"Bootstrap iteration is not zero: {saved.get('iter')}")
        optimizer_state = saved.get("optimizer_state_dict", {}).get("state", {})
        if optimizer_state:
            raise RuntimeError("Bootstrap optimizer unexpectedly contains old moment state")
        saved_state = saved["model_state_dict"]
        for key, expected in mapped.items():
            if not torch.equal(saved_state[key].cpu(), expected.cpu()):
                raise RuntimeError(f"Saved state verification failed for {key}")

        print(f"[BOOTSTRAP] source={source}")
        print(f"[BOOTSTRAP] source_sha256={_sha256(source)}")
        print(f"[BOOTSTRAP] output={output}")
        print(f"[BOOTSTRAP] output_sha256={_sha256(output)}")
        print("[BOOTSTRAP] copied actor/critic; converted std->log_std; optimizer=fresh; iter=0")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
