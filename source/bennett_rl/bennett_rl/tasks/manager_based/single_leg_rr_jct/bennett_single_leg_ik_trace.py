# 单腿 sim2real 映射验证

# 先运行 python damiao_1.py 来控制电机，确保电机连接正确并且驱动安装好。这个脚本会初始化电机控制器，并在循环中发送控制命令。
# 再运行 python source\bennett_rl\bennett_rl\tasks\manager_based\single_leg_rr_jct\bennett_single_leg_ik_trace.py 来发送 UDP 数据包控制电机。
# 打开的isaacsim场景是 bennett_single_leg_ik_trace.usd，里面有一个单腿 Bennett 机器人，
# UDP 数据包里包含了 FR_thigh 和 FR_calf 的目标位置偏移，isaac sim 里会根据这个偏移计算出 FR_thigh 和 FR_calf 的目标位置，
# 并通过 UDP 发给 damiao_1.py 来控制真机的 FR_thigh 和 FR_calf 跟随。你可以在 isaac sim 里按键来调整 FR_thigh 和 FR_calf 的偏移，
# 观察真机的跟随效果。



import argparse
import json  #用来把目标角打包成文本数据
import math 
import socket   #用来发送 UDP 数据包
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# 直接 python 运行即可。
# 方案1：auto，按照 DEFAULT_JOINT_SEQUENCE_DEG 自动循环。
# 方案2：manual，键盘控制。默认用方案2，省得每次命令行都写 --mode manual。
DEFAULT_MODE = "manual"

# 你主要改这里：每一行是一个目标姿态，单位是度，格式为 (大腿角度, 小腿角度)。
# 这些角度都是“相对默认站立姿态”的偏移量，不是绝对关节角。
# 例如 (30.0, -10.0) 表示大腿比默认多转 +30 度，小腿比默认多转 -10 度。
DEFAULT_JOINT_SEQUENCE_DEG = [
    (30.0, 10.0),
    (10.0, 40.0),
    (-10.0, 40.0),
    (-10.0, 10.0),
    # (10.0, -15.0),
    # (10.0, 0.0),
]

# --joint_speed_deg_s：关节角速度，单位 度/秒，越大越快
# --sequence：命令行临时覆盖 DEFAULT_JOINT_SEQUENCE_DEG，例如 "30,0;30,20;-20,20;-20,-15"
# --trail_points：保留多少个轨迹点
# --trail_stride：隔多少仿真步记录一个轨迹点，越小轨迹越密
# --endpoint_radius：当前绿球大小
# --trail_radius：轨迹小绿球大小
# --cycles：循环次数，0 表示无限循环

parser = argparse.ArgumentParser(description="Bennett fixed-base RR single-leg joint motion test.")
parser.add_argument(
    "--mode",
    type=str,
    default=DEFAULT_MODE,
    choices=["auto", "manual"],
    help="auto runs the preset joint sequence; manual lets arrow keys control thigh/calf targets.",
)
parser.add_argument("--leg", type=str, default="RR", choices=["FL", "FR", "RL", "RR"], help="Leg prefix to test.")
parser.add_argument(
    "--joint_speed_deg_s",
    type=float,
    default=55.0,
    help="Joint interpolation speed in degrees per second. Larger means faster.",
)
parser.add_argument(
    "--sequence",
    type=str,
    default="",
    help='Optional keyframe override, e.g. "30,0;30,20;-20,20;-20,-15".',
)
parser.add_argument("--cycles", type=int, default=0, help="Number of motion cycles. Use 0 for infinite loop.")
parser.add_argument(
    "--manual_joint_speed_deg_s",
    type=float,
    default=35.0,   #键盘按住时，每个关节目标角速度 = 35 deg/s  ，即35度每秒，按住越久。
    help="Manual mode target-angle speed while an arrow key is held.",
)
parser.add_argument(
    "--manual_limit_deg",
    type=float,
    default=60.0,   #仿真键盘控制默认允许：±60 deg
    help="Manual mode max absolute offset from the default thigh/calf pose.",
)
parser.add_argument("--trail_points", type=int, default=780, help="Number of endpoint positions kept in the green trail.")
parser.add_argument("--trail_stride", type=int, default=1, help="Simulation steps between two saved trail points.")
parser.add_argument("--endpoint_radius", type=float, default=0.016, help="Current endpoint green sphere radius.")
parser.add_argument("--trail_radius", type=float, default=0.002, help="Trail green sphere radius.")
parser.add_argument(
    "--udp_target", #UDP 参数，默认 127.0.0.1:15001
    type=str,
    default="127.0.0.1:15001",
    help="UDP target for manual leg offset streaming, formatted as host:port.",
)
#IsaacSim 每秒发 250 次目标角，把当前仿真单腿目标角发给 damiao.py
parser.add_argument("--udp_rate_hz", type=float, default=250.0, help="UDP streaming rate in manual mode.")
parser.add_argument("--disable_udp", action="store_true", help="Disable UDP streaming in manual mode.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
for parent in THIS_DIR.parents:
    if (parent / "bennett_rl" / "assets").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from bennett_rl.assets.robots.bennett import BENNETT_CFG_V5  # noqa: E402


def make_robot() -> Articulation:
    """Spawn Bennett with a fixed root link."""
    robot_cfg = BENNETT_CFG_V5.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.spawn.articulation_props.fix_root_link = True
    return Articulation(cfg=robot_cfg)


def design_scene() -> Articulation:
    """Create a minimal scene for single-leg joint testing."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    return make_robot()


def reset_robot(robot: Articulation):
    """Reset root and joints to the configured Bennett default pose."""
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()


def clamp_to_soft_limits(robot: Articulation, joint_ids: list[int], joint_pos_des: torch.Tensor) -> torch.Tensor:
    """Clamp commanded joint positions to the robot soft joint limits."""
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    return torch.clamp(joint_pos_des, limits[:, :, 0], limits[:, :, 1])


def parse_joint_sequence_deg(sequence_text: str) -> list[tuple[float, float]]:
    """Parse a thigh/calf keyframe sequence in degrees."""
    if not sequence_text:
        return DEFAULT_JOINT_SEQUENCE_DEG

    sequence = []
    for item in sequence_text.split(";"):
        item = item.strip()
        if not item:
            continue
        values = [value.strip() for value in item.split(",")]
        if len(values) != 2:
            raise ValueError(f"Invalid sequence item '{item}'. Expected 'thigh_deg,calf_deg'.")
        sequence.append((float(values[0]), float(values[1])))

    if not sequence:
        raise ValueError("Empty --sequence. Expected items like '30,0;30,20;-20,20'.")  
    return sequence


def make_joint_targets_from_sequence(
    default_leg_target: torch.Tensor, sequence_deg: list[tuple[float, float]], device: str
) -> list[tuple[str, torch.Tensor]]:
    """Create absolute joint targets from degree offsets relative to default pose."""
    targets = []
    for index, (thigh_deg, calf_deg) in enumerate(sequence_deg, start=1):
        offset_rad = torch.tensor(
            [[math.radians(thigh_deg), math.radians(calf_deg)]],
            device=device,
            dtype=default_leg_target.dtype,
        )
        label = f"keyframe {index}: thigh={thigh_deg:+.1f}deg, calf={calf_deg:+.1f}deg"
        targets.append((label, default_leg_target + offset_rad))
    return targets


def update_endpoint_marker(
    robot: Articulation,
    endpoint_marker: VisualizationMarkers,
    endpoint_body_id: int,
    endpoint_trail: list[torch.Tensor],
    step_index: int,
):
    """Update current endpoint marker and its green trail."""
    endpoint_pos_w = robot.data.body_pos_w[:, endpoint_body_id].clone()
    if step_index % max(1, int(args_cli.trail_stride)) == 0:
        endpoint_trail.append(endpoint_pos_w)
        del endpoint_trail[: max(0, len(endpoint_trail) - args_cli.trail_points)]
    marker_positions = torch.cat([*endpoint_trail, endpoint_pos_w], dim=0)
    marker_indices = torch.zeros(marker_positions.shape[0], device=robot.device, dtype=torch.long)
    marker_indices[-1] = 1
    endpoint_marker.visualize(marker_positions, marker_indices=marker_indices)


class ManualLegKeyboard:
    """Small keyboard state holder for manual single-leg joint testing."""

    def __init__(self):
        import carb.input  # noqa: PLC0415
        import omni.appwindow  # noqa: PLC0415

        self._carb_input = carb.input
        self._pressed_keys: set[str] = set()
        self.reset_requested = False
        self.clear_trail_requested = False
        self.quit_requested = False

        appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event, *args, **kwargs):
        key_name = event.input.name
        if event.type == self._carb_input.KeyboardEventType.KEY_PRESS:
            if key_name in {"UP", "DOWN", "RIGHT", "LEFT"}:
                self._pressed_keys.add(key_name)
            elif key_name == "R":
                self.reset_requested = True
            elif key_name == "C":
                self.clear_trail_requested = True
            elif key_name in {"Q", "ESCAPE"}:
                self.quit_requested = True
        elif event.type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._pressed_keys.discard(key_name)
        return True

    def joint_direction(self) -> tuple[float, float]:
        """Return normalized target directions for thigh and calf."""
        thigh_direction = float("UP" in self._pressed_keys) - float("DOWN" in self._pressed_keys)
        calf_direction = float("RIGHT" in self._pressed_keys) - float("LEFT" in self._pressed_keys)
        return thigh_direction, calf_direction

    def clear_one_shot_flags(self):
        self.reset_requested = False
        self.clear_trail_requested = False


def interpolate_joint_targets(
    robot: Articulation,
    sim: SimulationContext,
    endpoint_marker: VisualizationMarkers,
    endpoint_body_id: int,
    endpoint_trail: list[torch.Tensor],
    active_joint_ids: list[int],
    leg_joint_ids: list[int],
    default_active_target: torch.Tensor,
    start_target: torch.Tensor,
    end_target: torch.Tensor,
    label: str,
    duration_s: float,
    elapsed_s: float,
) -> float:
    """Linearly interpolate selected leg targets at uniform angular speed."""
    sim_dt = sim.get_physics_dt()
    num_steps = max(1, int(duration_s / sim_dt))
    end_target = clamp_to_soft_limits(robot, leg_joint_ids, end_target)

    print(
        f"[SEGMENT] {label}: duration={duration_s:.2f}s, "
        f"target=({end_target[0, 0].item():+.3f}, {end_target[0, 1].item():+.3f}) rad"
    )

    for step in range(num_steps):
        alpha = (step + 1) / num_steps
        leg_target = (1.0 - alpha) * start_target + alpha * end_target
        leg_target = clamp_to_soft_limits(robot, leg_joint_ids, leg_target)

        robot.set_joint_position_target(default_active_target, joint_ids=active_joint_ids)
        robot.set_joint_position_target(leg_target, joint_ids=leg_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        update_endpoint_marker(robot, endpoint_marker, endpoint_body_id, endpoint_trail, step)

        if step % max(1, int(1.0 / sim_dt)) == 0:
            leg_pos = robot.data.joint_pos[:, leg_joint_ids][0]
            leg_torque = robot.data.applied_torque[:, leg_joint_ids][0]
            print(
                f"[STEP] t={elapsed_s + step * sim_dt:6.2f}s "
                f"q=({leg_pos[0].item():+.3f}, {leg_pos[1].item():+.3f}) "
                f"tau=({leg_torque[0].item():+.2f}, {leg_torque[1].item():+.2f})"
            )

    return elapsed_s + num_steps * sim_dt


def run_manual_control(
    robot: Articulation,
    sim: SimulationContext,
    endpoint_marker: VisualizationMarkers,
    endpoint_body_id: int,
    active_joint_ids: list[int],
    leg_joint_ids: list[int],
    default_active_target: torch.Tensor,
    default_leg_target: torch.Tensor,
):
    """Run keyboard-controlled thigh/calf target offsets."""
    keyboard = ManualLegKeyboard()
    sim_dt = sim.get_physics_dt()
    target = default_leg_target.clone()
    endpoint_trail: list[torch.Tensor] = []
    elapsed_s = 0.0
    step_index = 0
    manual_limit_rad = math.radians(args_cli.manual_limit_deg)
    manual_speed_rad_s = math.radians(args_cli.manual_joint_speed_deg_s)
    udp_sock = None
    udp_target = None
    udp_period_s = 1.0 / max(args_cli.udp_rate_hz, 1.0e-6)
    last_udp_send_s = -1.0e9

    if not args_cli.disable_udp and args_cli.udp_target:
        udp_host, udp_port = args_cli.udp_target.rsplit(":", 1)
        udp_target = (udp_host, int(udp_port))
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[UDP] streaming manual {args_cli.leg} offsets to {udp_host}:{udp_port} at {args_cli.udp_rate_hz:.1f} Hz")

    print("[MANUAL] Keyboard control enabled.")
    print("[MANUAL] Up/Down = thigh +/-, Left/Right = calf -/+, R = reset, C = clear trail, Q/Esc = quit.")
    print(
        f"[MANUAL] speed={args_cli.manual_joint_speed_deg_s:.1f} deg/s, "
        f"limit=+/-{args_cli.manual_limit_deg:.1f} deg from default pose"
    )

    while simulation_app.is_running() and not keyboard.quit_requested:
        thigh_dir, calf_dir = keyboard.joint_direction()
        delta = torch.tensor(
            [[thigh_dir, calf_dir]],
            device=robot.device,
            dtype=target.dtype,
        ) * manual_speed_rad_s * sim_dt
        target = target + delta

        limit_min = default_leg_target - manual_limit_rad
        limit_max = default_leg_target + manual_limit_rad
        target = torch.clamp(target, limit_min, limit_max)
        target = clamp_to_soft_limits(robot, leg_joint_ids, target)

        if keyboard.reset_requested:
            target = default_leg_target.clone()
            print("[MANUAL] reset target to default pose")
        if keyboard.clear_trail_requested:
            endpoint_trail.clear()
            print("[MANUAL] cleared endpoint trail")
        keyboard.clear_one_shot_flags()
        # 核心联动代码：manual 里发 UDP，把当前 RR 单腿目标角发给 damiao.py / damiao_*.py。
        if udp_sock is not None and udp_target is not None and elapsed_s - last_udp_send_s >= udp_period_s:
            offset_rad = (target - default_leg_target)[0]
            payload = {
                "leg": args_cli.leg,
                #发的不是 CAN 命令，而是仿真关节相对默认站姿的偏移量
                "thigh_offset_rad": float(offset_rad[0].item()),
                "calf_offset_rad": float(offset_rad[1].item()),
                "time_s": float(elapsed_s),
            }
            udp_sock.sendto(json.dumps(payload, separators=(",", ":")).encode("ascii"), udp_target)
            last_udp_send_s = elapsed_s
        #这一块是仿真自己动
        robot.set_joint_position_target(default_active_target, joint_ids=active_joint_ids)
        robot.set_joint_position_target(target, joint_ids=leg_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        update_endpoint_marker(robot, endpoint_marker, endpoint_body_id, endpoint_trail, step_index)

        if step_index % max(1, int(0.5 / sim_dt)) == 0:
            leg_pos = robot.data.joint_pos[:, leg_joint_ids][0]
            leg_torque = robot.data.applied_torque[:, leg_joint_ids][0]
            offset_deg = torch.rad2deg(target - default_leg_target)[0]
            print(
                f"[MANUAL] t={elapsed_s:6.2f}s "
                f"offset_deg=({offset_deg[0].item():+.1f}, {offset_deg[1].item():+.1f}) "
                f"q=({leg_pos[0].item():+.3f}, {leg_pos[1].item():+.3f}) "
                f"tau=({leg_torque[0].item():+.2f}, {leg_torque[1].item():+.2f})"
            )

        elapsed_s += sim_dt
        step_index += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.8, -1.2, 0.65], target=[0.0, -0.08, 0.12])

    robot = design_scene()
    marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/BennettSingleLegEndpoint"
    marker_cfg.markers["target_far"].radius = args_cli.trail_radius
    marker_cfg.markers["target_far"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    marker_cfg.markers["target_near"].radius = args_cli.endpoint_radius
    marker_cfg.markers["target_near"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    endpoint_marker = VisualizationMarkers(marker_cfg)

    sim.reset()
    reset_robot(robot)

    leg_joint_names = [f"{args_cli.leg}_thigh", f"{args_cli.leg}_calf"]
    leg_joint_ids, resolved_joint_names = robot.find_joints(leg_joint_names, preserve_order=True)
    active_joint_ids, active_joint_names = robot.find_joints([".*_thigh", ".*_calf"])
    endpoint_body_ids, resolved_endpoint_body_names = robot.find_bodies([f"{args_cli.leg}_2"], preserve_order=True)
    if len(leg_joint_ids) != 2:
        raise RuntimeError(f"Failed to resolve {args_cli.leg} leg joints: {resolved_joint_names}")
    if len(endpoint_body_ids) != 1:
        raise RuntimeError(f"Failed to resolve endpoint body {args_cli.leg}_2: {resolved_endpoint_body_names}")

    default_active_target = robot.data.default_joint_pos[:, active_joint_ids].clone()
    default_leg_target = robot.data.default_joint_pos[:, leg_joint_ids].clone()
    sequence_deg = parse_joint_sequence_deg(args_cli.sequence)
    cycle_targets = make_joint_targets_from_sequence(default_leg_target, sequence_deg, robot.device)

    print("[INFO] Bennett fixed-base single-leg joint motion")
    print(f"[INFO] leg={args_cli.leg}, joints={resolved_joint_names}")
    print(f"[INFO] endpoint marker body={resolved_endpoint_body_names[0]}")
    print(f"[INFO] active joints held at default: {active_joint_names}")
    print(
        f"[INFO] default q=({default_leg_target[0, 0].item():+.3f}, "
        f"{default_leg_target[0, 1].item():+.3f}) rad"
    )
    print(f"[INFO] joint_speed={args_cli.joint_speed_deg_s:.3f} deg/s")
    print(f"[INFO] mode={args_cli.mode}")

    if args_cli.mode == "manual":
        run_manual_control(
            robot,
            sim,
            endpoint_marker,
            endpoint_body_ids[0],
            active_joint_ids,
            leg_joint_ids,
            default_active_target,
            default_leg_target,
        )
    else:
        print(f"[INFO] cycles={'infinite' if args_cli.cycles == 0 else args_cli.cycles}")
        print("[INFO] joint sequence:")
        for index, (thigh_deg, calf_deg) in enumerate(sequence_deg, start=1):
            print(f"  {index}: thigh={thigh_deg:+.1f} deg, calf={calf_deg:+.1f} deg")

        elapsed_s = 0.0
        current_target = default_leg_target.clone()
        endpoint_trail: list[torch.Tensor] = []
        cycle_index = 0
        while simulation_app.is_running() and (args_cli.cycles == 0 or cycle_index < args_cli.cycles):
            cycle_index += 1
            print(f"[CYCLE] {cycle_index}" if args_cli.cycles == 0 else f"[CYCLE] {cycle_index}/{args_cli.cycles}")
            for label, next_target in cycle_targets:
                max_delta_deg = math.degrees(torch.max(torch.abs(next_target - current_target)).item())
                duration_s = max(0.1, max_delta_deg / max(args_cli.joint_speed_deg_s, 1.0e-6))
                elapsed_s = interpolate_joint_targets(
                    robot,
                    sim,
                    endpoint_marker,
                    endpoint_body_ids[0],
                    endpoint_trail,
                    active_joint_ids,
                    leg_joint_ids,
                    default_active_target,
                    current_target,
                    next_target,
                    label,
                    duration_s,
                    elapsed_s,
                )
                current_target = clamp_to_soft_limits(robot, leg_joint_ids, next_target)


if __name__ == "__main__":
    main()
    simulation_app.close()
