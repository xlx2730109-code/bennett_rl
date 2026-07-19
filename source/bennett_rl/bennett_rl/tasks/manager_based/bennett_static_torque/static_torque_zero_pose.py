# 测试关节角度与关节力矩的关系。

# 1. 设置该角度
# 2. reset / 写入关节状态
# 3. 仿真等待 settle-steps
# 4. 开始采样关节力矩

# 使用方法：
# 扫全部 8 个关节：
# python source\bennett_rl\bennett_rl\tasks\manager_based\bennett_static_torque\static_torque_zero_pose.py --headless --sweep --sweep-joints all 



from __future__ import annotations

import argparse
import csv
import math
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import torch

from isaaclab.app import AppLauncher


ACTIVE_JOINTS = [
    "FL_thigh",
    "FL_calf",
    "FR_thigh",
    "FR_calf",
    "RL_thigh",
    "RL_calf",
    "RR_thigh",
    "RR_calf",
]

FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
DEFAULT_TASK = "Isaac-BennettRL-Flat-Go2-8-Play-v0"

# Absolute USD/articulation joint positions for the nominal static pose.
# This is not an action offset.
BASE_JOINT_POS = {
    "FL_thigh": +0.1,
    "FR_thigh": -0.1,
    "RL_thigh": +0.1,
    "RR_thigh": -0.1,
    "FL_calf": -0.2,
    "FR_calf": -0.2,
    "RL_calf": -0.2,
    "RR_calf": -0.2,
}

ZERO_JOINT_POS = {
    "FL_thigh": 0.0,
    "FR_thigh": 0.0,
    "RL_thigh": 0.0,
    "RR_thigh": 0.0,
    "FL_calf": 0.0,
    "FR_calf": 0.0,
    "RL_calf": 0.0,
    "RR_calf": 0.0,
}

# Coupled all-joint sweep from the USD zero angle.
# If the thigh command is a, the calf command is -2a so the leg angle changes consistently.
COUPLED_SWEEP_MULTIPLIERS = {
    "FL_thigh": +1.0,
    "FR_thigh": -1.0,
    "RL_thigh": +1.0,
    "RR_thigh": -1.0,
    "FL_calf": -2.0,
    "FR_calf": -2.0,
    "RL_calf": -2.0,
    "RR_calf": -2.0,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def _parse_joint_override(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --joint item '{value}'. Expected NAME=RAD.")
        name, raw = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --joint item '{value}'. Joint name is empty.")
        overrides[name] = float(raw)
    return overrides


def _format_table(headers: list[str], rows: list[list[object]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, rows: list[dict[str, object]], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    lines = [f"# {title}", "", "## Metadata"]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Torque Summary", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _metadata_tensor_xyz(values: torch.Tensor, names: list[str]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "x": round(float(values[idx, 0]), 8),
            "y": round(float(values[idx, 1]), 8),
            "z": round(float(values[idx, 2]), 8),
        }
        for idx, name in enumerate(names)
    }


def _cell_name(row: int, col: int) -> str:
    letters = ""
    while col:
        col, remainder = divmod(col - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{row}"


def _xlsx_cell(row: int, col: int, value: object) -> str:
    ref = _cell_name(row, col)
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value), {'"': "&quot;"})
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _write_minimal_xlsx(path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = []
    for raw_name in sheets:
        safe_name = raw_name[:31]
        suffix = 1
        while safe_name in sheet_names:
            suffix += 1
            safe_name = f"{raw_name[:28]}_{suffix}"
        sheet_names.append(safe_name)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
"""
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for idx in range(1, len(sheet_names) + 1)
            )
            + "\n</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>
"""
            + "".join(
                f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
                for idx, name in enumerate(sheet_names, start=1)
            )
            + "\n</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
"""
            + "".join(
                f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
                for idx in range(1, len(sheet_names) + 1)
            )
            + "\n</Relationships>",
        )
        for sheet_idx, (raw_name, rows) in enumerate(sheets.items(), start=1):
            headers = list(rows[0].keys()) if rows else ["empty"]
            xml_rows = []
            xml_rows.append(
                f'<row r="1">'
                + "".join(_xlsx_cell(1, col_idx, header) for col_idx, header in enumerate(headers, start=1))
                + "</row>"
            )
            for row_idx, row in enumerate(rows, start=2):
                xml_rows.append(
                    f'<row r="{row_idx}">'
                    + "".join(_xlsx_cell(row_idx, col_idx, row.get(header, "")) for col_idx, header in enumerate(headers, start=1))
                    + "</row>"
                )
            max_col = _cell_name(max(1, len(rows) + 1), max(1, len(headers)))
            archive.writestr(
                f"xl/worksheets/sheet{sheet_idx}.xml",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<dimension ref="A1:{max_col}"/>
<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
<sheetData>{''.join(xml_rows)}</sheetData>
</worksheet>""",
            )


def _write_angle_torque_plots(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    sweep_joints: list[str],
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot skipped: matplotlib is not available in this Python environment.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[Path] = []
    for sweep_joint in sweep_joints:
        rows = [
            row
            for row in summary_rows
            if row.get("joint") == sweep_joint
            and row.get("sweep_joint") in (sweep_joint, "all_active_joints")
        ]
        if not rows:
            continue

        rows.sort(key=lambda row: float(row["sweep_angle_deg"]))
        target_deg = [float(row["sweep_angle_deg"]) for row in rows]
        actual_deg = [float(row["q_deg"]) for row in rows]
        torque = [float(row["applied_tau_mean_Nm"]) for row in rows]
        abs_torque = [abs(value) for value in torque]
        actual_points = sorted(zip(actual_deg, torque, abs_torque), key=lambda point: point[0])
        sorted_actual_deg = [point[0] for point in actual_points]
        sorted_actual_torque = [point[1] for point in actual_points]
        sorted_actual_abs_torque = [point[2] for point in actual_points]

        points_path = output_dir / f"{sweep_joint}_angle_torque_points.csv"
        _write_csv(
            points_path,
            [
                "target_angle_deg",
                "actual_q_deg",
                f"{sweep_joint}_mean_applied_tau_Nm",
                f"{sweep_joint}_abs_mean_applied_tau_Nm",
            ],
            [
                {
                    "target_angle_deg": target,
                    "actual_q_deg": actual,
                    f"{sweep_joint}_mean_applied_tau_Nm": tau,
                    f"{sweep_joint}_abs_mean_applied_tau_Nm": abs_tau,
                }
                for target, actual, tau, abs_tau in zip(target_deg, actual_deg, torque, abs_torque)
            ],
        )

        fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
        axes[0].plot(target_deg, torque, marker="o", linewidth=1.6, label="X = target sweep angle")
        axes[0].plot(sorted_actual_deg, sorted_actual_torque, marker="s", linewidth=1.3, label="X = measured actual q_deg")
        axes[0].axhline(0, color="0.55", linewidth=0.8)
        axes[0].set_title(f"{sweep_joint} angle vs mean applied torque")
        axes[0].set_xlabel("Angle (deg)")
        axes[0].set_ylabel("Signed torque (Nm)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(target_deg, abs_torque, marker="o", linewidth=1.6, label="abs torque vs target angle")
        axes[1].plot(sorted_actual_deg, sorted_actual_abs_torque, marker="s", linewidth=1.3, label="abs torque vs measured q_deg")
        axes[1].set_title(f"{sweep_joint} angle vs absolute mean torque")
        axes[1].set_xlabel("Angle (deg)")
        axes[1].set_ylabel("Abs torque (Nm)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plot_path = output_dir / f"{sweep_joint}_angle_torque_curve.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def _float_mean(values: list[float]) -> float:
    return float(torch.tensor(values).mean()) if values else float("nan")


def _float_std(values: list[float]) -> float:
    return float(torch.tensor(values).std(unbiased=False)) if values else float("nan")


def _metadata_rows(metadata: dict[str, object]) -> list[dict[str, object]]:
    return [{"key": key, "value": value} for key, value in metadata.items()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fixed-pose or swept-pose static torques for Bennett active joints."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Registered Isaac Lab task to load.")
    parser.add_argument("--num_envs", type=int, default=1, help="Keep this at 1 for static diagnostics.")
    parser.add_argument(
        "--pose",
        choices=("base_joint_pos", "zero_joint_pos", "task_default"),
        default="base_joint_pos",
        help="Base pose before per-joint overrides or sweep values.",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=RAD",
        help="Override one base joint position in radians. Can be passed multiple times.",
    )
    parser.add_argument("--include-passive", action="store_true", help="Also write passive USD joints.")
    parser.add_argument(
        "--disable-task-events",
        action="store_true",
        default=True,
        help="Disable task reset/randomization events so the static diagnostic is repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir(), help="Output directory.")
    parser.add_argument("--csv-name", default="settled_zero_pose_torque.csv", help="Summary CSV output file name.")
    parser.add_argument("--trials-csv-name", default="settled_zero_pose_torque_trials.csv", help="Per-trial CSV file.")
    parser.add_argument("--md-name", default="settled_zero_pose_torque.md", help="Markdown output file name.")
    parser.add_argument("--xlsx-name", default="关节角度-力矩数据.xlsx", help="Excel workbook output file name.")
    parser.add_argument("--no-plots", action="store_true", help="Do not write angle-torque PNG plots after sweep.")
    parser.add_argument("--num-trials", type=int, default=5, help="Repeated measurements per pose.")
    parser.add_argument("--settle-steps", type=int, default=600, help="Physics steps before sampling.")
    parser.add_argument("--sample-steps", type=int, default=100, help="Physics steps used for each average.")
    parser.add_argument("--sweep", action="store_true", help="Enable stepped joint-angle sweep.")
    parser.add_argument(
        "--sweep-joints",
        nargs="+",
        default=["FL_thigh"],
        # default=["all"],
        help="Joints to sweep. Use 'all' for all 8 active joints.",
    )
    parser.add_argument("--sweep-start-deg", type=float, default=0.0, help="First sweep angle in degrees.")
    parser.add_argument("--sweep-step-deg", type=float, default=0.5, help="Sweep angle increment in degrees.")
    parser.add_argument("--sweep-max-deg", type=float, default=30.0, help="Final sweep angle in degrees.")
    parser.add_argument(
        "--sweep-mode",
        choices=("signed_absolute", "positive_absolute", "additive"),
        default="signed_absolute",
        help=(
            "signed_absolute sets the swept joint to sign(base)*angle; "
            "positive_absolute sets it to +angle; additive sets it to base+angle."
        ),
    )
    parser.add_argument("--root-z", type=float, default=None, help="Optional absolute root z position before settling.")
    parser.add_argument("--ground-size", type=float, default=4.0, help="Local static ground box XY size in meters.")
    parser.add_argument("--ground-thickness", type=float, default=0.05, help="Local static ground box thickness.")
    parser.add_argument("--max-joint-vel", type=float, default=0.02, help="Settled threshold for max joint speed rad/s.")
    parser.add_argument("--max-base-lin-vel", type=float, default=0.02, help="Settled threshold for base linear speed m/s.")
    parser.add_argument("--max-base-ang-vel", type=float, default=0.05, help="Settled threshold for base angular speed rad/s.")
    parser.add_argument("--min-foot-force", type=float, default=1.0, help="Minimum mean vertical foot force in N.")
    parser.add_argument(
        "--disable-remote-assets",
        action="store_true",
        default=True,
        help="Disable terrain and command visualizer to avoid loading remote Isaac assets.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    if args.num_envs != 1:
        raise ValueError("Static torque diagnostics should use --num_envs 1.")
    if args.num_trials < 1:
        raise ValueError("--num-trials must be >= 1.")
    if args.sweep_step_deg <= 0.0:
        raise ValueError("--sweep-step-deg must be > 0.")
    if args.sweep_max_deg < args.sweep_start_deg:
        raise ValueError("--sweep-max-deg must be >= --sweep-start-deg.")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = None
    try:
        repo_source = _repo_root() / "source"
        if str(repo_source) not in sys.path:
            sys.path.insert(0, str(repo_source))

        import gymnasium as gym
        import isaaclab.sim as sim_utils

        import bennett_rl.tasks  # noqa: F401
        from isaaclab.assets import AssetBaseCfg
        from isaaclab_tasks.utils import parse_env_cfg

        use_fabric = not getattr(args, "disable_fabric", False)
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=use_fabric)

        if args.disable_remote_assets:
            env_cfg.scene.terrain = None
            env_cfg.scene.local_ground = AssetBaseCfg(
                prim_path="/World/local_ground",
                spawn=sim_utils.CuboidCfg(
                    size=(args.ground_size, args.ground_size, args.ground_thickness),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        rigid_body_enabled=True,
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    physics_material=env_cfg.sim.physics_material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5 * args.ground_thickness)),
            )
            if hasattr(env_cfg.commands, "base_velocity"):
                env_cfg.commands.base_velocity.debug_vis = False

        if args.disable_task_events:
            for event_name in list(vars(env_cfg.events).keys()):
                if not event_name.startswith("_"):
                    setattr(env_cfg.events, event_name, None)

        env = gym.make(args.task, cfg=env_cfg)
        robot = env.unwrapped.scene["robot"]

        joint_names = list(robot.data.joint_names)
        active_joint_ids = []
        missing = []
        for joint_name in ACTIVE_JOINTS:
            if joint_name in joint_names:
                active_joint_ids.append(joint_names.index(joint_name))
            else:
                missing.append(joint_name)
        if missing:
            raise RuntimeError(f"Missing active joints in articulation: {missing}. Available joints: {joint_names}")

        if args.pose == "task_default":
            base_joint_pos = robot.data.default_joint_pos.clone()
            base_pose_source = "task default_joint_pos"
        else:
            source_dict = BASE_JOINT_POS if args.pose == "base_joint_pos" else ZERO_JOINT_POS
            base_joint_pos = robot.data.default_joint_pos.clone()
            for joint_name, value in source_dict.items():
                if joint_name not in joint_names:
                    raise RuntimeError(f"{args.pose} contains unknown joint '{joint_name}'. Available: {joint_names}")
                base_joint_pos[:, joint_names.index(joint_name)] = value
            base_pose_source = f"{args.pose} absolute rad dictionary"

        overrides = _parse_joint_override(args.joint)
        for joint_name, value in overrides.items():
            if joint_name not in joint_names:
                raise RuntimeError(f"Unknown joint override '{joint_name}'. Available joints: {joint_names}")
            base_joint_pos[:, joint_names.index(joint_name)] = value

        if not args.include_passive:
            passive_ids = [idx for idx in range(len(joint_names)) if idx not in active_joint_ids]
            base_joint_pos[:, passive_ids] = robot.data.default_joint_pos[:, passive_ids]

        joint_vel = torch.zeros_like(robot.data.default_joint_vel)
        sweep_all_together = "all" in args.sweep_joints
        sweep_joints = args.sweep_joints
        if sweep_all_together:
            sweep_joints = ACTIVE_JOINTS
        for sweep_joint in sweep_joints:
            if sweep_joint not in ACTIVE_JOINTS:
                raise RuntimeError(f"Unknown sweep joint '{sweep_joint}'. Valid: {ACTIVE_JOINTS} or all")

        pose_specs = []
        if args.sweep:
            num_steps = int(round((args.sweep_max_deg - args.sweep_start_deg) / args.sweep_step_deg)) + 1
            for step_index in range(1, num_steps + 1):
                angle_deg = args.sweep_start_deg + (step_index - 1) * args.sweep_step_deg
                angle_rad = math.radians(angle_deg)
                pose_joint_pos = base_joint_pos.clone()
                target_joint_values_rad: dict[str, float] = {}
                for sweep_joint in sweep_joints:
                    joint_id = joint_names.index(sweep_joint)
                    base_value = float(base_joint_pos[0, joint_id])
                    direction = -1.0 if base_value < 0.0 else 1.0
                    if sweep_all_together:
                        target_value = COUPLED_SWEEP_MULTIPLIERS[sweep_joint] * angle_rad
                    elif args.sweep_mode == "signed_absolute":
                        target_value = direction * angle_rad
                    elif args.sweep_mode == "positive_absolute":
                        target_value = angle_rad
                    else:
                        target_value = base_value + angle_rad
                    pose_joint_pos[:, joint_id] = target_value
                    target_joint_values_rad[sweep_joint] = target_value
                pose_specs.append(
                    {
                        "sweep_joint": "all_active_joints" if sweep_all_together else sweep_joints[0],
                        "sweep_step": step_index,
                        "sweep_angle_deg": angle_deg,
                        "sweep_angle_rad": angle_rad,
                        "target_joint_value_rad": "all_active_joints" if sweep_all_together else target_joint_values_rad[sweep_joints[0]],
                        "target_joint_values_rad": target_joint_values_rad,
                        "joint_pos": pose_joint_pos,
                    }
                )
        else:
            pose_specs.append(
                {
                    "sweep_joint": "fixed_pose",
                    "sweep_step": 0,
                    "sweep_angle_deg": 0.0,
                    "sweep_angle_rad": 0.0,
                    "target_joint_value_rad": 0.0,
                    "target_joint_values_rad": {},
                    "joint_pos": base_joint_pos,
                }
            )

        contact_sensor = env.unwrapped.scene.sensors.get("contact_forces")
        foot_robot_body_ids = [robot.data.body_names.index(name) for name in FOOT_BODIES if name in robot.data.body_names]
        foot_pos_names = [name for name in FOOT_BODIES if name in robot.data.body_names]

        def reset_fixed_pose(pose_joint_pos: torch.Tensor) -> None:
            root_state = robot.data.default_root_state.clone()
            if args.root_z is not None:
                root_state[:, 2] = args.root_z
            root_state[:, 7:] = 0.0
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(pose_joint_pos, joint_vel)
            robot.set_joint_position_target(pose_joint_pos)
            robot.set_joint_velocity_target(joint_vel)
            robot.set_joint_effort_target(torch.zeros_like(pose_joint_pos))
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()

        def step_fixed_pose(pose_joint_pos: torch.Tensor) -> None:
            robot.set_joint_position_target(pose_joint_pos)
            robot.set_joint_velocity_target(joint_vel)
            robot.set_joint_effort_target(torch.zeros_like(pose_joint_pos))
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)

        def measure_trial(pose_spec: dict[str, object], trial_index: int) -> tuple[list[dict[str, object]], dict[str, object]]:
            pose_joint_pos = pose_spec["joint_pos"]
            reset_fixed_pose(pose_joint_pos)
            for _ in range(args.settle_steps):
                step_fixed_pose(pose_joint_pos)

            applied_samples = []
            computed_samples = []
            joint_pos_samples = []
            joint_vel_samples = []
            root_state_samples = []
            contact_samples = []
            for _ in range(args.sample_steps):
                step_fixed_pose(pose_joint_pos)
                applied_samples.append(robot.data.applied_torque.detach().clone().cpu())
                computed_samples.append(robot.data.computed_torque.detach().clone().cpu())
                joint_pos_samples.append(robot.data.joint_pos.detach().clone().cpu())
                joint_vel_samples.append(robot.data.joint_vel.detach().clone().cpu())
                root_state_samples.append(robot.data.root_state_w.detach().clone().cpu())
                if contact_sensor is not None:
                    contact_samples.append(contact_sensor.data.net_forces_w.detach().clone().cpu())

            applied = torch.stack(applied_samples, dim=0)[:, 0, :]
            computed = torch.stack(computed_samples, dim=0)[:, 0, :]
            measured_joint_pos_window = torch.stack(joint_pos_samples, dim=0)[:, 0, :]
            measured_joint_vel_window = torch.stack(joint_vel_samples, dim=0)[:, 0, :]
            root_state_window = torch.stack(root_state_samples, dim=0)[:, 0, :]
            body_pos = robot.data.body_pos_w.detach().clone().cpu()
            foot_pos = body_pos[0, foot_robot_body_ids, :] if foot_robot_body_ids else torch.empty((0, 3))

            root_pos = root_state_window[:, :3]
            root_vel = root_state_window[:, 7:13]
            base_height_mean = float(root_pos[:, 2].mean())
            base_height_std = float(root_pos[:, 2].std(unbiased=False))
            base_lin_vel_norm_mean = float(torch.linalg.norm(root_vel[:, :3], dim=1).mean())
            base_ang_vel_norm_mean = float(torch.linalg.norm(root_vel[:, 3:6], dim=1).mean())
            max_joint_vel_abs = float(measured_joint_vel_window[:, active_joint_ids].abs().max())
            settled = (
                max_joint_vel_abs <= args.max_joint_vel
                and base_lin_vel_norm_mean <= args.max_base_lin_vel
                and base_ang_vel_norm_mean <= args.max_base_ang_vel
            )

            contact_force_shape: object = "contact_sensor_not_available"
            contact_force_body_names: object = "contact_sensor_not_available"
            contact_force_body_name_source = "contact_sensor_not_available"
            foot_force_z_numeric: dict[str, float] = {}
            foot_contacts_above_threshold: dict[str, bool] = {}
            num_feet_in_contact = 0
            all_feet_in_contact = False
            total_force_z_mean = float("nan")
            if contact_samples and foot_robot_body_ids:
                contact = torch.stack(contact_samples, dim=0)[:, 0, :, :]
                contact_force_shape = tuple(contact.shape)
                sensor_body_names = list(getattr(contact_sensor, "body_names", []))
                if sensor_body_names and len(sensor_body_names) == contact.shape[1]:
                    contact_body_names = sensor_body_names
                    contact_force_body_name_source = "contact_sensor.body_names"
                elif len(robot.data.body_names) == contact.shape[1]:
                    contact_body_names = list(robot.data.body_names)
                    contact_force_body_name_source = "robot.data.body_names"
                elif len(FOOT_BODIES) == contact.shape[1]:
                    contact_body_names = FOOT_BODIES
                    contact_force_body_name_source = "FOOT_BODIES"
                else:
                    contact_body_names = []
                    contact_force_body_name_source = "unknown_order"
                contact_force_body_names = contact_body_names
                foot_contact_ids = [contact_body_names.index(name) for name in FOOT_BODIES if name in contact_body_names]
                foot_contact_names = [name for name in FOOT_BODIES if name in contact_body_names]
                foot_contact = (
                    contact[:, foot_contact_ids, :] if foot_contact_ids else torch.empty((contact.shape[0], 0, 3))
                )
                foot_force_z_mean = foot_contact[:, :, 2].mean(dim=0)
                total_force_z_mean = float(foot_force_z_mean.sum())
                foot_force_z_numeric = {
                    name: float(foot_force_z_mean[idx]) for idx, name in enumerate(foot_contact_names)
                }
                foot_contacts_above_threshold = {
                    name: value >= args.min_foot_force for name, value in foot_force_z_numeric.items()
                }
                num_feet_in_contact = sum(1 for value in foot_contacts_above_threshold.values() if value)
                all_feet_in_contact = len(foot_contacts_above_threshold) == len(FOOT_BODIES) and all(
                    foot_contacts_above_threshold.values()
                )

            valid_standing_sample = settled and all_feet_in_contact
            trial_rows: list[dict[str, object]] = []
            for joint_name, joint_id in zip(ACTIVE_JOINTS, active_joint_ids):
                target_joint_values_rad = pose_spec.get("target_joint_values_rad", {})
                if isinstance(target_joint_values_rad, dict):
                    target_joint_value_rad = target_joint_values_rad.get(joint_name, pose_spec["target_joint_value_rad"])
                else:
                    target_joint_value_rad = pose_spec["target_joint_value_rad"]
                q_rad = float(measured_joint_pos_window[:, joint_id].mean())
                q_deg = math.degrees(q_rad)
                applied_mean = float(applied[:, joint_id].mean())
                computed_mean = float(computed[:, joint_id].mean())
                row = {
                    "sweep_joint": pose_spec["sweep_joint"],
                    "sweep_step": pose_spec["sweep_step"],
                    "sweep_angle_deg": pose_spec["sweep_angle_deg"],
                    "sweep_angle_rad": pose_spec["sweep_angle_rad"],
                    "target_joint_value_rad": target_joint_value_rad,
                    "trial": trial_index,
                    "joint": joint_name,
                    "joint_id": joint_id,
                    "q_rad": q_rad,
                    "q_deg": q_deg,
                    "q_std_rad": float(measured_joint_pos_window[:, joint_id].std(unbiased=False)),
                    "joint_vel_abs_mean_rad_s": float(measured_joint_vel_window[:, joint_id].abs().mean()),
                    "applied_tau_mean_Nm": applied_mean,
                    "applied_tau_std_Nm": float(applied[:, joint_id].std(unbiased=False)),
                    "computed_tau_mean_Nm": computed_mean,
                    "computed_tau_std_Nm": float(computed[:, joint_id].std(unbiased=False)),
                    "abs_applied_tau_mean_Nm": abs(applied_mean),
                    "settled": settled,
                    "all_feet_in_contact": all_feet_in_contact,
                    "valid_standing_sample": valid_standing_sample,
                    "total_foot_force_z_mean_N": total_force_z_mean,
                }
                for foot_name in FOOT_BODIES:
                    row[f"{foot_name}_fz_N"] = foot_force_z_numeric.get(foot_name, float("nan"))
                trial_rows.append(row)

            trial_metrics = {
                "sweep_joint": pose_spec["sweep_joint"],
                "sweep_step": pose_spec["sweep_step"],
                "sweep_angle_deg": pose_spec["sweep_angle_deg"],
                "sweep_angle_rad": pose_spec["sweep_angle_rad"],
                "target_joint_value_rad": pose_spec["target_joint_value_rad"],
                "trial": trial_index,
                "base_height_mean_m": base_height_mean,
                "base_height_std_m": base_height_std,
                "base_lin_vel_norm_mean_m_s": base_lin_vel_norm_mean,
                "base_ang_vel_norm_mean_rad_s": base_ang_vel_norm_mean,
                "max_active_joint_vel_abs_rad_s": max_joint_vel_abs,
                "settled": settled,
                "foot_positions_w_m": _metadata_tensor_xyz(foot_pos, foot_pos_names),
                "contact_force_shape": contact_force_shape,
                "contact_force_body_name_source": contact_force_body_name_source,
                "contact_force_body_names": contact_force_body_names,
                "foot_force_z_numeric": foot_force_z_numeric,
                "foot_contacts_above_threshold": foot_contacts_above_threshold,
                "num_feet_in_contact": num_feet_in_contact,
                "all_feet_in_contact": all_feet_in_contact,
                "total_foot_force_z_mean_N": total_force_z_mean,
                "valid_standing_sample": valid_standing_sample,
            }
            return trial_rows, trial_metrics

        all_trial_rows: list[dict[str, object]] = []
        trial_metrics_list: list[dict[str, object]] = []
        for pose_index, pose_spec in enumerate(pose_specs, start=1):
            print(
                f"POSE {pose_index}/{len(pose_specs)}: {pose_spec['sweep_joint']} "
                f"{pose_spec['sweep_angle_deg']} deg",
                flush=True,
            )
            for trial_index in range(1, args.num_trials + 1):
                trial_rows, trial_metrics = measure_trial(pose_spec, trial_index)
                all_trial_rows.extend(trial_rows)
                trial_metrics_list.append(trial_metrics)

        summary_rows: list[dict[str, object]] = []
        torque_by_trial_rows: list[dict[str, object]] = []
        group_keys = []
        for row in all_trial_rows:
            key = (
                row["sweep_joint"],
                row["sweep_step"],
                row["sweep_angle_deg"],
                row["sweep_angle_rad"],
                row["target_joint_value_rad"],
                row["joint"],
                row["joint_id"],
            )
            if key not in group_keys:
                group_keys.append(key)

        for key in group_keys:
            sweep_joint, sweep_step, sweep_angle_deg, sweep_angle_rad, target_joint_value_rad, joint_name, joint_id = key
            rows_for_key = [
                row
                for row in all_trial_rows
                if (
                    row["sweep_joint"],
                    row["sweep_step"],
                    row["sweep_angle_deg"],
                    row["sweep_angle_rad"],
                    row["target_joint_value_rad"],
                    row["joint"],
                    row["joint_id"],
                )
                == key
            ]
            applied_values = [float(row["applied_tau_mean_Nm"]) for row in rows_for_key]
            computed_values = [float(row["computed_tau_mean_Nm"]) for row in rows_for_key]
            q_values = [float(row["q_rad"]) for row in rows_for_key]
            vel_values = [float(row["joint_vel_abs_mean_rad_s"]) for row in rows_for_key]
            applied_within_std = [float(row["applied_tau_std_Nm"]) for row in rows_for_key]
            foot_force_values = {
                foot_name: [float(row[f"{foot_name}_fz_N"]) for row in rows_for_key] for foot_name in FOOT_BODIES
            }
            applied_mean = _float_mean(applied_values)
            torque_by_trial = {
                "sweep_joint": sweep_joint,
                "sweep_step": sweep_step,
                "sweep_angle_deg": sweep_angle_deg,
                "sweep_angle_rad": sweep_angle_rad,
                "target_joint_value_rad": target_joint_value_rad,
                "joint": joint_name,
                "joint_id": joint_id,
                "num_trials": args.num_trials,
            }
            for trial_number in range(1, args.num_trials + 1):
                trial_rows = [row for row in rows_for_key if int(row["trial"]) == trial_number]
                torque_by_trial[f"trial_{trial_number}_applied_tau_Nm"] = (
                    float(trial_rows[0]["applied_tau_mean_Nm"]) if trial_rows else float("nan")
                )
            torque_by_trial[f"applied_tau_{args.num_trials}trial_mean_Nm"] = applied_mean
            torque_by_trial[f"applied_tau_{args.num_trials}trial_std_Nm"] = _float_std(applied_values)
            torque_by_trial[f"abs_applied_tau_{args.num_trials}trial_mean_Nm"] = abs(applied_mean)
            torque_by_trial["valid_standing_sample_all_trials"] = all(
                bool(row["valid_standing_sample"]) for row in rows_for_key
            )
            torque_by_trial_rows.append(torque_by_trial)
            summary = {
                "sweep_joint": sweep_joint,
                "sweep_step": sweep_step,
                "sweep_angle_deg": sweep_angle_deg,
                "sweep_angle_rad": sweep_angle_rad,
                "target_joint_value_rad": target_joint_value_rad,
                "joint": joint_name,
                "joint_id": joint_id,
                "num_trials": args.num_trials,
                "q_rad": _float_mean(q_values),
                "q_deg": math.degrees(_float_mean(q_values)),
                "q_between_trial_std_rad": _float_std(q_values),
                "joint_vel_abs_mean_rad_s": _float_mean(vel_values),
                "applied_tau_mean_Nm": applied_mean,
                "applied_tau_between_trial_std_Nm": _float_std(applied_values),
                "applied_tau_within_trial_std_mean_Nm": _float_mean(applied_within_std),
                "computed_tau_mean_Nm": _float_mean(computed_values),
                "computed_tau_between_trial_std_Nm": _float_std(computed_values),
                "abs_applied_tau_mean_Nm": abs(applied_mean),
                "settled_all_trials": all(bool(row["settled"]) for row in rows_for_key),
                "all_feet_in_contact_all_trials": all(bool(row["all_feet_in_contact"]) for row in rows_for_key),
                "valid_standing_sample_all_trials": all(bool(row["valid_standing_sample"]) for row in rows_for_key),
                "total_foot_force_z_mean_N": _float_mean([float(row["total_foot_force_z_mean_N"]) for row in rows_for_key]),
            }
            for foot_name, values in foot_force_values.items():
                summary[f"{foot_name}_fz_mean_N"] = _float_mean(values)
                summary[f"{foot_name}_fz_between_trial_std_N"] = _float_std(values)
            summary_rows.append(summary)

        first_metrics = trial_metrics_list[0]
        metadata = {
            "task": args.task,
            "pose": args.pose,
            "pose_joint_pos_source": base_pose_source,
            "base_joint_pos": BASE_JOINT_POS,
            "zero_joint_pos": ZERO_JOINT_POS,
            "joint_overrides": overrides,
            "sweep_enabled": args.sweep,
            "sweep_joints": sweep_joints if args.sweep else [],
            "sweep_all_together": sweep_all_together if args.sweep else False,
            "coupled_sweep_multipliers": COUPLED_SWEEP_MULTIPLIERS if args.sweep and sweep_all_together else {},
            "sweep_mode": args.sweep_mode,
            "sweep_start_deg": args.sweep_start_deg,
            "sweep_step_deg": args.sweep_step_deg,
            "sweep_max_deg": args.sweep_max_deg,
            "num_pose_specs": len(pose_specs),
            "num_trials": args.num_trials,
            "robot_usd": env.unwrapped.cfg.scene.robot.spawn.usd_path,
            "all_joint_names": joint_names,
            "body_names": list(robot.data.body_names),
            "foot_positions_w_m_first_trial": first_metrics["foot_positions_w_m"],
            "settle_steps": args.settle_steps,
            "sample_steps": args.sample_steps,
            "physics_dt": env.unwrapped.physics_dt,
            "disable_remote_assets": args.disable_remote_assets,
            "disable_task_events": args.disable_task_events,
            "contact_force_body_name_source": first_metrics["contact_force_body_name_source"],
            "contact_force_body_names": first_metrics["contact_force_body_names"],
            "note": (
                "TorqueByTrial has trial_1..trial_5 torque columns plus the 5-trial mean. "
                "Summary has one row per sweep angle and measured joint. Trials has raw trial rows; "
                "trial numbering starts at 1."
            ),
        }

        csv_path = args.output_dir / args.csv_name
        trial_csv_path = args.output_dir / args.trials_csv_name
        md_path = args.output_dir / args.md_name
        xlsx_path = args.output_dir / args.xlsx_name
        _write_csv(csv_path, list(summary_rows[0].keys()), summary_rows)
        _write_csv(trial_csv_path, list(all_trial_rows[0].keys()), all_trial_rows)
        _write_markdown(md_path, "Bennett Joint Angle Torque Data", summary_rows, metadata)
        plot_paths = []
        if args.sweep and not args.no_plots:
            plot_paths = _write_angle_torque_plots(args.output_dir, summary_rows, sweep_joints)
        xlsx_error = None
        try:
            _write_minimal_xlsx(
                xlsx_path,
                {
                    "TorqueByTrial": torque_by_trial_rows,
                    "Summary": summary_rows,
                    "Trials": all_trial_rows,
                    "Metadata": _metadata_rows(metadata),
                },
            )
        except PermissionError as exc:
            xlsx_error = exc

        print("JOINT_ANGLE_TORQUE_OK")
        print(f"Summary CSV: {csv_path}")
        print(f"Trials CSV: {trial_csv_path}")
        print(f"Markdown: {md_path}")
        if xlsx_error is None:
            print(f"Excel: {xlsx_path}")
        else:
            print(f"Excel skipped: {xlsx_path} is locked or not writable ({xlsx_error})")
        for plot_path in plot_paths:
            print(f"Plot: {plot_path}")
        print(f"Pose specs: {len(pose_specs)}")
        print(f"Trials per pose: {args.num_trials}")
        print("")
        preview_headers = list(summary_rows[0].keys())[:12]
        preview_rows = [[row[header] for header in preview_headers] for row in summary_rows[: min(12, len(summary_rows))]]
        print(_format_table(preview_headers, preview_rows))

    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
