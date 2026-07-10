"""Scenario definition: arm + objects + cameras + task, compiled to one MuJoCo model.

A scenario is a small YAML/JSON file describing the environment around the arm:

    name: pick_cube
    task: "pick up the red cube"
    settle_steps: 200            # physics steps after reset before control starts
    objects:
      - name: cube
        type: box                # box | sphere | cylinder | mesh
        size: [0.015, 0.015, 0.015]
        pos: [0.25, 0.0, 0.02]
        rgba: [0.9, 0.1, 0.1, 1]
        mass: 0.03
        yaw_range: [-3.14, 3.14]      # optional uniform randomization per reset
        pos_noise: [0.05, 0.05, 0.0]  # optional +/- uniform noise per axis
      - name: tray
        type: mesh
        file: /path/to/tray.stl       # any STL/OBJ; scale optional
        pos: [0.15, -0.15, 0.0]
        static: true                  # fixed to the world (no free joint)
    cameras:
      - {name: front, pos: [0.6, 0.0, 0.35], lookat: [0.2, 0.0, 0.1], fovy: 58}
      - {name: top,   pos: [0.25, 0.0, 0.7], lookat: [0.25, 0.0, 0.0], fovy: 58}

Build with `load_scenario(path)` then `build_model(scenario)`; the result is a
single mjModel whose first 6 qpos are the arm (same order as everywhere else),
followed by one free joint per non-static object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .config import so101_mjcf_path

DEFAULT_CAMERAS = [
    {"name": "front", "pos": [0.6, 0.0, 0.35], "lookat": [0.2, 0.0, 0.1], "fovy": 58.0},
    {"name": "top", "pos": [0.25, 0.0, 0.7], "lookat": [0.25, 0.0, 0.0], "fovy": 58.0},
]


@dataclass
class ObjectSpec:
    name: str
    type: str = "box"  # box | sphere | cylinder | mesh
    size: list = field(default_factory=lambda: [0.02, 0.02, 0.02])
    pos: list = field(default_factory=lambda: [0.25, 0.0, 0.05])
    rgba: list = field(default_factory=lambda: [0.8, 0.2, 0.2, 1.0])
    mass: float = 0.05
    file: str | None = None  # mesh file for type=mesh
    scale: float = 1.0
    static: bool = False
    yaw_range: list | None = None  # [lo, hi] radians
    pos_noise: list | None = None  # [+/-x, +/-y, +/-z] meters


@dataclass
class CameraSpec:
    name: str
    pos: list | None = None  # None -> camera must already exist in the arm MJCF
    lookat: list | None = None
    fovy: float = 58.0


@dataclass
class Scenario:
    name: str = "empty"
    task: str = ""
    objects: list[ObjectSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=lambda: [CameraSpec(**c) for c in DEFAULT_CAMERAS])
    settle_steps: int = 100
    source_path: str | None = None


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from YAML (or JSON — YAML is a superset)."""
    text = Path(path).read_text()
    try:
        import yaml

        raw = yaml.safe_load(text)
    except ImportError:
        raw = json.loads(text)
    objects = [ObjectSpec(**o) for o in raw.get("objects", [])]
    cameras = [CameraSpec(**c) for c in raw.get("cameras", DEFAULT_CAMERAS)]
    return Scenario(
        name=raw.get("name", Path(path).stem),
        task=raw.get("task", ""),
        objects=objects,
        cameras=cameras,
        settle_steps=int(raw.get("settle_steps", 100)),
        source_path=str(path),
    )


def _camera_quat(pos: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    """wxyz for a MuJoCo camera at pos looking at lookat (camera looks along -Z)."""
    forward = lookat - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ world_up) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.column_stack([right, up, -forward])  # camera x=right, y=up, z=-forward
    wxyz = np.empty(4)
    mujoco.mju_mat2Quat(wxyz, rot.reshape(-1))
    return wxyz


def build_model(scenario: Scenario, mjcf_path: Path | None = None) -> mujoco.MjModel:
    """Compile arm MJCF + floor + objects + cameras into one mjModel.

    qpos layout: [arm(5), gripper, obj1 free joint(7), obj2(7), ...] — the arm
    always comes first because it is compiled from the base spec.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path or so101_mjcf_path()))

    # Floor + a bit of light so cameras see something.
    floor_mat = spec.add_material(name="_scn_floor")
    floor_mat.rgba[:] = (0.35, 0.4, 0.45, 1.0)
    spec.worldbody.add_geom(
        name="_scn_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1],
        material="_scn_floor",
    )
    spec.worldbody.add_light(pos=[0.4, -0.4, 1.2], dir=[-0.3, 0.3, -1.0], castshadow=False)
    spec.worldbody.add_light(pos=[-0.4, 0.6, 1.0], dir=[0.3, -0.5, -1.0], castshadow=False)

    geom_types = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "mesh": mujoco.mjtGeom.mjGEOM_MESH,
    }
    for obj in scenario.objects:
        if obj.type not in geom_types:
            raise ValueError(f"object {obj.name!r}: unknown type {obj.type!r}")
        body = spec.worldbody.add_body(name=f"obj_{obj.name}", pos=obj.pos)
        kwargs = dict(
            name=f"obj_{obj.name}",
            type=geom_types[obj.type],
            rgba=obj.rgba,
            mass=obj.mass,
            condim=4,
            friction=[1.0, 0.02, 0.0005],
        )
        if obj.type == "mesh":
            if not obj.file:
                raise ValueError(f"object {obj.name!r}: type=mesh requires 'file'")
            mesh = spec.add_mesh(name=f"mesh_{obj.name}")
            mesh.file = str(Path(obj.file).expanduser().resolve())
            mesh.scale[:] = (obj.scale, obj.scale, obj.scale)
            kwargs["meshname"] = f"mesh_{obj.name}"
        else:
            size = list(obj.size) + [0.0] * (3 - len(obj.size))
            kwargs["size"] = size
        body.add_geom(**kwargs)
        if not obj.static:
            body.add_freejoint(name=f"free_{obj.name}")

    existing = {c.name for c in spec.cameras}  # e.g. the arm's built-in wrist_cam
    for cam in scenario.cameras:
        if cam.name in existing:
            continue
        if cam.pos is None or cam.lookat is None:
            raise ValueError(
                f"camera {cam.name!r} needs pos+lookat (or must exist in the arm model)"
            )
        pos = np.asarray(cam.pos, dtype=float)
        spec.worldbody.add_camera(
            name=cam.name,
            pos=pos,
            quat=_camera_quat(pos, np.asarray(cam.lookat, dtype=float)),
            fovy=cam.fovy,
        )

    return spec.compile()


def randomize_object_qpos(
    scenario: Scenario, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
) -> None:
    """Apply per-object pose randomization (call after mj_resetData)."""
    for obj in scenario.objects:
        if obj.static:
            continue
        jid = model.joint(f"free_{obj.name}").id
        adr = model.jnt_qposadr[jid]
        pos = np.asarray(obj.pos, dtype=float).copy()
        if obj.pos_noise:
            noise = np.asarray(obj.pos_noise, dtype=float)
            pos += rng.uniform(-noise, noise)
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if obj.yaw_range:
            yaw = rng.uniform(obj.yaw_range[0], obj.yaw_range[1])
            mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        data.qpos[adr : adr + 3] = pos
        data.qpos[adr + 3 : adr + 7] = quat
        vadr = model.jnt_dofadr[jid]
        data.qvel[vadr : vadr + 6] = 0.0
