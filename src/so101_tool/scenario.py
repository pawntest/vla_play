"""Scenario definition: arm + environment + objects + cameras + task, compiled
to one MuJoCo model.

A scenario is a small YAML file describing everything around the arm. Full
reference: docs/scenario_reference.md. Overview:

    name: tabletop
    task: "put the cube on the towel"
    settle_steps: 200
    environment:                       # optional
      floor: {rgba: [0.35, 0.4, 0.45, 1], checker: true}
      table: {size: [0.7, 0.5], height: 0.0, thickness: 0.04, rgba: [0.55, 0.42, 0.3, 1]}
      props:                           # static geometry: walls, shelves, backdrops
        - {name: wall, type: box, size: [0.01, 0.4, 0.2], pos: [-0.15, 0, 0.2]}
      light_diffuse: 0.8
    objects:
      - name: cube                     # rigid: box | sphere | cylinder | mesh
        type: box
        size: [0.015, 0.015, 0.015]
        pos: [0.25, 0.0, 0.02]
        rgba: [0.9, 0.1, 0.1, 1]
        mass: 0.03
        pos_noise: [0.04, 0.06, 0.0]   # per-reset randomization
        yaw_range: [-0.6, 0.6]
      - name: towel                    # deformable cloth (MuJoCo native flex)
        type: cloth
        size: [0.12, 0.12]             # meters
        pos: [0.2, -0.12, 0.03]
        rgba: [0.2, 0.4, 0.9, 1]
        mass: 0.03
        cloth: {resolution: 9, young: 3.0e4, poisson: 0.1, thickness: 0.01, damping: 0.01}
    cameras:                           # any number, world-fixed or body-attached
      - {name: front, pos: [0.55, -0.35, 0.35], lookat: [0.25, 0, 0.05], fovy: 52}
      - {name: wrist_cam}              # built into the arm model
      - {name: grip_cam, attach_to: gripper, pos: [0.02, 0, -0.04], lookat: [0.1, 0, -0.25]}

Coordinates: arm base at the origin, +X forward, +Z up, meters/radians.
`build_model(scenario)` returns one mjModel whose first 6 qpos are the arm,
followed by one free joint per rigid object, then cloth vertex DOFs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .config import so101_mjcf_path

DEFAULT_CAMERAS = [
    {"name": "front", "pos": [0.6, 0.0, 0.35], "lookat": [0.2, 0.0, 0.1], "fovy": 58.0},
    {"name": "top", "pos": [0.25, 0.0, 0.7], "lookat": [0.25, 0.0, 0.0], "fovy": 58.0},
]
# arm bodies a camera can attach to (from the vendored MJCF)
ATTACHABLE_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
DEFAULT_CLOTH = {"resolution": 9, "young": 3.0e4, "poisson": 0.1, "thickness": 0.01,
                 "damping": 0.01}

RIGID_TYPES = ("box", "sphere", "cylinder", "mesh")


@dataclass
class ObjectSpec:
    """One object: a rigid body (free or static) or a deformable cloth."""

    name: str
    type: str = "box"  # box | sphere | cylinder | mesh | cloth
    size: list = field(default_factory=lambda: [0.02, 0.02, 0.02])
    pos: list = field(default_factory=lambda: [0.25, 0.0, 0.05])
    rgba: list = field(default_factory=lambda: [0.8, 0.2, 0.2, 1.0])
    mass: float = 0.05
    file: str | None = None  # mesh file for type=mesh
    scale: float = 1.0
    static: bool = False
    yaw_range: list | None = None  # [lo, hi] radians, randomized per reset
    pos_noise: list | None = None  # [+/-x, +/-y, +/-z] meters, randomized per reset
    cloth: dict | None = None  # type=cloth: resolution/young/poisson/thickness/damping

    @property
    def is_cloth(self) -> bool:
        return self.type == "cloth"

    def cloth_params(self) -> dict:
        return {**DEFAULT_CLOTH, **(self.cloth or {})}


@dataclass
class CameraSpec:
    """A render camera: world-fixed (pos/lookat in world frame), attached to an
    arm body (attach_to + pos/lookat in that body's frame), or a camera that
    already exists in the arm MJCF (name only, e.g. wrist_cam)."""

    name: str
    pos: list | None = None
    lookat: list | None = None
    fovy: float = 58.0
    attach_to: str | None = None  # arm body name (see ATTACHABLE_BODIES)


@dataclass
class TableSpec:
    size: list = field(default_factory=lambda: [0.7, 0.5])  # [width x, depth y] m
    height: float = 0.0  # z of the tabletop surface (arm base sits at z=0)
    thickness: float = 0.04
    pos: list = field(default_factory=lambda: [0.15, 0.0])  # tabletop center [x, y]
    rgba: list = field(default_factory=lambda: [0.55, 0.42, 0.3, 1.0])


@dataclass
class FloorSpec:
    rgba: list = field(default_factory=lambda: [0.35, 0.4, 0.45, 1.0])
    rgba2: list | None = None  # second checker color; None -> solid
    checker: bool = False


@dataclass
class EnvironmentSpec:
    """Static surroundings: floor, optional table under the arm, props, light."""

    floor: FloorSpec = field(default_factory=FloorSpec)
    table: TableSpec | None = None
    props: list[ObjectSpec] = field(default_factory=list)
    light_diffuse: float = 0.8


@dataclass
class Scenario:
    name: str = "empty"
    task: str = ""
    environment: EnvironmentSpec = field(default_factory=EnvironmentSpec)
    objects: list[ObjectSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(
        default_factory=lambda: [CameraSpec(**c) for c in DEFAULT_CAMERAS]
    )
    settle_steps: int = 100
    source_path: str | None = None

    @property
    def rigid_objects(self) -> list[ObjectSpec]:
        return [o for o in self.objects if not o.is_cloth]

    @property
    def cloth_objects(self) -> list[ObjectSpec]:
        return [o for o in self.objects if o.is_cloth]


# --------------------------------------------------------------------------
# YAML I/O
# --------------------------------------------------------------------------


def _environment_from_raw(raw: dict) -> EnvironmentSpec:
    floor = FloorSpec(**raw.get("floor", {}))
    table = TableSpec(**raw["table"]) if raw.get("table") else None
    props = [ObjectSpec(**p) for p in raw.get("props", [])]
    return EnvironmentSpec(
        floor=floor, table=table, props=props,
        light_diffuse=float(raw.get("light_diffuse", 0.8)),
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Scenario(
        name=raw.get("name", Path(path).stem),
        task=raw.get("task", ""),
        environment=_environment_from_raw(raw.get("environment", {})),
        objects=[ObjectSpec(**o) for o in raw.get("objects", [])],
        cameras=[CameraSpec(**c) for c in raw.get("cameras", DEFAULT_CAMERAS)],
        settle_steps=int(raw.get("settle_steps", 100)),
        source_path=str(path),
    )


def save_scenario(scenario: Scenario, path: str | Path) -> Path:
    """Write a scenario back to YAML (inverse of load_scenario)."""
    import yaml

    def clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None and k != "source_path"}

    env = scenario.environment
    raw = {
        "name": scenario.name,
        "task": scenario.task,
        "settle_steps": scenario.settle_steps,
        "environment": {
            "floor": clean(dataclasses.asdict(env.floor)),
            **({"table": clean(dataclasses.asdict(env.table))} if env.table else {}),
            **({"props": [clean(dataclasses.asdict(p)) for p in env.props]}
               if env.props else {}),
            "light_diffuse": env.light_diffuse,
        },
        "objects": [clean(dataclasses.asdict(o)) for o in scenario.objects],
        "cameras": [clean(dataclasses.asdict(c)) for c in scenario.cameras],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
    return path


# --------------------------------------------------------------------------
# model building
# --------------------------------------------------------------------------


def _camera_quat(pos: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    """wxyz for a MuJoCo camera at pos looking at lookat (camera looks along -Z).
    Works in whatever frame pos/lookat are expressed in (world or body-local)."""
    forward = lookat - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(forward @ world_up) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rot = np.column_stack([right, up, -forward])
    wxyz = np.empty(4)
    mujoco.mju_mat2Quat(wxyz, rot.reshape(-1))
    return wxyz


def _add_environment(spec: mujoco.MjSpec, env: EnvironmentSpec) -> None:
    floor_z = (env.table.height - 0.4) if env.table else 0.0
    mat = spec.add_material(name="_scn_floor")
    if env.floor.checker:
        tex = spec.add_texture(
            name="_scn_floor_tex",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            width=512, height=512,
        )
        c1 = env.floor.rgba
        c2 = env.floor.rgba2 or [c * 0.6 for c in env.floor.rgba[:3]] + [1.0]
        tex.rgb1[:] = c1[:3]
        tex.rgb2[:] = c2[:3]
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "_scn_floor_tex"
        mat.texrepeat[:] = (8, 8)
    else:
        mat.rgba[:] = env.floor.rgba
    spec.worldbody.add_geom(
        name="_scn_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.1], material="_scn_floor", pos=[0, 0, floor_z],
    )
    if env.table:
        t = env.table
        spec.worldbody.add_geom(
            name="_scn_table", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[t.size[0] / 2, t.size[1] / 2, t.thickness / 2],
            pos=[t.pos[0], t.pos[1], t.height - t.thickness / 2],
            rgba=t.rgba,
        )
    for prop in env.props:
        _add_rigid_object(spec, prop, force_static=True, name_prefix="prop_")
    for lpos, ldir in ([0.4, -0.4, 1.2], [-0.3, 0.3, -1.0]), ([-0.4, 0.6, 1.0], [0.3, -0.5, -1.0]):
        light = spec.worldbody.add_light(pos=lpos, dir=ldir, castshadow=False)
        light.diffuse[:] = (env.light_diffuse,) * 3


_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


def _add_rigid_object(spec: mujoco.MjSpec, obj: ObjectSpec, force_static: bool = False,
                      name_prefix: str = "obj_") -> None:
    if obj.type not in _GEOM_TYPES:
        raise ValueError(f"object {obj.name!r}: unknown type {obj.type!r}")
    body = spec.worldbody.add_body(name=f"{name_prefix}{obj.name}", pos=obj.pos)
    kwargs = dict(
        name=f"{name_prefix}{obj.name}", type=_GEOM_TYPES[obj.type], rgba=obj.rgba,
        mass=obj.mass, condim=4, friction=[1.0, 0.02, 0.0005],
    )
    if obj.type == "mesh":
        if not obj.file:
            raise ValueError(f"object {obj.name!r}: type=mesh requires 'file'")
        mesh = spec.add_mesh(name=f"mesh_{obj.name}")
        mesh.file = str(Path(obj.file).expanduser().resolve())
        mesh.scale[:] = (obj.scale,) * 3
        kwargs["meshname"] = f"mesh_{obj.name}"
    else:
        kwargs["size"] = list(obj.size) + [0.0] * (3 - len(obj.size))
    body.add_geom(**kwargs)
    if not (obj.static or force_static):
        body.add_freejoint(name=f"free_{obj.name}")


def _add_cameras(spec: mujoco.MjSpec, cameras: list[CameraSpec]) -> None:
    existing = {c.name for c in spec.cameras}  # e.g. the arm's built-in wrist_cam
    for cam in cameras:
        if cam.name in existing:
            continue
        if cam.pos is None or cam.lookat is None:
            raise ValueError(
                f"camera {cam.name!r} needs pos+lookat (or must exist in the arm model)"
            )
        parent = spec.worldbody
        if cam.attach_to:
            parent = spec.body(cam.attach_to)
            if parent is None:
                raise ValueError(
                    f"camera {cam.name!r}: attach_to body {cam.attach_to!r} not found "
                    f"(try one of {ATTACHABLE_BODIES})"
                )
        pos = np.asarray(cam.pos, dtype=float)
        parent.add_camera(
            name=cam.name, pos=pos,
            quat=_camera_quat(pos, np.asarray(cam.lookat, dtype=float)), fovy=cam.fovy,
        )


def _cloth_xml(obj: ObjectSpec) -> str:
    p = obj.cloth_params()
    res = max(3, int(p["resolution"]))
    w, h = (obj.size + [obj.size[0]])[:2]
    spacing = (w / (res - 1), h / (res - 1))
    rgba = " ".join(str(c) for c in obj.rgba)
    pos = " ".join(str(c) for c in obj.pos)
    return f"""
    <flexcomp name="obj_{obj.name}" type="grid" count="{res} {res} 1"
              spacing="{spacing[0]} {spacing[1]} 0.01" pos="{pos}" dim="2"
              mass="{obj.mass}" rgba="{rgba}">
      <elasticity young="{p['young']}" poisson="{p['poisson']}"
                  thickness="{p['thickness']}" damping="{p['damping']}"/>
      <edge equality="false"/>
      <contact selfcollide="none"/>
    </flexcomp>"""


def _model_assets() -> dict[str, bytes]:
    adir = so101_mjcf_path().parent / "assets"
    return {f.name: f.read_bytes() for f in adir.iterdir() if f.is_file()}


def build_model(scenario: Scenario, mjcf_path: Path | None = None) -> mujoco.MjModel:
    """Compile arm MJCF + environment + objects + cameras into one mjModel.

    qpos layout: [arm(5), gripper, rigid-object free joints (7 each), cloth
    vertex DOFs (3 per vertex)] — the arm always comes first.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path or so101_mjcf_path()))
    _add_environment(spec, scenario.environment)
    for obj in scenario.rigid_objects:
        _add_rigid_object(spec, obj)
    _add_cameras(spec, scenario.cameras)
    model = spec.compile()

    cloths = scenario.cloth_objects
    if not cloths:
        return model
    # <flexcomp> is a parser-level construct with no MjSpec builder API, so
    # deformables go through an XML round-trip: serialize the compiled spec,
    # inject the flexcomp elements, recompile with in-memory assets.
    xml = spec.to_xml()
    cloth_block = "".join(_cloth_xml(o) for o in cloths)
    xml = xml.replace("</worldbody>", f"{cloth_block}\n  </worldbody>")
    return mujoco.MjModel.from_xml_string(xml, _model_assets())


# --------------------------------------------------------------------------
# per-reset randomization
# --------------------------------------------------------------------------


def _shift_cloth(model: mujoco.MjModel, data: mujoco.MjData, flex_name: str,
                 offset: np.ndarray) -> None:
    """Translate every vertex of a flex by offset (slide-joint qpos shift)."""
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, flex_name)
    if fid < 0:
        return
    adr, num = model.flex_vertadr[fid], model.flex_vertnum[fid]
    for vid in range(adr, adr + num):
        bid = model.flex_vertbodyid[vid]
        for j in range(model.body_jntadr[bid], model.body_jntadr[bid] + model.body_jntnum[bid]):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                axis = model.jnt_axis[j]
                data.qpos[model.jnt_qposadr[j]] += float(axis @ offset)


def randomize_object_qpos(
    scenario: Scenario, model: mujoco.MjModel, data: mujoco.MjData,
    rng: np.random.Generator,
) -> None:
    """Apply per-object pose randomization (call after mj_resetData)."""
    for obj in scenario.objects:
        if obj.static:
            continue
        noise = np.zeros(3)
        if obj.pos_noise:
            n = np.asarray(obj.pos_noise, dtype=float)
            noise = rng.uniform(-n, n)
        if obj.is_cloth:
            if obj.pos_noise:
                _shift_cloth(model, data, f"obj_{obj.name}", noise)
            continue
        jid = model.joint(f"free_{obj.name}").id
        adr = model.jnt_qposadr[jid]
        data.qpos[adr : adr + 3] = np.asarray(obj.pos, dtype=float) + noise
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if obj.yaw_range:
            yaw = rng.uniform(obj.yaw_range[0], obj.yaw_range[1])
            mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        data.qpos[adr + 3 : adr + 7] = quat
        vadr = model.jnt_dofadr[jid]
        data.qvel[vadr : vadr + 6] = 0.0
