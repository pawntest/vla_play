import mujoco
import numpy as np
import pytest

from so101_tool.scenario import (
    CameraSpec,
    ObjectSpec,
    Scenario,
    build_model,
    load_scenario,
    randomize_object_qpos,
)

SCENARIO_YAML = """
name: test_scene
task: "pick up the cube"
objects:
  - name: cube
    type: box
    size: [0.015, 0.015, 0.015]
    pos: [0.25, 0.0, 0.02]
    pos_noise: [0.03, 0.03, 0.0]
    yaw_range: [-1.0, 1.0]
  - name: pillar
    type: cylinder
    size: [0.02, 0.04]
    pos: [0.15, -0.15, 0.04]
    static: true
cameras:
  - {name: front, pos: [0.6, 0.0, 0.35], lookat: [0.2, 0.0, 0.1]}
  - {name: wrist_cam}
"""


def test_load_and_build(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text(SCENARIO_YAML)
    sc = load_scenario(path)
    assert sc.task == "pick up the cube"
    assert [o.name for o in sc.objects] == ["cube", "pillar"]

    m = build_model(sc)
    # 6 arm qpos + 7 for the cube free joint; the static pillar adds none
    assert m.nq == 13
    cams = {m.camera(i).name for i in range(m.ncam)}
    assert {"front", "wrist_cam"} <= cams
    assert m.body("obj_cube") is not None
    assert m.body("obj_pillar") is not None


def test_randomization_bounds():
    sc = Scenario(
        objects=[ObjectSpec(name="c", pos=[0.25, 0.0, 0.02], pos_noise=[0.03, 0.03, 0.0])]
    )
    m = build_model(sc)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    xs = []
    for _ in range(20):
        mujoco.mj_resetData(m, d)
        randomize_object_qpos(sc, m, d, rng)
        adr = m.jnt_qposadr[m.joint("free_c").id]
        pos = d.qpos[adr : adr + 3]
        assert abs(pos[0] - 0.25) <= 0.03 + 1e-9
        assert abs(pos[1]) <= 0.03 + 1e-9
        xs.append(pos[0])
    assert np.std(xs) > 0  # actually randomized


def test_unknown_camera_requires_pos():
    sc = Scenario(cameras=[CameraSpec(name="nope")])
    with pytest.raises(ValueError, match="pos"):
        build_model(sc)


def test_mesh_object_requires_file():
    sc = Scenario(objects=[ObjectSpec(name="m", type="mesh")])
    with pytest.raises(ValueError, match="file"):
        build_model(sc)


def test_environment_cloth_and_attached_camera():
    from so101_tool.scenario import CameraSpec, EnvironmentSpec, FloorSpec, TableSpec

    sc = Scenario(
        name="v2",
        environment=EnvironmentSpec(
            floor=FloorSpec(checker=True),
            table=TableSpec(),
            props=[ObjectSpec(name="wall", size=[0.01, 0.3, 0.15], pos=[-0.15, 0, 0.15])],
        ),
        objects=[
            ObjectSpec(name="cube", pos=[0.25, 0, 0.02]),
            ObjectSpec(name="towel", type="cloth", size=[0.1, 0.1],
                       pos=[0.2, -0.1, 0.04], mass=0.03, pos_noise=[0.02, 0.02, 0]),
        ],
        cameras=[
            CameraSpec(name="front", pos=[0.5, -0.3, 0.3], lookat=[0.25, 0, 0.05]),
            CameraSpec(name="grip", attach_to="gripper", pos=[0, -0.07, 0.01],
                       lookat=[0.02, 0, -0.22]),
        ],
    )
    m = build_model(sc)
    assert m.nflex == 1
    assert m.geom("_scn_table") is not None
    assert m.geom("prop_wall") is not None
    assert {m.camera(i).name for i in range(m.ncam)} >= {"front", "grip"}
    # attached camera lives on the gripper body
    cam_body = m.body(m.cam_bodyid[m.camera("grip").id]).name
    assert cam_body == "gripper"

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    before = d.qpos.copy()
    randomize_object_qpos(sc, m, d, np.random.default_rng(3))
    assert not np.allclose(before, d.qpos)  # cube AND cloth vertices moved
    for _ in range(300):
        mujoco.mj_step(m, d)
    # cloth settled on the tabletop, not exploded
    assert -0.02 < d.flexvert_xpos[:, 2].min() < 0.05


def test_v2_yaml_round_trip(tmp_path):
    from so101_tool.scenario import load_scenario, save_scenario

    sc = load_scenario("examples/tabletop_cloth.yaml")
    sc2 = load_scenario(save_scenario(sc, tmp_path / "rt.yaml"))
    assert sc2.environment.table is not None and sc2.environment.floor.checker
    assert [o.name for o in sc2.objects] == [o.name for o in sc.objects]
    assert sc2.cloth_objects[0].cloth_params()["resolution"] == 9
    assert any(c.attach_to == "gripper" for c in sc2.cameras)
