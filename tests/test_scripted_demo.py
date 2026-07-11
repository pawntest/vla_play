import numpy as np

from so101_tool.demo.scripted import make_backend_and_pick
from so101_tool.scenario import ObjectSpec, Scenario


def _scenario():
    return Scenario(
        name="t",
        task="pick up the cube",
        objects=[
            ObjectSpec(
                name="cube",
                size=[0.015, 0.015, 0.015],
                pos=[0.25, 0.0, 0.02],
                pos_noise=[0.02, 0.02, 0.0],
                mass=0.03,
            )
        ],
        settle_steps=100,
        cameras=[],  # no rendering needed for this test
    )


def test_scripted_pick_lifts_cube_deterministically():
    backend, pick = make_backend_and_pick(_scenario(), seed=7, sample_hz=15)
    try:
        backend.reset(randomize=True)
        samples = []

        def cb(state, q_cmd, g_cmd):
            samples.append((state.q.copy(), q_cmd, g_cmd))

        ok = pick.run_episode(cb)
        assert ok, "seeded scripted pick should succeed"
        assert backend.object_pose("cube")[0][2] > 0.06
        assert len(samples) > 20  # sampled at ~15 Hz sim time throughout
        qs, q_cmds, g_cmds = samples[len(samples) // 2]
        assert qs.shape == (5,) and q_cmds.shape == (5,)
        assert 0.0 <= g_cmds <= 1.0
    finally:
        backend.disconnect()


def test_scripted_pick_multiple_episodes_reproducible():
    results1, results2 = [], []
    for results in (results1, results2):
        backend, pick = make_backend_and_pick(_scenario(), seed=123)
        try:
            for _ in range(3):
                backend.reset(randomize=True)
                results.append(
                    (pick.run_episode(), np.round(backend.object_pose("cube")[0], 6).tolist())
                )
        finally:
            backend.disconnect()
    assert results1 == results2  # fake clock -> fully deterministic


def test_unreachable_object_fails_cleanly():
    sc = _scenario()
    sc.objects[0].pos = [0.9, 0.0, 0.02]  # far outside the reach
    sc.objects[0].pos_noise = None
    backend, pick = make_backend_and_pick(sc, seed=0)
    try:
        backend.reset(randomize=True)
        assert pick.run_episode() is False
    finally:
        backend.disconnect()
