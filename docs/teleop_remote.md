# Remote teleoperation (SSH / Codespaces / Brev)

日本語版: [docs/ja/teleop_remote.md](ja/teleop_remote.md)

Run so101-tool on a remote container (GitHub Codespaces, NVIDIA Brev, any SSH
box) and drive it with the **leader arm plugged into your laptop** — for live
control of the remote sim, dataset recording, or a follower arm attached to
the remote machine.

```
laptop (leader arm)                      remote container (so101-tool)
so101-tool teleop-client  ── SSH tunnel ──▶  TeleopReceiver 127.0.0.1:8765
        │ 50 Hz JSON joint stream                  │ TELEOP mode
   SO-101 leader                        sim / recording / follower
```

## Quick start

Remote (Codespace/Brev):

```bash
so101-tool run --scenario examples/pick_cube.yaml --teleop
# prints:  teleop receiver: 127.0.0.1:8765  + a session token
```

Laptop (leader arm attached, `pip install "so101-tool[real]"`):

```bash
ssh -L 8765:localhost:8765 <remote>      # or use the Codespaces port-forward UI
so101-tool teleop-client --connect localhost:8765 \
    --token <printed-token> --port /dev/ttyACM0 --robot-id my_leader
```

In the browser UI, the **Remote teleop** panel shows 🟢 streaming — press
**Enable TELEOP mode**. Recording works as usual (Record dataset panel), so
you can collect real-teleop demonstrations into a remote LeRobotDataset.

No hardware? Test the link end to end with `--source sine` (synthetic
trajectory) from any machine.

## Security design

- The receiver **only binds 127.0.0.1** (a non-loopback bind is rejected in
  code). Nothing is ever exposed to the network; the only path in is your SSH
  tunnel, so transport encryption + authentication are SSH's.
- A **random per-session token** is required in the handshake
  (`secrets.compare_digest`); other local users on a shared container cannot
  inject motion. Override with `--teleop-token`/`SO101_TELEOP_TOKEN` if you
  need a stable token.
- Fixed JSON schema, 4 KiB line cap, NaN/shape validation, joint-limit
  clamping — and every target still passes the control loop's **safety filter**
  (velocity clamp, limits). If the stream stalls for >0.5 s the arm **holds**
  instead of drifting; the e-stop always wins.
- On Codespaces, forwarded ports default to *private* (your GitHub account
  only) — keep them private; you never need to make them public.

## Real↔MuJoCo link (`--link`)

Beyond one-way teleop, `--link` couples the real arm and the MuJoCo sim in
**both directions**, switchable at runtime from the 🔗 dropdown in the UI
header:

| mode | meaning |
| --- | --- |
| `to_sim` | 実機→MuJoCo — the real arm is the source of truth; the sim arm follows it (and physically interacts with scene objects). Move the real arm by hand (torque off) and watch the sim mirror it. |
| `to_real` | MuJoCo→実機 — GUI / NL / policy commands drive the sim, and the same targets are shadowed to the real arm. A real-link hiccup never stops the sim. |
| `both` | 実機↔MuJoCo — commands go to the real arm and the sim always follows the real measured joints, so hand-moving the arm AND commanding it both stay in sync. |

The "real" side is chosen automatically:

- **Local serial arm** — `so101-tool run --backend real --link both --scenario …`
- **Over SSH (no serial on the remote box)** — just add `--link` and the
  receiver starts automatically; on your laptop run the client with
  `--source follower` so your local arm streams its joints up *and applies
  the target frames sent back* (full duplex on the same tunneled socket):

```bash
remote$ so101-tool run --scenario examples/pick_cube.yaml --link both
laptop$ ssh -L 8765:localhost:8765 <remote>
laptop$ so101-tool teleop-client --connect localhost:8765 \
            --token <printed-token> --source follower --port /dev/ttyACM0
```

The security posture is unchanged: same 127.0.0.1-only socket, same token,
and the downstream `target_q` frames are just the receiver pushing back on
the already-authenticated connection — no new port, no new attack surface.

## Notes

- Works with the physics sim (`--scenario`), the bare-arm sim, and
  `--backend real` (leader on your desk, follower on the remote machine's
  bus — the usual lerobot wiring but split across the network).
- Latency: a continental SSH round-trip (~30-80 ms) is fine for teleop at
  50 Hz; the receiver always uses the latest frame (no queue buildup).
