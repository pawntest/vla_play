"""Live 3D view of the arm in viser, driven directly by the MuJoCo model.

One viser frame per MuJoCo body that has visual geoms; meshes are built from
the mjModel mesh arrays (no STL re-reading, no URDF). sync() runs FK on the
owned Kinematics instance and writes body world transforms into the frame
handles — viser and MuJoCo share the wxyz quaternion convention.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser

from ..config import TCP_SITE
from ..kinematics import Kinematics

_VISUAL_GROUP = 2  # the `visual` default class in the vendored MJCF


def _geom_mesh(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh:
    mid = model.geom_dataid[gid]
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    verts = model.mesh_vert[va : va + vn]
    faces = model.mesh_face[fa : fa + fn]  # face indices are local to the mesh
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _geom_rgba(model: mujoco.MjModel, gid: int) -> np.ndarray:
    matid = model.geom_matid[gid]
    if matid >= 0:
        return model.mat_rgba[matid]
    return model.geom_rgba[gid]


class RobotView:
    def __init__(self, server: viser.ViserServer, kinematics: Kinematics, root: str = "/robot"):
        self._kin = kinematics
        model = kinematics.model
        self._tcp_sid = model.site(TCP_SITE).id

        server.scene.add_grid("/grid", width=1.2, height=1.2, cell_size=0.1)

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:  # sensible default view
            client.camera.position = (0.65, -0.65, 0.45)
            client.camera.look_at = (0.2, 0.0, 0.15)
        self._frames: list[tuple[int, viser.FrameHandle]] = []
        body_geoms: dict[int, list[int]] = {}
        for gid in range(model.ngeom):
            if model.geom_group[gid] != _VISUAL_GROUP:
                continue
            body_geoms.setdefault(int(model.geom_bodyid[gid]), []).append(gid)

        for bid, gids in sorted(body_geoms.items()):
            body_name = model.body(bid).name or f"body{bid}"
            frame = server.scene.add_frame(f"{root}/{body_name}", show_axes=False)
            self._frames.append((bid, frame))
            for gid in gids:
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue  # the SO-101 visual class is mesh-only
                mesh = _geom_mesh(model, gid)
                rgba = _geom_rgba(model, gid)
                server.scene.add_mesh_simple(
                    f"{root}/{body_name}/geom{gid}",
                    vertices=mesh.vertices.astype(np.float32),
                    faces=mesh.faces.astype(np.uint32),
                    color=tuple(rgba[:3]),
                    opacity=float(rgba[3]) if rgba[3] < 1.0 else None,
                    flat_shading=False,
                    position=model.geom_pos[gid],
                    wxyz=model.geom_quat[gid],
                )

        self._tcp_frame = server.scene.add_frame("/tcp", axes_length=0.05, axes_radius=0.0025)
        self.sync(np.zeros(5), 0.0)

    def sync(self, q: np.ndarray, gripper: float) -> None:
        """Update all body transforms (and the TCP axes) from joint values."""
        self._kin.set_qpos(q, gripper)
        data = self._kin.data
        for bid, frame in self._frames:
            frame.position = data.xpos[bid]
            frame.wxyz = data.xquat[bid]
        wxyz = np.empty(4)
        mujoco.mju_mat2Quat(wxyz, data.site_xmat[self._tcp_sid].reshape(-1))
        self._tcp_frame.position = data.site_xpos[self._tcp_sid]
        self._tcp_frame.wxyz = wxyz
