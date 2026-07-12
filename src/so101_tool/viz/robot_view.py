"""Live 3D view of the scene (arm + scenario objects) in viser, driven directly
by a MuJoCo model.

One viser frame per MuJoCo body that has visual geoms; meshes are built from
the mjModel mesh arrays (no STL re-reading, no URDF), primitive geoms via
trimesh.creation. sync()/sync_qpos() run FK on an owned MjData and write body
world transforms into the frame handles — viser and MuJoCo share the wxyz
quaternion convention.
"""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import viser

from ..config import TCP_SITE, gripper_fraction_to_rad
from ..kinematics import Kinematics

# group 2 = the arm MJCF `visual` class; group 0 = plain geoms (scenario objects)
_VISIBLE_GROUPS = (0, 2)


def _geom_trimesh(model: mujoco.MjModel, gid: int) -> trimesh.Trimesh | None:
    gtype = model.geom_type[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va : va + vn]
        faces = model.mesh_face[fa : fa + fn]  # face indices are local to the mesh
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * size)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    return None  # planes etc. — the grid covers the floor


def _geom_rgba(model: mujoco.MjModel, gid: int) -> np.ndarray:
    matid = model.geom_matid[gid]
    if matid >= 0:
        return model.mat_rgba[matid]
    return model.geom_rgba[gid]


class RobotView:
    def __init__(
        self,
        server: viser.ViserServer,
        source: Kinematics | mujoco.MjModel,
        root: str = "/robot",
    ):
        model = source.model if isinstance(source, Kinematics) else source
        self.model = model
        self._data = mujoco.MjData(model)
        self._tcp_sid = model.site(TCP_SITE).id

        self._handles = []
        self._handles.append(server.scene.add_grid("/grid", width=1.2, height=1.2, cell_size=0.1))

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:  # sensible default view
            client.camera.position = (0.65, -0.65, 0.45)
            client.camera.look_at = (0.2, 0.0, 0.15)

        self._frames: list[tuple[int, viser.FrameHandle]] = []
        # (body_name, mesh_handle) for every rigid geom mesh — arm bodies and
        # obj_* scenario bodies. Consumed by DirectDrag to bind on_drag per node.
        # Cloth flex meshes live under /cloth/ and are intentionally NOT here.
        self.mesh_nodes: list[tuple[str, viser.MeshHandle]] = []
        body_geoms: dict[int, list[int]] = {}
        for gid in range(model.ngeom):
            if model.geom_group[gid] not in _VISIBLE_GROUPS:
                continue
            if model.geom_bodyid[gid] == 0:  # world body (floor plane): grid covers it
                continue
            body_geoms.setdefault(int(model.geom_bodyid[gid]), []).append(gid)

        for bid, gids in sorted(body_geoms.items()):
            body_name = model.body(bid).name or f"body{bid}"
            frame = server.scene.add_frame(f"{root}/{body_name}", show_axes=False)
            self._frames.append((bid, frame))
            self._handles.append(frame)
            for gid in gids:
                mesh = _geom_trimesh(model, gid)
                if mesh is None:
                    continue
                rgba = _geom_rgba(model, gid)
                handle = server.scene.add_mesh_simple(
                    f"{root}/{body_name}/geom{gid}",
                    vertices=mesh.vertices.astype(np.float32),
                    faces=mesh.faces.astype(np.uint32),
                    color=tuple(rgba[:3]),
                    opacity=float(rgba[3]) if rgba[3] < 1.0 else None,
                    flat_shading=False,
                    position=model.geom_pos[gid],
                    wxyz=model.geom_quat[gid],
                )
                self._handles.append(handle)
                self.mesh_nodes.append((body_name, handle))

        self._tcp_frame = server.scene.add_frame("/tcp", axes_length=0.05, axes_radius=0.0025)
        self._handles.append(self._tcp_frame)

        # Deformables (cloth): flex meshes get their vertices re-sent per sync.
        self._server = server
        self._flexes: list[tuple[int, str, np.ndarray, tuple]] = []
        self._flex_handles: dict[str, viser.MeshHandle] = {}
        self._flex_tick = 0
        for fid in range(model.nflex):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, fid) or f"flex{fid}"
            ea, en = model.flex_elemadr[fid], model.flex_elemnum[fid]
            faces = model.flex_elem[ea * 3 : (ea + en) * 3].reshape(-1, 3)
            faces = faces - model.flex_vertadr[fid]  # local vertex indices
            rgba = model.flex_rgba[fid]
            self._flexes.append((fid, f"/cloth/{name}", faces.astype(np.uint32), tuple(rgba)))

        self.sync_qpos(self._data.qpos)

    def remove(self) -> None:
        """Remove every scene node this view created (App scene rebuild)."""
        for h in reversed(self._handles):  # children (meshes) before parent frames
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._frames.clear()
        self.remove_flex_handles()

    def sync_qpos(self, qpos: np.ndarray) -> None:
        """Update all body transforms (and the TCP axes) from a full model qpos."""
        self._data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self._data)
        data = self._data
        for bid, frame in self._frames:
            frame.position = data.xpos[bid]
            frame.wxyz = data.xquat[bid]
        if self._flexes:
            self._sync_flexes()
        wxyz = np.empty(4)
        mujoco.mju_mat2Quat(wxyz, data.site_xmat[self._tcp_sid].reshape(-1))
        self._tcp_frame.position = data.site_xpos[self._tcp_sid]
        self._tcp_frame.wxyz = wxyz

    def _sync_flexes(self) -> None:
        """Re-send cloth meshes (viser meshes are immutable, but cloth grids are
        tiny — ~100 vertices — so replacing the node at ~15 Hz is cheap)."""
        self._flex_tick += 1
        if self._flex_tick % 2:
            return
        mujoco.mj_flex(self.model, self._data)
        for fid, path, faces, rgba in self._flexes:
            va, vn = self.model.flex_vertadr[fid], self.model.flex_vertnum[fid]
            verts = self._data.flexvert_xpos[va : va + vn].astype(np.float32)
            handle = self._server.scene.add_mesh_simple(
                path, vertices=verts, faces=faces, color=rgba[:3],
                flat_shading=False, side="double",
            )
            self._flex_handles[path] = handle

    def remove_flex_handles(self) -> None:
        for h in self._flex_handles.values():
            try:
                h.remove()
            except Exception:
                pass
        self._flex_handles.clear()

    def sync(self, q: np.ndarray, gripper: float) -> None:
        """Arm-only convenience: joint values -> qpos (model must be the bare arm)."""
        qpos = self._data.qpos
        qpos[:5] = q
        qpos[5] = gripper_fraction_to_rad(gripper)
        self.sync_qpos(qpos)
