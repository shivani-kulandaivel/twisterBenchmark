"""
Blender script: build a smoother, more realistic Twister humanoid and export OBJ parts.

Run inside Blender (Scripting tab):
    Open this file and click Run Script

Requires Blender 3.x or 4.x. Exports to:
    my-env/sim/mjcf/assets/humanoid_meshes/

After export, meshes are picked up automatically by humanoid_twister.xml.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import bmesh


# Resolve export dir relative to this .blend or script location.
SCRIPT_DIR = Path(bpy.path.abspath("//")) if bpy.data.filepath else Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parent / "sim" / "mjcf" / "assets" / "humanoid_meshes"
if not (SCRIPT_DIR / "sim").exists():
    EXPORT_DIR = Path(__file__).resolve().parents[1] / "sim" / "mjcf" / "assets" / "humanoid_meshes"


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _smooth_capsule(name: str, p0, p1, radius: float, segments: int = 32, rings: int = 12):
    """Create a smooth capsule mesh between p0 and p1."""
    dx = [p1[i] - p0[i] for i in range(3)]
    length = math.sqrt(sum(v * v for v in dx))
    if length < 1e-6:
        dx = [0.0, 0.0, 1.0]
        length = 1.0
    dx = [v / length for v in dx]

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments,
        radius=radius,
        depth=length,
        location=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2),
    )
    obj = bpy.context.active_object
    obj.name = name

    # Align cylinder Z axis to segment direction (MuJoCo body local frame).
    z = (0.0, 0.0, 1.0)
    axis = (
        z[1] * dx[2] - z[2] * dx[1],
        z[2] * dx[0] - z[0] * dx[2],
        z[0] * dx[1] - z[1] * dx[0],
    )
    angle = math.acos(max(-1.0, min(1.0, z[0] * dx[0] + z[1] * dx[1] + z[2] * dx[2])))
    if math.sqrt(sum(a * a for a in axis)) > 1e-6:
        obj.rotation_mode = "AXIS_ANGLE"
        obj.rotation_axis_angle = (angle, axis[0], axis[1], axis[2])

    # Add sphere caps via subdivision for a clay/stylized look.
    bpy.ops.object.modifier_add(type="SUBSURF")
    obj.modifiers["Subdivision"].levels = 2
    obj.modifiers["Subdivision"].render_levels = 2
    bpy.ops.object.shade_smooth()
    return obj


def _smooth_box(name: str, center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(cx, cy, cz))
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (sx * 2, sy * 2, sz * 2)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.modifier_add(type="BEVEL")
    obj.modifiers["Bevel"].width = min(sx, sy, sz) * 0.15
    obj.modifiers["Bevel"].segments = 3
    bpy.ops.object.shade_smooth()
    return obj


def _smooth_sphere(name: str, center, radius: float, segments: int = 32, rings: int = 16):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=center,
    )
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.modifier_add(type="SUBSURF")
    obj.modifiers["Subdivision"].levels = 1
    bpy.ops.object.shade_smooth()
    return obj


def _export_selected_meshes(objects: list[bpy.types.Object]) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        out = EXPORT_DIR / f"{obj.name}.obj"
        if hasattr(bpy.ops.wm, "obj_export"):
            bpy.ops.wm.obj_export(filepath=str(out), export_selected_objects=True)
        else:
            bpy.ops.export_scene.obj(filepath=str(out), use_selection=True)
        print(f"Exported {out}")


def main() -> None:
    _clear_scene()

    parts = [
        _smooth_capsule("pelvis", (0, 0, -0.02), (0, 0, 0.06), 0.085),
        _smooth_box("pelvis_shorts", (0, 0, 0.0), (0.11, 0.09, 0.06)),
        _smooth_capsule("lumbar", (0, 0, 0.0), (0, 0, 0.14), 0.10),
        _smooth_capsule("torso", (0, 0, 0.01), (0, 0, 0.17), 0.105),
        _smooth_capsule("shirt", (0, 0, 0.02), (0, 0, 0.165), 0.118),
        _smooth_sphere("head", (0, 0, 0.0), 0.102),
        _smooth_sphere("hair", (0, -0.005, 0.055), 0.108),
        _smooth_capsule("left_upper_arm", (0, 0, 0), (0.26, 0, 0), 0.048),
        _smooth_capsule("left_forearm", (0, 0, 0), (0.24, 0, 0), 0.042),
        _smooth_capsule("right_upper_arm", (0, 0, 0), (-0.26, 0, 0), 0.048),
        _smooth_capsule("right_forearm", (0, 0, 0), (-0.24, 0, 0), 0.042),
        _smooth_capsule("left_thigh", (0, 0, 0), (0, 0, -0.34), 0.058),
        _smooth_capsule("left_shin", (0, 0, 0), (0, 0, -0.32), 0.052),
        _smooth_capsule("right_thigh", (0, 0, 0), (0, 0, -0.34), 0.058),
        _smooth_capsule("right_shin", (0, 0, 0), (0, 0, -0.32), 0.052),
        _smooth_capsule("left_pants_thigh", (0, 0, -0.02), (0, 0, -0.31), 0.065),
        _smooth_capsule("left_pants_shin", (0, 0, -0.01), (0, 0, -0.30), 0.058),
        _smooth_capsule("right_pants_thigh", (0, 0, -0.02), (0, 0, -0.31), 0.065),
        _smooth_capsule("right_pants_shin", (0, 0, -0.01), (0, 0, -0.30), 0.058),
        _smooth_box("left_shoe", (0.03, 0, -0.025), (0.11, 0.12, 0.025)),
        _smooth_box("right_shoe", (-0.03, 0, -0.025), (0.11, 0.12, 0.025)),
    ]

    _export_selected_meshes(parts)
    print(f"Done — {len(parts)} parts exported to {EXPORT_DIR}")


if __name__ == "__main__":
    main()
