"""
Blender helper script: export selected humanoid meshes to OBJ files.

Usage inside Blender:
1) Open the humanoid .blend file.
2) Select only mesh objects you want to export.
3) Run this script from Blender's Scripting tab.
4) OBJ files are written to:
   my-env/sim/mjcf/assets/humanoid_meshes/

The exported filenames are based on body-part names expected by the MJCF docs.
"""

from __future__ import annotations

from pathlib import Path
import re

import bpy


EXPORT_DIR = Path(bpy.path.abspath("//")) / "sim" / "mjcf" / "assets" / "humanoid_meshes"

# Optional explicit mapping from Blender object names to target mesh slot names.
OBJECT_TO_PART = {
    "Pelvis": "pelvis",
    "Torso": "torso",
    "Head": "head",
    "LeftArm": "left_arm",
    "RightArm": "right_arm",
    "LeftLeg": "left_leg",
    "RightLeg": "right_leg",
    "LeftShoe": "left_shoe",
    "RightShoe": "right_shoe",
}


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower()


def _part_name(obj_name: str) -> str:
    if obj_name in OBJECT_TO_PART:
        return OBJECT_TO_PART[obj_name]
    return _slugify(obj_name)


def _deselect_all() -> None:
    for obj in bpy.context.selected_objects:
        obj.select_set(False)


def _export_obj(obj: bpy.types.Object, out_path: Path) -> None:
    _deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Blender 4.x path:
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(out_path),
            export_selected_objects=True,
            export_normals=True,
            export_uv=True,
            export_materials=False,
        )
        return

    # Blender 3.x fallback:
    bpy.ops.export_scene.obj(
        filepath=str(out_path),
        use_selection=True,
        use_normals=True,
        use_uvs=True,
        use_materials=False,
        keep_vertex_order=True,
    )


def main() -> None:
    selected_meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not selected_meshes:
        raise RuntimeError("Select at least one mesh object before running export.")

    print(f"Exporting {len(selected_meshes)} selected mesh objects to: {EXPORT_DIR}")
    for obj in selected_meshes:
        part = _part_name(obj.name)
        out_path = EXPORT_DIR / f"{part}.obj"
        _export_obj(obj, out_path)
        print(f"  - {obj.name} -> {out_path.name}")

    print("Done. Next: wire these OBJ names into MJCF optional mesh slots.")


if __name__ == "__main__":
    main()
