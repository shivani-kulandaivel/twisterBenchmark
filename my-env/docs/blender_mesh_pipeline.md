# Blender Mesh Pipeline (Visual-Only)

This project now uses a **visual-vs-collision** layout:

- Physics and contacts stay on primitive geoms (capsules/boxes/spheres), grouped as collision geoms (`group="3"`).
- Render-only geoms are in visual group (`group="1"`) and use `contype="0" conaffinity="0"` where applicable.
- Existing clay-style primitive visuals remain the fallback and should always work.

## Export Blender Meshes

1. Open your humanoid `.blend`.
2. Select only the body-part meshes you want to export.
3. Run `my-env/tools/export_blender_meshes.py` from Blender's Scripting tab.
4. The script exports OBJ files to:
   - `my-env/sim/mjcf/assets/humanoid_meshes/`

No Blender dependency is required at runtime; export happens offline.

## Naming Conventions

Use these OBJ names to match the optional MJCF mesh slots:

- `pelvis.obj`
- `torso.obj`
- `head.obj`
- `left_arm.obj`
- `right_arm.obj`
- `left_leg.obj`
- `right_leg.obj`
- `left_shoe.obj`
- `right_shoe.obj`

The MJCF currently keeps these mesh assets commented out so model loading does not fail when files are absent.

## Wiring Meshes Into MJCF

In `my-env/sim/mjcf/humanoid_twister.xml`:

1. Uncomment the `<mesh .../>` entries under `<asset>` for files you exported.
2. Add visual-only `geom type="mesh"` geoms on matching bodies (or switch fallback visual geoms), for example:

```xml
<geom class="mesh_visual_geom" mesh="mesh_torso" material="mesh_cloth"/>
```

3. Keep primitive collision geoms in place (`group="3"`), unchanged for physics.

## Viewer Layer Controls

`viewer.py` now defaults to visual-only rendering:

- visual group (`1`): ON
- collision group (`3`): OFF

Use `--show-collision` to show collision geoms for debugging:

```bash
mjpython viewer.py --spins 1 --show-collision
```
