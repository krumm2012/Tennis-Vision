"""Export video-bound MHR animation and exact corner UVs to self-contained Blender files.

Run with Blender --background --python this_file -- --protocol ... --library ...
--layout ... --texture ... --output ... . No external frame-change script is needed
to play a saved file: every source frame is an absolute, linearly keyed shape.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import bpy
import numpy as np
from mathutils import Vector


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def converted(vertices):
    # Camera x right, y down, z away -> Blender x right, y away, z up.
    return np.ascontiguousarray(vertices[..., [0, 2, 1]] * [1, 1, -1], dtype=np.float32)


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def material(texture):
    image = bpy.data.images.load(str(texture), check_existing=True)
    image.name = 'Video_Color_4K_observed_RGBA'
    image.colorspace_settings.name = 'sRGB'
    image.pack()
    mat = bpy.data.materials.new('Video_color__baked_lighting__unknown_gray')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = .82
    bsdf.inputs['Specular IOR Level'].default_value = .18
    tex = nodes.new('ShaderNodeTexImage'); tex.image = image
    tex.interpolation = 'Linear'; tex.extension = 'EXTEND'
    tex.label = 'Native video samples; alpha = observed support'
    tex.location = (-600, 100)
    uv = nodes.new('ShaderNodeUVMap'); uv.uv_map = 'MHR_exact_corner_UV'
    uv.location = (-800, 100)
    links.new(uv.outputs['UV'], tex.inputs['Vector'])
    mix = nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MIX'
    mix.inputs[1].default_value = (.18, .22, .25, 1)
    mix.location = (-250, 100)
    links.new(tex.outputs['Alpha'], mix.inputs[0])
    links.new(tex.outputs['Color'], mix.inputs[2])
    links.new(mix.outputs[0], bsdf.inputs['Base Color'])
    nodes.active = tex
    # Kept available in the file for material override, not assigned by default.
    diagnostic = bpy.data.materials.new('UV_support__green_observed__magenta_unknown')
    diagnostic.use_nodes = True; diagnostic.use_fake_user = True
    dn, dl = diagnostic.node_tree.nodes, diagnostic.node_tree.links
    dn.clear(); out = dn.new('ShaderNodeOutputMaterial'); emit = dn.new('ShaderNodeEmission')
    dt = dn.new('ShaderNodeTexImage'); dt.image = image
    dm = dn.new('ShaderNodeMixRGB'); dm.inputs[1].default_value = (1, 0, .35, 1)
    dm.inputs[2].default_value = (.02, .7, .25, 1)
    dl.new(dt.outputs['Alpha'], dm.inputs[0]); dl.new(dm.outputs[0], emit.inputs[0])
    dl.new(emit.outputs[0], out.inputs['Surface'])
    return mat


def mesh_object(name, vertices, faces, uv, uv_faces, mat):
    mesh = bpy.data.meshes.new(name + '_mesh')
    mesh.from_pydata(vertices.tolist(), [], faces.tolist()); mesh.update()
    layer = mesh.uv_layers.new(name='MHR_exact_corner_UV')
    layer.data.foreach_set('uv', np.ascontiguousarray(uv[uv_faces].ravel(), dtype=np.float32))
    mesh.polygons.foreach_set('use_smooth', np.ones(len(faces), dtype=bool))
    mesh.materials.append(mat)
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    return obj


def studio(body, coordinates, fps):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'; scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1080; scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = round(fps); scene.render.fps_base = round(fps) / fps
    scene.render.image_settings.file_format = 'PNG'; scene.render.image_settings.color_mode = 'RGBA'
    scene.view_settings.view_transform = 'AgX'
    scene.unit_settings.system = 'METRIC'; scene.unit_settings.scale_length = 1
    if scene.world is None:
        scene.world = bpy.data.worlds.new('Studio_world')
    scene.world.color = (.16, .16, .16)
    lo, hi = coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))
    center = (lo + hi) / 2
    floor = float(lo[2]) - .018
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, floor))
    ground = bpy.context.object; ground.name = 'Studio_floor'
    gm = bpy.data.materials.new('Studio_neutral'); gm.diffuse_color = (.14, .17, .21, 1)
    gm.use_nodes = True; gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.14, .17, .21, 1)
    gm.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .9
    ground.data.materials.append(gm)
    extent = float(np.linalg.norm(hi - lo))
    cameras = {}
    for name, direction in [('Front', (0, -1, .2)), ('Back', (0, 1, .2)), ('Oblique', (1, -1.5, .25))]:
        data = bpy.data.cameras.new(name); cam = bpy.data.objects.new(name, data)
        scene.collection.objects.link(cam)
        cam.location = center + np.array(direction) * 8
        data.type = 'ORTHO'; data.ortho_scale = extent * 1.12
        aim(cam, center); cameras[name] = cam
    for name, location, power, size in [('Key', (-3, -4, 5), 450, 4), ('Fill', (3, -2, 3), 220, 4), ('Rim', (1, 3, 4), 400, 3)]:
        light = bpy.data.lights.new(name, 'AREA'); light.energy = power; light.shape = 'DISK'; light.size = size
        obj = bpy.data.objects.new(name, light); scene.collection.objects.link(obj)
        obj.location = location; aim(obj, center)
    scene.camera = cameras['Oblique']
    bpy.ops.object.select_all(action='DESELECT'); body.select_set(True)
    bpy.context.view_layer.objects.active = body
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.spaces.active.region_3d.view_perspective = 'CAMERA'
                area.spaces.active.shading.type = 'MATERIAL'
    return cameras


def animation(body, coordinates):
    for i, vertices in enumerate(coordinates):
        key = body.shape_key_add(name=f'frame_{i + 1:04d}', from_mix=False)
        key.data.foreach_set('co', vertices.ravel()); key.interpolation = 'KEY_LINEAR'
    keys = body.data.shape_keys; keys.use_relative = False
    for i, key in enumerate(keys.key_blocks):
        keys.eval_time = key.frame
        keys.keyframe_insert(data_path='eval_time', frame=i + 1)
    action = keys.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for point in curve.keyframe_points:
                        point.interpolation = 'LINEAR'
    bpy.context.scene.frame_start = 1; bpy.context.scene.frame_end = len(coordinates)


def check(body, coordinates, uv, uv_faces):
    checks = []
    for frame in sorted({1, max(1, len(coordinates) // 2), len(coordinates)}):
        bpy.context.scene.frame_set(frame)
        obj = body.evaluated_get(bpy.context.evaluated_depsgraph_get()); mesh = obj.to_mesh()
        actual = np.empty(coordinates.shape[1] * 3, dtype=np.float32)
        mesh.vertices.foreach_get('co', actual); obj.to_mesh_clear()
        error = float(np.max(np.abs(actual.reshape(-1, 3) - coordinates[frame - 1])))
        if error > 2e-6: raise ValueError(f'Animation export mismatch: {frame}: {error}')
        checks.append({'blender_frame': frame, 'max_coordinate_error_m': error})
    actual_uv = np.empty(uv_faces.size * 2, dtype=np.float32)
    body.data.uv_layers.active.data.foreach_get('uv', actual_uv)
    error = float(np.max(np.abs(actual_uv - uv[uv_faces].ravel())))
    if error > 1e-7: raise ValueError('Corner UV mismatch')
    return {'animation_frames_checked': checks, 'max_corner_uv_error': error}


def main(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    rig = np.load(args.layout, allow_pickle=False)
    faces, uv, uv_faces = rig['faces'], rig['uv'], rig['uv_faces']
    report = json.loads((args.texture.parent / 'texture_report.json').read_text())
    if report['layout_sha256'] != sha(args.layout) or report['artifact_sha256']['body_texture_rgba.png'] != sha(args.texture):
        raise ValueError('Texture provenance mismatch')
    if args.rest_only:
        dest = output / 'UV_workbench.blend'
        if dest.exists(): raise ValueError(f'Output exists: {dest}')
        bpy.ops.wm.read_factory_settings(use_empty=True)
        coords = converted(rig['rest_vertices'])[None]
        body = mesh_object('MHR_template_UV_workbench', coords[0], faces, uv, uv_faces, material(args.texture.resolve()))
        body['geometry_note'] = 'Canonical template rest pose, not a separately reconstructed scan of the subject'
        studio(body, coords, 25)
        image = bpy.data.images.get('Video_Color_4K_observed_RGBA')
        for screen in bpy.data.screens:
            for area in screen.areas:
                if area.type == 'IMAGE_EDITOR': area.spaces.active.image = image
        bpy.ops.wm.save_as_mainfile(filepath=str(dest), compress=True)
        print('EXPORTED UV_workbench', flush=True)
        return []
    protocol = json.loads(args.protocol.read_text()); results = []
    for entry in protocol['entries']:
        if args.clip and entry['name'] != args.clip: continue
        name = entry['name']; dest = output / (name + '.blend')
        if dest.exists(): raise ValueError(f'Output exists: {dest}')
        folder = args.library / urlsplit(entry['viewer']).path.split('/')[2] / 'result'
        meta = json.loads((folder / 'mesh_meta.json').read_text())
        if not np.array_equal(np.fromfile(folder / 'mesh_faces.bin', '<u4').reshape(-1, 3), faces):
            raise ValueError('Mesh/UV topology mismatch')
        if meta['video_sha256'] not in {r['video_sha256'] for r in report['inputs']}:
            raise ValueError('Video is not part of the same-outfit texture group')
        path = folder / 'mesh_refined.bin'
        # Preserve the currently delivered mesh: do not refit or change vertex order.
        source = np.fromfile(path, '<f4').reshape(meta['frames'], meta['vertices'], 3)
        if not np.isfinite(source).all(): raise ValueError('Nonfinite mesh')
        coords = converted(source)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        body = mesh_object('Person_' + name, coords[0], faces, uv, uv_faces, material(args.texture.resolve()))
        body['video_sha256'] = meta['video_sha256']; body['mesh_sha256'] = sha(path)
        body['coordinate_transform'] = '(x,y,z) camera -> (x,z,-y) Blender; metres, root-relative'
        body['texture_limit'] = 'Observed video color includes capture lighting; not calibrated albedo. Unknown texels gray.'
        animation(body, coords); cameras = studio(body, coords, meta['fps'])
        validation = check(body, coords, uv, uv_faces)
        scene = bpy.context.scene; scene.frame_set(min(126, meta['frames']))
        note = bpy.data.texts.new('READ_ME')
        note.write('Person animation from the matching video. Frame 1 = video frame 0.\n'
                   'Every source frame is stored in absolute shape keys; no Python playback handler.\n'
                   'UVs use exact MHR corner indices; all images packed.\n'
                   'Front / Back / Oblique cameras, Cycles 96 samples with denoising.\n'
                   'Video colors contain capture lighting. No invented normal/displacement maps.\n'
                   'Use UV_support material for green observation / magenta missing areas.\n'
                   'Root-relative motion, estimated metric scale; no armature or calibrated world trajectory.\n')
        bpy.ops.wm.save_as_mainfile(filepath=str(dest), compress=True)
        # Reopen the deliverable and evaluate its saved animation and UV data.
        bpy.ops.wm.open_mainfile(filepath=str(dest))
        body = bpy.data.objects['Person_' + name]
        validation['reopened'] = check(body, coords, uv, uv_faces)
        if any(not image.packed_file for image in bpy.data.images if image.source == 'FILE'):
            raise ValueError('Unpacked texture dependency')
        scene = bpy.context.scene; scene.frame_set(min(126, meta['frames']))
        cameras = {view: bpy.data.objects[view] for view in ['Front', 'Back', 'Oblique']}
        scene.cycles.samples = 16; scene.render.resolution_x = 640; scene.render.resolution_y = 800
        previews = []
        for view in (['Front', 'Back', 'Oblique'] if name == '48.43' else ['Oblique']):
            scene.camera = cameras[view]
            preview = output / f'{name}_{view.lower()}.png'; scene.render.filepath = str(preview)
            bpy.ops.render.render(write_still=True); previews.append(preview.name)
        results.append({'clip': name, 'blend': dest.name, 'blend_sha256': sha(dest), 'frames': meta['frames'],
                        'fps': meta['fps'], 'vertices': meta['vertices'], 'triangles': len(faces),
                        'source_mesh': str(path.resolve()), 'source_mesh_sha256': sha(path),
                        'video_sha256': meta['video_sha256'], 'texture_sha256': sha(args.texture),
                        'validation': validation, 'previews': previews})
        (output / f'{name}_export.json').write_text(json.dumps(results[-1], indent=2) + '\n')
        print('EXPORTED', name, flush=True)
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['protocol', 'library', 'layout', 'texture', 'output']:
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--clip')
    p.add_argument('--rest-only', action='store_true')
    main(p.parse_args(sys.argv[sys.argv.index('--') + 1:]))
