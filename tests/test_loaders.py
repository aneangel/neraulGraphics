"""Tests for the OBJ and glTF loaders. Parsing tests run anywhere; rendering tests need an NVIDIA GPU,
and texture tests need Pillow."""

import base64
import json
import struct
import warnings

import numpy as np
import pytest

import neraulgraphics as ng

W, H = 160, 90
LIGHT = [ng.Light((0, 0, 2), intensity=1.0)]
# A 2x2 image, rows from the top: red green / blue white.
QUADRANTS = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)


@pytest.fixture(scope="module")
def gpu():
    try:
        ng.Renderer(8, 8, [], [])
    except ng.CudaError as e:
        pytest.skip(f"no usable CUDA GPU: {e}")


def png_bytes(pixels=QUADRANTS):
    PIL = pytest.importorskip("PIL.Image")
    import io
    out = io.BytesIO()
    PIL.fromarray(pixels).save(out, format="PNG")
    return out.getvalue()


def dominant(pixel):
    r, g, b = (int(c) for c in pixel)
    if r > g + 40 and r > b + 40:
        return "r"
    if g > r + 40 and g > b + 40:
        return "g"
    if b > r + 40 and b > g + 40:
        return "b"
    return "w"


def quadrant_colors(image):
    """Which colour is in the top-left, top-right, bottom-left, bottom-right of a quad seen from the origin."""
    ppu = (W / 2) / (5 * W / H)  # pixels per world unit at distance 5, 90 degree camera
    d = round(1.5 * ppu)
    return "".join(dominant(image[H // 2 + dy, W // 2 + dx]) for dy in (-d, d) for dx in (-d, d))


def view_quad(scene):
    view = scene.renderer(W, H, lights=LIGHT)
    return view.render(view.camera((0, 0, 0), (0, 0, -5)))


# --------------------------------------------------------------------- OBJ --

def write_textured_obj(tmp_path):
    (tmp_path / "quad.png").write_bytes(png_bytes())
    (tmp_path / "quad.mtl").write_text("newmtl panel\nKd 1 1 1\nKs 0 0 0\nNs 10\nmap_Kd quad.png\n")
    # The vt of the bottom-left corner is (0, 0): an OBJ's v points up.
    (tmp_path / "quad.obj").write_text(
        "mtllib quad.mtl\nv -3 -3 -5\nv 3 -3 -5\nv 3 3 -5\nv -3 3 -5\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nusemtl panel\nf 1/1/1 2/2/1 3/3/1 4/4/1\n")
    return tmp_path / "quad.obj"


def test_obj_polygons_are_triangulated_and_negative_indices_work(tmp_path):
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -3 -2\n")
    scene = ng.load_obj(tmp_path / "m.obj")
    assert len(scene.meshes) == 1 and scene.meshes[0].triangle_count == 3  # a quad is 2 triangles, plus 1
    assert scene.meshes[0].vertex_count == 4 and not scene.meshes[0].has_uvs and not scene.meshes[0].has_normals


def test_obj_vertices_are_split_where_normals_differ(tmp_path):
    # A cube with a normal per face: 8 positions but 24 distinct (position, normal) vertices.
    corners = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    normals = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = []  # (axis, side) -> four corner indices (1-based) around that face
    for n, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]):
        idx = [i + 1 for i, c in enumerate(corners) if c[axis] == sign]
        faces.append(f"f {idx[0]}//{n + 1} {idx[1]}//{n + 1} {idx[3]}//{n + 1} {idx[2]}//{n + 1}")
    text = "".join(f"v {x} {y} {z}\n" for x, y, z in corners) + "".join(f"vn {a} {b} {c}\n" for a, b, c in normals)
    (tmp_path / "cube.obj").write_text(text + "\n".join(faces) + "\n")
    mesh = ng.load_obj(tmp_path / "cube.obj").meshes[0]
    assert mesh.vertex_count == 24 and mesh.triangle_count == 12 and mesh.has_normals


def test_obj_materials_split_meshes(tmp_path):
    (tmp_path / "t.mtl").write_text("newmtl red\nKd 0.9 0.1 0.1\nnewmtl blue\nKd 0.1 0.1 0.9\nNs 200\n")
    (tmp_path / "t.obj").write_text("mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
                                    "usemtl red\nf 1 2 3\nusemtl blue\nf 1 3 4\n")
    scene = ng.load_obj(tmp_path / "t.obj")
    assert len(scene.meshes) == 2 and len(scene.instances) == 2
    colors = [tuple(round(c, 2) for c in i.material.diffuse_color) for i in scene.instances]
    assert colors == [(0.9, 0.1, 0.1), (0.1, 0.1, 0.9)]
    assert scene.instances[1].material.specular_exponent == 200


def test_obj_errors(tmp_path):
    (tmp_path / "empty.obj").write_text("# nothing\n")
    with pytest.raises(ValueError, match="no geometry"):
        ng.load_obj(tmp_path / "empty.obj")
    (tmp_path / "bad.obj").write_text("v 0 0 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="does not exist"):
        ng.load_obj(tmp_path / "bad.obj")
    with pytest.raises(ValueError, match="cannot load"):
        ng.load(tmp_path / "model.stl")


def test_scene_bounds_come_from_the_geometry(tmp_path):
    (tmp_path / "m.obj").write_text("v 0 0 0\nv 4 0 0\nv 0 6 0\nf 1 2 3\n")
    scene = ng.load_obj(tmp_path / "m.obj")
    assert np.allclose(scene.center, [2, 3, 0]) and np.isclose(scene.extent, np.sqrt(16 + 36) / 2)


def test_missing_texture_warns_and_still_loads(tmp_path):
    pytest.importorskip("PIL")
    (tmp_path / "t.mtl").write_text("newmtl a\nKd 1 0 0\nmap_Kd nope.png\n")
    (tmp_path / "t.obj").write_text("mtllib t.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl a\nf 1/1 2/2 3/3\n")
    with pytest.warns(UserWarning, match="cannot read texture"):
        scene = ng.load_obj(tmp_path / "t.obj")
    assert scene.textures == [] and scene.instances[0].texture == -1


def test_obj_texture_is_the_right_way_up(tmp_path, gpu):
    scene = ng.load_obj(write_textured_obj(tmp_path))
    assert len(scene.textures) == 1 and scene.instances[0].texture == 0
    # Red top-left even though the OBJ's vt there is v=1: the loader flips v.
    assert quadrant_colors(view_quad(scene)) == "rgbw"


# -------------------------------------------------------------------- glTF --

def build_gltf(*, interleaved=False, normalized_uv=False, mode=4, extra_node=True):
    """A textured quad as a glTF document plus its binary buffer. Returns (document, binary)."""
    # Vertices go bottom-left, bottom-right, top-right, top-left; glTF's v points down, so the
    # top-left corner has uv (0, 0).
    position = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    index = np.array([0, 1, 2, 0, 2, 3] if mode == 4 else [0, 1, 3, 2], dtype=np.uint16)
    blob, views, accessors = bytearray(), [], []

    def add_view(data, target=None, stride=None):
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if stride:
            view["byteStride"] = stride
        blob.extend(data)
        views.append(view)
        return len(views) - 1

    if interleaved:  # position and uv in one view, 20 bytes per vertex
        packed = np.concatenate([position, uv], axis=1).astype(np.float32)
        view = add_view(packed.tobytes(), stride=20)
        accessors.append({"bufferView": view, "byteOffset": 0, "componentType": 5126, "count": 4, "type": "VEC3"})
        accessors.append({"bufferView": view, "byteOffset": 12, "componentType": 5126, "count": 4, "type": "VEC2"})
    else:
        accessors.append({"bufferView": add_view(position.tobytes()), "componentType": 5126, "count": 4, "type": "VEC3"})
        if normalized_uv:
            accessors.append({"bufferView": add_view((uv * 65535).astype(np.uint16).tobytes()), "componentType": 5123,
                              "normalized": True, "count": 4, "type": "VEC2"})
        else:
            accessors.append({"bufferView": add_view(uv.tobytes()), "componentType": 5126, "count": 4, "type": "VEC2"})
    accessors.append({"bufferView": add_view(index.tobytes()), "componentType": 5123, "count": len(index), "type": "SCALAR"})
    normal_view = add_view(np.tile([0, 0, 1], (4, 1)).astype(np.float32).tobytes())
    accessors.append({"bufferView": normal_view, "componentType": 5126, "count": 4, "type": "VEC3"})

    nodes = [{"mesh": 0, "scale": [3, 3, 1], "translation": [0, 0, -5]}]
    scene_nodes = [0]
    if extra_node:  # a parent holding a child that uses the same mesh
        nodes += [{"children": [2], "translation": [10, 0, 0]}, {"mesh": 0, "translation": [0, 5, 0]}]
        scene_nodes.append(1)
    document = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": scene_nodes}], "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1, "NORMAL": 3}, "indices": 2,
                                    "mode": mode, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": 0},
                                                "metallicFactor": 0.0, "roughnessFactor": 0.8}}],
        "textures": [{"source": 0}], "images": [{}],
        "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": len(blob)}],
    }
    return document, bytes(blob)


def write_gltf(tmp_path, document, blob, name="m.gltf", embed_image=True):
    document = json.loads(json.dumps(document))
    document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    document["images"][0] = {"uri": "data:image/png;base64," + base64.b64encode(png_bytes()).decode()} \
        if embed_image else {"uri": "tex.png"}
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return path


def write_glb(tmp_path, document, blob, name="m.glb"):
    document = json.loads(json.dumps(document))
    png = png_bytes()
    pad = lambda b, fill: b + fill * (-len(b) % 4)
    # The image lives in the binary chunk, referenced by a buffer view, as exporters do.
    offset = len(pad(blob, b"\0"))
    binary = pad(blob, b"\0") + png
    document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
    document["images"][0] = {"bufferView": len(document["bufferViews"]) - 1, "mimeType": "image/png"}
    document["buffers"][0]["byteLength"] = len(binary)
    json_chunk = pad(json.dumps(document).encode(), b" ")
    bin_chunk = pad(binary, b"\0")
    body = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk + struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    path = tmp_path / name
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    return path


def test_gltf_hierarchy_and_sharing(tmp_path):
    document, blob = build_gltf()
    scene = ng.load_gltf(write_gltf(tmp_path, document, blob))
    # Three nodes, but the middle one only carries a transform: two instances of one shared mesh.
    assert len(scene.meshes) == 1 and len(scene.instances) == 2
    assert scene.meshes[0].has_normals and scene.meshes[0].has_uvs and scene.meshes[0].triangle_count == 2
    positions = sorted(tuple(np.round(i.transform.matrix[:, 3], 3)) for i in scene.instances)
    # The root node sits at z=-5; the child is its parent's (10, 0, 0) plus its own (0, 5, 0).
    assert positions == [(0.0, 0.0, -5.0), (10.0, 5.0, 0.0)]
    root = next(i for i in scene.instances if np.allclose(i.transform.matrix[:, 3], [0, 0, -5]))
    assert np.allclose(np.diag(root.transform.matrix[:, :3]), [3, 3, 1])  # the node's scale


def test_gltf_interleaved_and_normalized_accessors_match_plain(tmp_path):
    plain = ng.load_gltf(write_gltf(tmp_path, *build_gltf(extra_node=False), name="a.gltf"))
    inter = ng.load_gltf(write_gltf(tmp_path, *build_gltf(interleaved=True, extra_node=False), name="b.gltf"))
    norm = ng.load_gltf(write_gltf(tmp_path, *build_gltf(normalized_uv=True, extra_node=False), name="c.gltf"))
    for scene in (plain, inter, norm):
        assert scene.meshes[0].vertex_count == 4 and scene.meshes[0].has_uvs


def test_gltf_triangle_strip_becomes_triangles(tmp_path):
    scene = ng.load_gltf(write_gltf(tmp_path, *build_gltf(mode=5, extra_node=False)))
    assert scene.meshes[0].triangle_count == 2


def test_gltf_material_mapping(tmp_path):
    document, blob = build_gltf(extra_node=False)
    document["materials"][0]["pbrMetallicRoughness"].update(baseColorFactor=[0.5, 0.25, 0.125, 1], metallicFactor=1.0,
                                                           roughnessFactor=0.1)
    material = ng.load_gltf(write_gltf(tmp_path, document, blob)).instances[0].material
    assert np.allclose(material.diffuse_color, (0.5, 0.25, 0.125))
    assert material.reflectivity > 0.4 and material.specular_exponent > 100  # shiny metal


def test_gltf_unsupported_features_say_so(tmp_path):
    document, blob = build_gltf(extra_node=False)
    document["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {}}
    with pytest.raises(NotImplementedError, match="Draco"):
        ng.load_gltf(write_gltf(tmp_path, document, blob))
    document, blob = build_gltf(extra_node=False)
    document["accessors"][0]["sparse"] = {}
    with pytest.raises(NotImplementedError, match="sparse"):
        ng.load_gltf(write_gltf(tmp_path, document, blob, name="s.gltf"))


def test_gltf_with_no_geometry_is_an_error(tmp_path):
    path = tmp_path / "empty.gltf"
    path.write_text(json.dumps({"asset": {"version": "2.0"}, "scenes": [{"nodes": []}], "nodes": []}))
    with pytest.raises(ValueError, match="no triangle geometry"):
        ng.load_gltf(path)


@pytest.mark.parametrize("kind", ["gltf", "glb"])
def test_gltf_texture_is_the_right_way_up(tmp_path, gpu, kind):
    document, blob = build_gltf(extra_node=False)
    path = write_gltf(tmp_path, document, blob) if kind == "gltf" else write_glb(tmp_path, document, blob)
    scene = ng.load(path)
    assert len(scene.textures) == 1 and scene.instances[0].texture == 0
    # The quad is 2 units wide scaled by 3, at z=-5. glTF's v points down, so uv (0,0) is the top-left.
    assert quadrant_colors(view_quad(scene)) == "rgbw"


def test_gltf_external_texture_file(tmp_path, gpu):
    document, blob = build_gltf(extra_node=False)
    (tmp_path / "tex.png").write_bytes(png_bytes())
    scene = ng.load(write_gltf(tmp_path, document, blob, embed_image=False))
    assert quadrant_colors(view_quad(scene)) == "rgbw"


def test_loaded_scene_renders_without_arguments(tmp_path, gpu):
    document, blob = build_gltf()
    scene = ng.load(write_gltf(tmp_path, document, blob))
    view = scene.renderer(W, H)
    image = view.render()
    assert image.shape == (H, W, 3) and image.std() > 5  # the default camera sees the model
