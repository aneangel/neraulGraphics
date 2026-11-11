"""Load OBJ and glTF/GLB models into meshes, instances and textures.

::

    import neraulgraphics as ng

    scene = ng.load("model.glb")                 # or .gltf, .obj
    view = scene.renderer(1280, 720)             # frames the model and lights it
    image = view.render()                        # (720, 1280, 3) uint8

    # or hand the pieces to a Renderer yourself
    renderer = ng.Renderer(1280, 720, [], lights, meshes=scene.meshes,
                           instances=scene.instances, textures=scene.textures)

Meshes keep their vertex normals (so curved models shade smooth) and texture
coordinates, and each material's base colour and colour texture are carried over. glTF
node transforms become instance transforms, and a mesh used by many nodes is stored once.

Decoding image files needs Pillow (``pip install neraulgraphics[images]``). A model whose
textures cannot be read is still loaded, with a warning and plain colours.

Not supported: animation, skinning, normal or roughness maps, transparency, and glTF
files that use Draco compression or sparse accessors (these raise ``NotImplementedError``).
Material shininess and metalness are mapped to the renderer's simpler model, so they are
approximate.
"""

from __future__ import annotations

import base64
import json
import struct
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import numpy as np

from ._adapter import SimRenderer
from ._core import Instance, Material, Mesh, Texture, Transform

__all__ = ["LoadedScene", "SceneRenderer", "load", "load_gltf", "load_obj"]


# ----------------------------------------------------------------- the result --

@dataclass
class LoadedScene:
    """A model ready for a :class:`neraulgraphics.Renderer`."""

    meshes: list[Mesh] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    textures: list[Texture] = field(default_factory=list)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)  # both OBJ and glTF are normally Y-up
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))  # of the placed geometry
    extent: float = 1.0                                              # half its diagonal
    bounds_min: np.ndarray = field(default_factory=lambda: np.zeros(3))   # the axis-aligned box round the placed geometry
    bounds_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def renderer(self, width: int = 1280, height: int = 720, **kwargs) -> SceneRenderer:
        """A :class:`SceneRenderer` for this model, with a camera and lights chosen from its size."""
        return SceneRenderer(self, width, height, **kwargs)


class SceneRenderer(SimRenderer):
    """Draws a :class:`LoadedScene`. ``camera()`` frames the whole model; pass ``lights=`` to override."""

    def __init__(self, scene: LoadedScene, width: int = 1280, height: int = 720, *, lights=None,
                 fov: float = 1.0, ambient: float = 0.2, device: str = "auto") -> None:
        super().__init__(width, height, scene, up=scene.up, center=scene.center, extent=scene.extent,
                         lights=lights, fov=fov, ambient=ambient, device=device)

    def render_frame(self, camera=None, frame: int = 0, samples_per_frame: int = 1, *, rgb: bool = True,
                     depth: bool = False, normals: bool = False, object_ids: bool = False, zero_copy: bool = False) -> dict:
        """The images asked for as a dict: ``rgb``, ``depth``, ``normals``, ``object_ids``. See :meth:`neraulgraphics.Renderer.render_frame`."""
        return self.renderer.render_frame(self.camera() if camera is None else camera, frame, samples_per_frame, rgb=rgb,
                                          depth=depth, normals=normals, object_ids=object_ids, zero_copy=zero_copy)

    def render(self, camera=None, frame: int = 0, samples_per_frame: int = 1) -> np.ndarray:
        """Draws the model as a ``(height, width, 3)`` uint8 image. Increase ``frame`` to sharpen it."""
        return self.renderer.render(self.camera() if camera is None else camera, frame, samples_per_frame)


def _finish(scene: LoadedScene, mesh_bounds: list[tuple[np.ndarray, np.ndarray]]) -> LoadedScene:
    """Fills in center and extent from where the instances put their meshes."""
    corners = []
    for inst in scene.instances:
        lo, hi = mesh_bounds[inst.mesh]
        box = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        m = inst.transform.matrix.astype(np.float64)
        corners.append(box @ m[:, :3].T + m[:, 3])
    if corners:
        points = np.concatenate(corners)
        lo, hi = points.min(axis=0), points.max(axis=0)
        scene.center, scene.extent = (lo + hi) / 2.0, max(float(np.linalg.norm(hi - lo)) / 2.0, 1e-3)
        scene.bounds_min, scene.bounds_max = lo, hi
    return scene


def _bounds(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices.min(axis=0).astype(np.float64), vertices.max(axis=0).astype(np.float64)


def load(path: str | Path) -> LoadedScene:
    """Loads ``.obj``, ``.gltf`` or ``.glb``, chosen by the file's extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".obj":
        return load_obj(path)
    if suffix in (".gltf", ".glb"):
        return load_gltf(path)
    raise ValueError(f"cannot load {suffix!r} files; supported: .obj, .gltf, .glb")


# ---------------------------------------------------------------------- images --

class _Images:
    """Decodes image files to Textures, once each, warning (not failing) when it cannot."""

    def __init__(self) -> None:
        self.textures: list[Texture] = []
        self._by_key: dict[str, int] = {}

    def add(self, key: str, read) -> int:
        """The texture index for ``key``, decoding the bytes ``read()`` returns the first time. -1 if it fails."""
        if key not in self._by_key:
            self._by_key[key] = self._decode(key, read)
        return self._by_key[key]

    def _decode(self, key: str, read) -> int:
        try:
            from PIL import Image
        except ImportError:
            warnings.warn("neraulgraphics: Pillow is needed to read texture images, so textures are skipped "
                          "(pip install neraulgraphics[images])", stacklevel=4)
            return -1
        try:
            with Image.open(BytesIO(read())) as image:
                pixels = np.ascontiguousarray(np.asarray(image.convert("RGBA"), dtype=np.uint8))
        except Exception as error:  # a missing or corrupt file should not lose the whole model
            warnings.warn(f"neraulgraphics: cannot read texture {key!r}: {error}", stacklevel=4)
            return -1
        self.textures.append(Texture(pixels))
        return len(self.textures) - 1


def _read_file(path: Path):
    return lambda: path.read_bytes()


# -------------------------------------------------------------------------- OBJ --

def _parse_mtl(path: Path, images: _Images) -> dict[str, tuple[Material, int]]:
    """Materials of an .mtl file as (Material, texture index) by name."""
    out: dict[str, tuple[Material, int]] = {}
    current: dict | None = None

    def finish() -> None:
        if current is None:
            return
        spec = np.mean(current.get("Ks", (0.0, 0.0, 0.0)))
        material = Material(diffuse_color=tuple(float(c) for c in current.get("Kd", (0.8, 0.8, 0.8))),
                            diffuse_albedo=0.9, specular_albedo=float(np.clip(spec, 0.0, 1.0)) * 0.5,
                            reflectivity=0.0, specular_exponent=float(np.clip(current.get("Ns", 50.0), 1.0, 2000.0)))
        out[current["name"]] = (material, current.get("texture", -1))

    for line in path.read_text(errors="replace").splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        key, args = parts[0], parts[1:]
        if key == "newmtl" and args:
            finish()
            current = {"name": " ".join(args)}
        elif current is not None and key in ("Kd", "Ks") and len(args) >= 3:
            current[key] = [float(v) for v in args[:3]]
        elif current is not None and key == "Ns" and args:
            current["Ns"] = float(args[0])
        elif current is not None and key == "map_Kd" and args:
            # Options come first (-s 1 1 1 ...); the file name is the last word, or the rest if it has spaces.
            file = path.parent / unquote(" ".join(args[_skip_map_options(args):]).replace("\\", "/"))
            current["texture"] = images.add(str(file.resolve()), _read_file(file))
    finish()
    return out


def _skip_map_options(args: list[str]) -> int:
    """Index of the file name in a ``map_Kd`` line, past options such as ``-s 1 1 1`` or ``-clamp on``."""
    counts = {"-s": 3, "-o": 3, "-t": 3, "-mm": 2, "-bm": 1, "-blendu": 1, "-blendv": 1, "-cc": 1, "-clamp": 1,
              "-imfchan": 1, "-texres": 1, "-type": 1}
    i = 0
    while i < len(args) and args[i].startswith("-") and args[i] in counts:
        i += 1 + counts[args[i]]
    return min(i, len(args) - 1)


def load_obj(path: str | Path) -> LoadedScene:
    """Loads a Wavefront OBJ file with its .mtl materials and colour textures.

    Faces become one mesh per material. Polygons are fan-triangulated. Texture coordinates have
    their v flipped, because an OBJ's v points up and the renderer's points down.
    """
    path = Path(path)
    images = _Images()
    materials: dict[str, tuple[Material, int]] = {}

    positions: list[list[float]] = []
    uvs: list[list[float]] = []
    normals: list[list[float]] = []
    # Per material: face-vertex triplets (position, uv, normal) with -1 for "absent", and triangle starts.
    groups: dict[str, list[tuple[int, int, int]]] = {}
    triangles: dict[str, list[tuple[int, int, int]]] = {}
    current = ""

    def index(token: str, count: int) -> int:
        if not token:
            return -1
        i = int(token)
        return i - 1 if i > 0 else count + i  # negative indices count back from the latest

    with path.open(errors="replace") as file:
        for raw in file:
            line = raw.split("#")[0].strip()
            if not line:
                continue
            key, _, rest = line.partition(" ")
            args = rest.split()
            if key == "v" and len(args) >= 3:
                positions.append([float(a) for a in args[:3]])
            elif key == "vt" and args:
                uvs.append([float(args[0]), float(args[1]) if len(args) > 1 else 0.0])
            elif key == "vn" and len(args) >= 3:
                normals.append([float(a) for a in args[:3]])
            elif key == "mtllib" and rest.strip():
                mtl = path.parent / unquote(rest.strip())
                if mtl.exists():
                    materials.update(_parse_mtl(mtl, images))
                else:
                    warnings.warn(f"neraulgraphics: material library {mtl} not found", stacklevel=2)
            elif key == "usemtl":
                current = rest.strip()
            elif key == "f" and len(args) >= 3:
                verts = []
                for token in args:
                    part = token.split("/")
                    part += [""] * (3 - len(part))
                    verts.append((index(part[0], len(positions)), index(part[1], len(uvs)),
                                  index(part[2], len(normals))))
                tri_list = groups.setdefault(current, [])
                faces = triangles.setdefault(current, [])
                base = len(tri_list)
                tri_list.extend(verts)
                for k in range(1, len(verts) - 1):  # a fan from the first vertex
                    faces.append((base, base + k, base + k + 1))

    if not positions or not groups:
        raise ValueError(f"{path} has no geometry")
    pos = np.array(positions, dtype=np.float32)
    uv_array = np.array(uvs, dtype=np.float32) if uvs else None
    normal_array = np.array(normals, dtype=np.float32) if normals else None
    if uv_array is not None:
        uv_array[:, 1] = 1.0 - uv_array[:, 1]  # OBJ v points up; ours points down

    scene = LoadedScene()
    mesh_bounds = []
    default = (Material(diffuse_color=(0.8, 0.8, 0.8)), -1)
    for name, verts in groups.items():
        corner = np.array(verts, dtype=np.int64)
        face = np.array(triangles[name], dtype=np.int64)
        if corner[:, 0].min() < 0 or corner[:, 0].max() >= len(pos):
            raise ValueError(f"{path} has a face that refers to a vertex that does not exist")
        # Positions, uvs and normals are indexed separately in an OBJ; the renderer wants one
        # index per vertex, so each distinct (position, uv, normal) becomes one vertex.
        unique, inverse = np.unique(corner, axis=0, return_inverse=True)
        mesh_pos = pos[unique[:, 0]]
        mesh_uv = mesh_normal = None
        if uv_array is not None and (unique[:, 1] >= 0).any():
            mesh_uv = np.where((unique[:, 1] >= 0)[:, None], uv_array[np.clip(unique[:, 1], 0, None)], 0.0)
        if normal_array is not None and (unique[:, 2] >= 0).any():
            mesh_normal = np.where((unique[:, 2] >= 0)[:, None], normal_array[np.clip(unique[:, 2], 0, None)], 0.0)
        triangle_index = inverse.reshape(-1)[face]
        scene.meshes.append(Mesh(mesh_pos, triangle_index.astype(np.uint32),
                                 None if mesh_normal is None else mesh_normal.astype(np.float32),
                                 None if mesh_uv is None else mesh_uv.astype(np.float32)))
        mesh_bounds.append(_bounds(mesh_pos))
        material, texture = materials.get(name, default)
        scene.instances.append(Instance(len(scene.meshes) - 1, Transform(np.eye(4, dtype=np.float32)), material,
                                        texture if mesh_uv is not None else -1))
    scene.textures = images.textures
    return _finish(scene, mesh_bounds)


# ------------------------------------------------------------------------- glTF --

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class _Gltf:
    """The parts of a glTF document the loader reads: buffers and accessors."""

    def __init__(self, document: dict, base: Path, glb_binary: bytes | None) -> None:
        self.doc = document
        self.base = base
        self._buffers: dict[int, bytes] = {}
        self._glb = glb_binary

    def buffer(self, index: int) -> bytes:
        if index not in self._buffers:
            spec = self.doc["buffers"][index]
            uri = spec.get("uri")
            if uri is None:
                if self._glb is None:
                    raise ValueError(f"buffer {index} has no uri and the file has no binary chunk")
                self._buffers[index] = self._glb
            else:
                self._buffers[index] = self.read_uri(uri)
        return self._buffers[index]

    def read_uri(self, uri: str) -> bytes:
        if uri.startswith("data:"):
            header, _, payload = uri.partition(",")
            return base64.b64decode(payload) if header.endswith(";base64") else unquote(payload).encode("latin-1")
        return (self.base / unquote(uri)).read_bytes()

    def view_bytes(self, index: int) -> bytes:
        view = self.doc["bufferViews"][index]
        start = view.get("byteOffset", 0)
        return self.buffer(view["buffer"])[start:start + view["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        """An accessor as an array of shape (count, components), as float32 (normalized ints scaled to 0..1 or -1..1)
        or in its own integer type."""
        spec = self.doc["accessors"][index]
        if "sparse" in spec:
            raise NotImplementedError("glTF sparse accessors are not supported")
        dtype = np.dtype(_COMPONENT[spec["componentType"]])
        width, count = _WIDTH[spec["type"]], spec["count"]
        if "bufferView" not in spec:
            data = np.zeros((count, width), dtype=dtype)  # no view means zeros, per the spec
        else:
            view = self.doc["bufferViews"][spec["bufferView"]]
            buffer = self.buffer(view["buffer"])
            stride = view.get("byteStride") or width * dtype.itemsize
            offset = view.get("byteOffset", 0) + spec.get("byteOffset", 0)
            data = np.ndarray((count, width), dtype=dtype, buffer=buffer, offset=offset,
                              strides=(stride, dtype.itemsize)).copy()
        if spec.get("normalized") and dtype.kind in "iu":
            data = data.astype(np.float32) / (np.iinfo(dtype).max if dtype.kind == "u" else -float(np.iinfo(dtype).min))
            data = np.maximum(data, -1.0) if dtype.kind == "i" else data
        return data


def _read_glb(raw: bytes) -> tuple[dict, bytes | None]:
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError("not a version 2 GLB file")
    document, binary, offset = None, None, 12
    while offset < min(length, len(raw)):
        size, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8:offset + 8 + size]
        if kind == 0x4E4F534A:    # "JSON"
            document = json.loads(chunk)
        elif kind == 0x004E4942:  # "BIN\0"
            binary = chunk
        offset += 8 + size
    if document is None:
        raise ValueError("GLB file has no JSON chunk")
    return document, binary


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF stores columns
    m = np.eye(4)
    scale = np.diag([*node.get("scale", (1.0, 1.0, 1.0)), 1.0])
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m[:3, :3] = rot
    m[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return m @ scale


def _triangles(mode: int, index: np.ndarray) -> np.ndarray | None:
    if mode == 4:
        return index[: len(index) // 3 * 3].reshape(-1, 3)
    if mode == 5 and len(index) >= 3:  # strip: every other triangle is flipped to keep winding
        t = np.stack([index[:-2], index[1:-1], index[2:]], axis=1)
        t[1::2] = t[1::2][:, [1, 0, 2]]
        return t
    if mode == 6 and len(index) >= 3:  # fan
        return np.stack([np.full(len(index) - 2, index[0]), index[1:-1], index[2:]], axis=1)
    return None  # points and lines have no surface


def _gltf_material(spec: dict | None) -> tuple[Material, dict | None]:
    """The renderer's Material for a glTF material, and its base-colour texture reference."""
    spec = spec or {}
    pbr = spec.get("pbrMetallicRoughness", {})
    color, metallic, rough = pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), pbr.get("metallicFactor", 1.0), \
        pbr.get("roughnessFactor", 1.0)
    texture = pbr.get("baseColorTexture")
    emissive = spec.get("emissiveFactor", (0.0, 0.0, 0.0))
    strength = float(spec.get("extensions", {}).get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0))
    legacy = spec.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")
    if legacy is not None:  # the older workflow still found in many files
        color, texture = legacy.get("diffuseFactor", (1.0, 1.0, 1.0, 1.0)), legacy.get("diffuseTexture")
        metallic, rough = 0.0, 1.0 - legacy.get("glossinessFactor", 1.0)
    # The renderer is Blinn-Phong with a mirror term, so this is a likeness, not a conversion:
    # smoother surfaces get tighter highlights, metal gets reflection and less diffuse.
    rough = float(np.clip(rough, 0.05, 1.0))
    material = Material(diffuse_color=tuple(float(c) for c in color[:3]),
                        diffuse_albedo=float(1.0 - 0.6 * metallic), specular_albedo=float(0.15 + 0.25 * (1.0 - rough)),
                        reflectivity=float(metallic * (1.0 - rough) * 0.6),
                        specular_exponent=float(np.clip(2.0 / rough**2, 4.0, 1500.0)),
                        # What path tracing reads: glTF's own parameters, which it was designed around.
                        roughness=rough, metallic=float(np.clip(metallic, 0.0, 1.0)),
                        emission=tuple(float(c) * strength for c in emissive[:3]))
    return material, texture


def load_gltf(path: str | Path) -> LoadedScene:
    """Loads a glTF 2.0 file (``.gltf`` or binary ``.glb``): meshes, node transforms, base colours and textures.

    Each node that has a mesh becomes instances (one per primitive) placed by the node's accumulated
    transform, and a mesh shared by many nodes is stored once.
    """
    path = Path(path)
    raw = path.read_bytes()
    document, binary = _read_glb(raw) if raw[:4] == b"glTF" else (json.loads(raw), None)
    gltf = _Gltf(document, path.parent, binary)
    images = _Images()

    # One renderer Mesh per (glTF mesh, primitive, uv set); the uv set depends on the material's texture.
    mesh_cache: dict[tuple[int, int, int], int] = {}
    scene = LoadedScene()
    mesh_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    texture_cache: dict[int, int] = {}

    def texture_index(reference: dict | None) -> tuple[int, int]:
        """(renderer texture index or -1, uv set) for a glTF textureInfo."""
        if not reference:
            return -1, 0
        t = reference["index"]
        if t not in texture_cache:
            tex = document.get("textures", [])[t]
            source = tex.get("source")
            if source is None:
                texture_cache[t] = -1
            else:
                image = document["images"][source]
                if "uri" in image:
                    texture_cache[t] = images.add(f"image {source}", lambda uri=image["uri"]: gltf.read_uri(uri))
                else:
                    texture_cache[t] = images.add(f"image {source}", lambda v=image["bufferView"]: gltf.view_bytes(v))
        return texture_cache[t], int(reference.get("texCoord", 0))

    def primitive_mesh(mesh_id: int, prim_id: int, uv_set: int) -> int | None:
        key = (mesh_id, prim_id, uv_set)
        if key in mesh_cache:
            return mesh_cache[key]
        prim = document["meshes"][mesh_id]["primitives"][prim_id]
        if "KHR_draco_mesh_compression" in prim.get("extensions", {}):
            raise NotImplementedError("glTF files with Draco compression are not supported")
        attributes = prim.get("attributes", {})
        if "POSITION" not in attributes:
            return None
        position = gltf.accessor(attributes["POSITION"]).astype(np.float32)
        index = (gltf.accessor(prim["indices"]).reshape(-1).astype(np.int64) if "indices" in prim
                 else np.arange(len(position), dtype=np.int64))
        triangles = _triangles(prim.get("mode", 4), index)
        if triangles is None or len(triangles) == 0:
            mesh_cache[key] = None
            return None
        normal = gltf.accessor(attributes["NORMAL"]).astype(np.float32) if "NORMAL" in attributes else None
        uv_name = f"TEXCOORD_{uv_set}"
        uv = gltf.accessor(attributes[uv_name]).astype(np.float32) if uv_name in attributes else None
        scene.meshes.append(Mesh(position, triangles.astype(np.uint32), normal, uv))  # glTF's v already points down
        mesh_bounds.append(_bounds(position))
        mesh_cache[key] = len(scene.meshes) - 1
        return mesh_cache[key]

    def visit(node_id: int, parent: np.ndarray) -> None:
        node = document["nodes"][node_id]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            mesh_id = node["mesh"]
            for prim_id, prim in enumerate(document["meshes"][mesh_id]["primitives"]):
                material, reference = _gltf_material(document.get("materials", [None])[prim["material"]]
                                                     if "material" in prim else None)
                texture, uv_set = texture_index(reference)
                mesh = primitive_mesh(mesh_id, prim_id, uv_set)
                if mesh is None:
                    continue
                scene.instances.append(Instance(mesh, Transform(world[:3, :].astype(np.float32)), material,
                                                texture if scene.meshes[mesh].has_uvs else -1))
        for child in node.get("children", []):
            visit(child, world)

    scenes = document.get("scenes", [])
    if scenes:
        roots = scenes[document.get("scene", 0)].get("nodes", [])
    else:  # no scene: every node nothing else contains
        children = {c for n in document.get("nodes", []) for c in n.get("children", [])}
        roots = [i for i in range(len(document.get("nodes", []))) if i not in children]
    for root in roots:
        visit(root, np.eye(4))

    if not scene.instances:
        raise ValueError(f"{path} has no triangle geometry")
    scene.textures = images.textures
    return _finish(scene, mesh_bounds)
