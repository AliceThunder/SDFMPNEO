from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .uwpt_geometry import TaggedTetrahedralMesh, UnderwaterWPTGeometry, read_gmsh_v22_ascii


@dataclass(frozen=True)
class UWPTPhysicalTags:
    tx_copper: int = 101
    rx_copper: int = 102
    tx_package: int = 201
    rx_package: int = 202
    seawater: int = 301
    tx_terminal_start: int = 1001
    tx_terminal_end: int = 1002
    rx_terminal_start: int = 1003
    rx_terminal_end: int = 1004
    outer_boundary: int = 2001


@dataclass(frozen=True)
class GmshMeshingResult:
    mesh_path: Path
    tagged_mesh: TaggedTetrahedralMesh
    physical_tags: UWPTPhysicalTags
    geometry_tolerance: float
    mesh_size: float
    seawater_radius: float


def _normalized(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        raise ValueError("zero vector cannot define an orientation")
    return v / n


def _package_corners(half_extent: np.ndarray, pose) -> np.ndarray:
    h = np.asarray(half_extent, dtype=float)
    corners = np.array(
        [
            [sx * h[0], sy * h[1], sz * h[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    return pose.apply(corners)


def _outer_sphere(geometry: UnderwaterWPTGeometry, geometry_tolerance: float) -> tuple[np.ndarray, float]:
    tx_corners = _package_corners(geometry.package_half_extent, geometry.transmitter.pose)
    rx_corners = _package_corners(geometry.package_half_extent, geometry.receiver.pose)
    all_points = np.vstack([tx_corners, rx_corners])
    center = 0.5 * (
        np.asarray(geometry.transmitter.pose.translation, dtype=float)
        + np.asarray(geometry.receiver.pose.translation, dtype=float)
    )
    radius = float(np.max(np.linalg.norm(all_points - center, axis=1)))
    radius += float(geometry.seawater_padding) + float(geometry_tolerance)
    return center, radius


def _add_oriented_box(occ, half_extent: np.ndarray, pose) -> int:
    h = np.asarray(half_extent, dtype=float)
    tag = occ.addBox(-h[0], -h[1], -h[2], 2.0 * h[0], 2.0 * h[1], 2.0 * h[2])
    dimtags = [(3, tag)]
    if pose.roll:
        occ.rotate(dimtags, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, float(pose.roll))
    if pose.pitch:
        occ.rotate(dimtags, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, float(pose.pitch))
    if pose.yaw:
        occ.rotate(dimtags, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, float(pose.yaw))
    occ.translate(dimtags, *map(float, pose.translation))
    return tag


def _add_rectangular_pipe(gmsh, coil, geometry_tolerance: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Sweep the declared rectangular conductor section along a certified polyline."""

    occ = gmsh.model.occ
    points = np.asarray(coil.sample_for_geometric_tolerance(geometry_tolerance), dtype=float)
    if points.shape[0] < 2:
        raise RuntimeError("coil centerline meshing requires at least two points")

    p_tags = [occ.addPoint(*map(float, point)) for point in points]
    lines = [occ.addLine(p_tags[k], p_tags[k + 1]) for k in range(len(p_tags) - 1)]
    wire = occ.addWire(lines)

    tangent = _normalized(points[1] - points[0])
    thickness_axis = _normalized(coil.pose.rotation @ np.array([0.0, 0.0, 1.0]))
    width_axis = _normalized(np.cross(thickness_axis, tangent))
    center = points[0]
    hw = 0.5 * float(coil.conductor_width)
    ht = 0.5 * float(coil.conductor_thickness)
    corners = np.array(
        [
            center - hw * width_axis - ht * thickness_axis,
            center + hw * width_axis - ht * thickness_axis,
            center + hw * width_axis + ht * thickness_axis,
            center - hw * width_axis + ht * thickness_axis,
        ]
    )
    ctags = [occ.addPoint(*map(float, point)) for point in corners]
    boundary = [occ.addLine(ctags[k], ctags[(k + 1) % 4]) for k in range(4)]
    loop = occ.addCurveLoop(boundary)
    section = occ.addPlaneSurface([loop])
    swept = occ.addPipe([(2, section)], wire, "DiscreteTrihedron")
    volumes = [tag for dim, tag in swept if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError("Gmsh pipe sweep did not produce exactly one conductor volume")
    return int(volumes[0]), points[0].copy(), points[-1].copy()


def _surface_nearest_to(model, occ, volume_tag: int, point: np.ndarray) -> int:
    boundary = model.getBoundary([(3, int(volume_tag))], combined=False, oriented=False, recursive=False)
    candidates = [tag for dim, tag in boundary if dim == 2]
    if not candidates:
        raise RuntimeError("volume has no terminal surface candidates")
    target = np.asarray(point, dtype=float)
    distances = []
    for surface in candidates:
        center = np.asarray(occ.getCenterOfMass(2, int(surface)), dtype=float)
        distances.append(float(np.linalg.norm(center - target)))
    return int(candidates[int(np.argmin(distances))])


def _physical_group(model, dim: int, entities: list[int], tag: int, name: str) -> None:
    if not entities:
        raise RuntimeError(f"physical group {name!r} has no entities")
    model.addPhysicalGroup(dim, [int(v) for v in entities], int(tag))
    model.setPhysicalName(dim, int(tag), name)


def mesh_underwater_wpt_geometry(
    geometry: UnderwaterWPTGeometry,
    output_path: str | Path,
    *,
    geometry_tolerance: float,
    mesh_size: float,
    physical_tags: UWPTPhysicalTags = UWPTPhysicalTags(),
) -> GmshMeshingResult:
    """Create the production UWPT CAD, tetrahedralize it and return tagged mesh.

    ``gmsh`` is imported only inside this function, keeping it an optional CAD
    dependency. ``geometry_tolerance`` is the declared centerline polyline
    chord-error budget; no hidden segment count is introduced. ``mesh_size`` is
    explicit and is expected to come from the W4 spatial error budget.
    """

    tol = float(geometry_tolerance)
    h = float(mesh_size)
    if tol <= 0.0 or h <= 0.0 or not np.isfinite(tol + h):
        raise ValueError("geometry_tolerance and mesh_size must be finite and positive")
    try:
        import gmsh  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError(
            "automatic CAD meshing requires the optional 'gmsh' Python package"
        ) from exc

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    center, water_radius = _outer_sphere(geometry, tol)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Geometry.OCCBooleanPreserveNumbering", 1)
        gmsh.model.add("SDFMPNEO_UWPT")
        model = gmsh.model
        occ = model.occ

        tx_copper, tx_start, tx_end = _add_rectangular_pipe(gmsh, geometry.transmitter, tol)
        rx_copper, rx_start, rx_end = _add_rectangular_pipe(gmsh, geometry.receiver, tol)
        tx_box = _add_oriented_box(occ, geometry.package_half_extent, geometry.transmitter.pose)
        rx_box = _add_oriented_box(occ, geometry.package_half_extent, geometry.receiver.pose)

        tx_package_out, _ = occ.cut(
            [(3, tx_box)], [(3, tx_copper)], removeObject=True, removeTool=False
        )
        rx_package_out, _ = occ.cut(
            [(3, rx_box)], [(3, rx_copper)], removeObject=True, removeTool=False
        )
        tx_package = [tag for dim, tag in tx_package_out if dim == 3]
        rx_package = [tag for dim, tag in rx_package_out if dim == 3]
        if not tx_package or not rx_package:
            raise RuntimeError("package/copper Boolean subtraction failed")

        water_raw = occ.addSphere(*map(float, center), float(water_radius))
        solid_tools = (
            [(3, int(tag)) for tag in tx_package]
            + [(3, int(tag)) for tag in rx_package]
            + [(3, int(tx_copper)), (3, int(rx_copper))]
        )
        water_out, _ = occ.cut(
            [(3, water_raw)], solid_tools, removeObject=True, removeTool=False
        )
        water = [tag for dim, tag in water_out if dim == 3]
        if not water:
            raise RuntimeError("seawater Boolean subtraction failed")

        occ.synchronize()

        tx_s0 = _surface_nearest_to(model, occ, tx_copper, tx_start)
        tx_s1 = _surface_nearest_to(model, occ, tx_copper, tx_end)
        rx_s0 = _surface_nearest_to(model, occ, rx_copper, rx_start)
        rx_s1 = _surface_nearest_to(model, occ, rx_copper, rx_end)
        if len({tx_s0, tx_s1}) != 2 or len({rx_s0, rx_s1}) != 2:
            raise RuntimeError("terminal surface identification collapsed")

        outer_surfaces: list[int] = []
        for volume in water:
            for dim, surface in model.getBoundary(
                [(3, int(volume))], combined=False, oriented=False, recursive=False
            ):
                if dim != 2:
                    continue
                com = np.asarray(occ.getCenterOfMass(2, int(surface)), dtype=float)
                if float(np.linalg.norm(com - center)) > 0.75 * water_radius:
                    outer_surfaces.append(int(surface))
        outer_surfaces = sorted(set(outer_surfaces))

        _physical_group(model, 3, [tx_copper], physical_tags.tx_copper, "tx_copper")
        _physical_group(model, 3, [rx_copper], physical_tags.rx_copper, "rx_copper")
        _physical_group(model, 3, tx_package, physical_tags.tx_package, "tx_package")
        _physical_group(model, 3, rx_package, physical_tags.rx_package, "rx_package")
        _physical_group(model, 3, water, physical_tags.seawater, "seawater")
        _physical_group(model, 2, [tx_s0], physical_tags.tx_terminal_start, "tx_terminal_start")
        _physical_group(model, 2, [tx_s1], physical_tags.tx_terminal_end, "tx_terminal_end")
        _physical_group(model, 2, [rx_s0], physical_tags.rx_terminal_start, "rx_terminal_start")
        _physical_group(model, 2, [rx_s1], physical_tags.rx_terminal_end, "rx_terminal_end")
        _physical_group(model, 2, outer_surfaces, physical_tags.outer_boundary, "outer_boundary")

        model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    tagged = read_gmsh_v22_ascii(path)
    return GmshMeshingResult(
        mesh_path=path,
        tagged_mesh=tagged,
        physical_tags=physical_tags,
        geometry_tolerance=tol,
        mesh_size=h,
        seawater_radius=water_radius,
    )
