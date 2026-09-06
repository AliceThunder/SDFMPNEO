import numpy as np

from sdfmpneo.spatial import (
    RigidPose,
    SpiralCoilGeometry,
    UnderwaterWPTGeometry,
    read_gmsh_v22_ascii,
)


def test_circle_and_rounded_square_spiral_charts_are_finite_and_nondegenerate():
    tx = SpiralCoilGeometry(
        "circle", turns=3.0, outer_half_size=0.10, pitch=0.01,
        conductor_width=0.004, conductor_thickness=0.002,
    )
    rx = SpiralCoilGeometry(
        "rounded_square", turns=2.0, outer_half_size=0.11, pitch=0.01,
        conductor_width=0.004, conductor_thickness=0.002, corner_radius=0.04,
        pose=RigidPose(np.array([0.01, -0.02, 0.08]), pitch=0.1, yaw=-0.2),
    )
    geometry = UnderwaterWPTGeometry(tx, rx, np.array([0.14, 0.14, 0.01]), 0.2)
    points = rx.sample_for_geometric_tolerance(5e-4)
    assert points.ndim == 2 and points.shape[1] == 3
    assert np.all(np.isfinite(points))
    cert = geometry.chart_non_degeneracy()
    assert abs(cert["transmitter_rotation_det"] - 1.0) < 1e-13
    assert abs(cert["receiver_rotation_det"] - 1.0) < 1e-13
    assert cert["transmitter_inner_half_size"] > 0.0
    assert cert["receiver_inner_half_size"] > 0.0


def test_gmsh_v22_tags_feed_material_and_terminal_masks(tmp_path):
    text = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
$EndNodes
$Elements
5
1 2 1 11 1 2 3
2 2 1 12 1 2 4
3 2 1 13 1 3 4
4 2 1 14 2 3 4
5 4 1 21 1 2 3 4
$EndElements
"""
    path = tmp_path / "tagged.msh"
    path.write_text(text, encoding="utf-8")
    tagged = read_gmsh_v22_ascii(path)
    assert tagged.mesh.n_tetrahedra == 1
    assert tagged.tetra_mask(21).tolist() == [True]
    assert set(tagged.boundary_nodes(11).tolist()) == {0, 1, 2}
    assert tagged.boundary_mask(14).sum() == 1
