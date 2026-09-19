import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_geometry import (
    CoilGeometry,
    PackageGeometry,
    Pose,
    UnifiedUWPTGeometry,
)
from sdfmpneo.unified_thermal import (
    _partition_self_volume_anchors,
    _self_volume_local_window,
)


class _Background:
    def __init__(self):
        axis = np.linspace(-0.15, 0.15, 13)
        centers = 0.5 * (axis[:-1] + axis[1:])
        X, Y, Z = np.meshgrid(centers, centers, centers, indexing="ij")
        self.cell_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        self.dx = np.diff(axis)
        self.dy = np.diff(axis)
        self.dz = np.diff(axis)
        self.n_cells = self.cell_centers.shape[0]


def _geometry(tx_shift=(0.0, 0.0, 0.0), rx_shift=(0.0, 0.0, 0.06)):
    def coil(name, shift):
        pose = Pose(np.asarray(shift, float), np.zeros(3))
        return CoilGeometry(
            name=name,
            shape="circle",
            conductor_width=0.0015,
            conductor_thickness=0.001,
            pose=pose,
            turns=1.5,
            outer_half_size=0.025,
            pitch=0.002,
            corner_radius=0.012,
        )

    tx = coil("tx", tx_shift)
    rx = coil("rx", rx_shift)
    tx_package = PackageGeometry(np.array([0.035, 0.035, 0.005]), tx.pose)
    rx_package = PackageGeometry(np.array([0.035, 0.035, 0.005]), rx.pose)
    return UnifiedUWPTGeometry((tx, rx), (tx_package, rx_package))


def _anchor(A, b, case_label, source_kind, geometry_index=0, shift=10.0):
    b = np.asarray(b, float)
    u = np.asarray(b, float)  # A is identity in these partition tests.
    return {
        "A": A,
        "b": b,
        "u": u,
        "denom2": float(np.dot(u, b)),
        "label": f"geometry[{geometry_index}]/{case_label}/s={shift:g}",
        "case_label": case_label,
        "source_kind": source_kind,
        "shift": float(shift),
        "geometry_index": int(geometry_index),
        "rhs_norm": float(np.linalg.norm(b)),
    }


def test_self_volume_local_window_keeps_near_field_and_removes_far_field():
    background = _Background()
    geometry = _geometry()

    window = _self_volume_local_window(background, geometry, 0)

    assert window.shape == (background.n_cells,)
    assert np.all(np.isfinite(window))
    assert np.all((window >= 0.0) & (window <= 1.0))
    assert np.max(window) == 1.0
    assert np.min(window) == 0.0

    near = np.argmin(np.linalg.norm(background.cell_centers, axis=1))
    far = np.argmax(np.linalg.norm(background.cell_centers, axis=1))
    assert window[near] == 1.0
    assert window[far] == 0.0


def test_self_volume_partition_is_exact_and_keeps_cross_port_response_global():
    background = _Background()
    geometry = _geometry()
    n = background.n_cells
    A = sp.eye(n, format="csr")

    x = background.cell_centers
    b0 = np.exp(-np.sum((x - np.array([0.02, 0.0, 0.0])) ** 2, axis=1) / 0.003)
    b1 = np.exp(-np.sum((x - np.array([0.0, 0.0, 0.06])) ** 2, axis=1) / 0.003)
    cross_real = 0.25 * (b0 + b1)
    cross_quadrature = 0.1 * (b0 - b1)
    combined_real = b0 + b1 + cross_real
    combined_quadrature = b0 + b1 + cross_quadrature
    initial = np.ones(n)

    full = [
        _anchor(A, b0, "volume[0]", "volume"),
        _anchor(A, b1, "volume[1]", "volume"),
        _anchor(A, combined_real, "volume[2]", "volume"),
        _anchor(A, combined_quadrature, "volume[3]", "volume"),
        _anchor(A, initial, "initial[uniform]", "initial"),
    ]

    background_anchors, local = _partition_self_volume_anchors(
        background,
        geometry,
        full,
    )

    assert len(local) == 2
    assert len(local[0]) == 1
    assert len(local[1]) == 1

    labels = {row["case_label"] for row in background_anchors}
    assert "volume-cross-real[0,1]" in labels
    assert "volume-cross-quadrature[0,1]" in labels
    assert "initial[uniform]" in labels
    assert "volume-far[0]" in labels
    assert "volume-far[1]" in labels

    real_row = next(
        row for row in background_anchors
        if row["case_label"] == "volume-cross-real[0,1]"
    )
    quadrature_row = next(
        row for row in background_anchors
        if row["case_label"] == "volume-cross-quadrature[0,1]"
    )
    assert np.allclose(real_row["b"], cross_real)
    assert np.allclose(quadrature_row["b"], cross_quadrature)

    for port, original in enumerate((full[0], full[1])):
        local_row = local[port][0]
        far_row = next(
            row
            for row in background_anchors
            if row["case_label"] == f"volume-far[{port}]"
        )
        assert np.allclose(local_row["b"] + far_row["b"], original["b"])
        assert np.allclose(local_row["u"] + far_row["u"], original["u"])
        assert local_row["source_kind"] == "volume-local"
        assert far_row["source_kind"] == "volume-far"
