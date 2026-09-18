import numpy as np
import pytest
import scipy.sparse as sp

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_thermal import (
    _anchor_error,
    _anchor_relative_errors_grouped,
    _greedy_background_residual,
    _group_anchors,
    _transport_field,
    build_geometry_aware_thermal_library,
)
from sdfmpneo.unified_geometry import UnifiedUWPTGeometry


MATERIALS = {
    "tx_copper": dict(electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
                      reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
                      thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6),
    "rx_copper": dict(electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
                      reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
                      thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6),
    "tx_package": dict(electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
                       reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
                       thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6),
    "rx_package": dict(electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
                       reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
                       thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6),
    "seawater": dict(electrical_conductivity=5.0, resistivity_temperature_coefficient=0.0,
                     reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=80.0,
                     thermal_conductivity=0.6, volumetric_heat_capacity=4.1e6),
}


def make_geometry(offset=0.0, yaw=0.0):
    coil = dict(shape="circle", turns=0.5, outer_half_size=0.012, pitch=0.002,
                conductor_width=0.001, conductor_thickness=0.001, corner_radius=0.006)
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.010], angles=[0.0, 0.0, 0.0]),
        "receiver": dict(coil, translation=[offset, 0.0, 0.010], angles=[0.0, 0.0, yaw]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }



def test_grouped_resolvent_errors_match_scalar_evaluation():
    A = sp.diags([2.0, 3.0, 5.0, 7.0], format="csr")
    phi = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.1],
            [0.0, 0.3],
        ],
        dtype=float,
    )
    anchors = []
    for j, b in enumerate(
        (
            np.array([1.0, 0.2, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.3, 0.1]),
            np.array([0.1, 0.0, 0.0, 1.0]),
        )
    ):
        u = np.asarray(sp.linalg.spsolve(A, b), float)
        anchors.append(
            {
                "A": A,
                "b": b,
                "u": u,
                "denom2": float(u @ (A @ u)),
                "label": f"a{j}",
                "source_kind": "volume",
                "shift": 1.0,
                "geometry_index": 0,
                "port_index": None,
                "rhs_norm": float(np.linalg.norm(b)),
            }
        )
    grouped = dict(
        (anchor["label"], error)
        for anchor, error in _anchor_relative_errors_grouped(_group_anchors(anchors), phi)
    )
    scalar = dict(
        (anchor["label"], _anchor_error(anchor, phi)[0])
        for anchor in anchors
    )
    assert grouped.keys() == scalar.keys()
    for key in grouped:
        assert grouped[key] == pytest.approx(scalar[key], rel=1e-12, abs=1e-12)
