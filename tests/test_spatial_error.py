import numpy as np

from sdfmpneo.certification.spatial_error import (
    HomogeneousConductiveExterior,
    MeshApproximationProof,
    certify_conductive_outer_domain,
    certify_mesh_error,
    propagate_spatial_state_error,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def _one_tetra():
    return TetrahedralComplex3D.build(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=int),
    )


def test_mesh_certificate_is_fail_closed_and_uses_theorem_data_only():
    mesh = _one_tetra()
    proof = MeshApproximationProof(
        regularity_order=1.0,
        regularity_bound=2.0,
        interpolation_constant=3.0,
        certified=False,
        provenance="diagnostic only",
    )
    cert = certify_mesh_error(mesh, proof)
    assert not cert.certified
    assert not cert.as_ledger_term().certified
    assert np.isclose(cert.h_max, np.sqrt(2.0))
    assert cert.state_energy_error_bound >= 6.0 * np.sqrt(2.0)


def test_conductive_outer_certificate_uses_analytic_skin_depth_contraction():
    exterior = HomogeneousConductiveExterior(
        omega=2.0 * np.pi * 100e3,
        permeability=4e-7 * np.pi,
        conductivity=4.0,
        source_support_radius=0.2,
        inner_radius=0.4,
        outer_radius=0.8,
        certified_geometry=True,
        provenance="homogeneous spherical seawater exterior",
    )
    cert = certify_conductive_outer_domain(0.01, exterior)
    assert cert.certified
    assert 0.0 < cert.contraction_factor < 1.0
    assert cert.state_energy_error_bound >= 0.01
    assert cert.as_ledger_term().certified


def test_spatial_field_error_propagates_to_ports_and_heat_source():
    cert = propagate_spatial_state_error(
        state_energy_error_bound=0.02,
        omega=10.0,
        port_rhs_dual_norms=np.array([2.0, 3.0]),
        approximate_state_energy_norm=4.0,
        thermal_test_supremum_bounds=np.array([1.0, 0.5]),
    )
    assert cert.impedance_abs_error_bound.shape == (2, 2)
    assert np.all(cert.resistance_abs_error_bound == cert.impedance_abs_error_bound)
    assert np.allclose(cert.inductance_abs_error_bound, cert.impedance_abs_error_bound / 10.0)
    assert cert.heat_source_vector_error_bound > 0.0
