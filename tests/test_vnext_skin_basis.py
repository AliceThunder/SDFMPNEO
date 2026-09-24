import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    Scene,
    SuperellipseSpiral,
    adaptive_section_basis,
    polynomial_section_basis,
)


def test_constant_mode_is_exact_and_all_other_modes_have_zero_net_current():
    basis = polynomial_section_basis(
        1.6e-3,
        0.8e-3,
        3.5,
        degree=3,
        radial_order=8,
        angular_order=64,
    )
    expected = (
        1.0
        / np.sqrt(
            basis.area
        )
    )
    assert np.allclose(
        basis.values[:, 0],
        expected,
        rtol=2e-12,
        atol=2e-9,
    )
    assert np.isclose(
        basis.moments[0],
        np.sqrt(
            basis.area
        ),
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.max(
        np.abs(
            basis.moments[1:]
        )
    ) < 2e-12


def test_skin_boundary_enrichment_is_orthonormal_and_nested_in_rank():
    polynomial = polynomial_section_basis(
        2.0e-3,
        1.0e-3,
        4.0,
        degree=1,
        radial_order=6,
        angular_order=48,
    )
    enriched = adaptive_section_basis(
        2.0e-3,
        1.0e-3,
        4.0,
        degree=1,
        radial_order=6,
        angular_order=48,
        skin_parameter=8.0,
        skin_threshold=2.0,
        boundary_layers=2,
        boundary_angular_order=1,
    )
    assert (
        enriched.n_modes
        > polynomial.n_modes
    )
    gram = (
        enriched.values.T
        @ (
            enriched.quadrature.weights[
                :,
                None,
            ]
            * enriched.values
        )
    )
    assert np.allclose(
        gram,
        np.eye(
            enriched.n_modes
        ),
        rtol=2e-10,
        atol=2e-10,
    )
    assert np.max(
        np.abs(
            enriched.moments[
                1:
            ]
        )
    ) < 5e-11
    assert (
        enriched.skin_parameter
        == 8.0
    )


def test_teacher_activates_intrinsic_skin_modes_only_when_needed():
    copper = ConductorMaterial(
        5.8e7
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.03,
            0.026,
            0.75,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=2.0e-3,
            conductor_thickness=1.0e-3,
        ),
        copper,
    )
    scene = Scene(
        (coil,),
        HomogeneousMedium(),
    )
    config = MQSConfig(
        segments_per_turn=6,
        min_segments=6,
        section_degree=1,
        radial_order=5,
        angular_order=32,
        line_order=2,
        section_basis_family="adaptive",
        skin_enrichment_threshold=2.0,
        skin_boundary_layers=2,
        skin_angular_order=1,
    )
    low = DenseMQSTeacher(
        scene,
        100.0,
        config,
    )
    high = DenseMQSTeacher(
        scene,
        200_000.0,
        config,
    )
    low_basis = (
        low._segments[
            0
        ].basis
    )
    high_basis = (
        high._segments[
            0
        ].basis
    )
    assert (
        low_basis.skin_parameter
        < config.skin_enrichment_threshold
    )
    assert (
        high_basis.skin_parameter
        > config.skin_enrichment_threshold
    )
    assert (
        high_basis.n_modes
        > low_basis.n_modes
    )


def test_polynomial_policy_can_be_forced_for_reference_comparison():
    copper = ConductorMaterial(
        5.8e7
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.03,
            0.026,
            0.75,
            0.001,
            0.001,
            conductor_width=2.0e-3,
            conductor_thickness=1.0e-3,
        ),
        copper,
    )
    scene = Scene(
        (coil,),
        HomogeneousMedium(),
    )
    config = MQSConfig(
        segments_per_turn=6,
        min_segments=6,
        section_degree=2,
        radial_order=6,
        angular_order=32,
        section_basis_family="polynomial",
    )
    teacher = DenseMQSTeacher(
        scene,
        500_000.0,
        config,
    )
    assert (
        teacher._segments[
            0
        ].basis.skin_parameter
        == 0.0
    )
