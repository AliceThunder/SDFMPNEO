import numpy as np

from sdfmpneo_vnext import (
    HomogeneousThermalMedium,
    PreparedMultiThermalInterfaceField,
    RigidPose,
    SuperquadricPackageGeometry,
    ThermalSourceQuadrature,
    haar_rotation,
)


def _source(
    positions,
):
    positions = np.asarray(
        positions,
        dtype=float,
    )
    count = len(
        positions
    )
    weights = np.full(
        count,
        1.0e-6,
        dtype=float,
    )
    matrices = np.full(
        (
            count,
            1,
            1,
        ),
        1.0e6,
        dtype=complex,
    )
    return ThermalSourceQuadrature(
        positions=positions,
        volume_weights=weights,
        coil_index=np.zeros(
            count,
            dtype=int,
        ),
        arc_fraction=np.zeros(
            count,
            dtype=float,
        ),
        xy=np.zeros(
            (
                count,
                2,
            ),
            dtype=float,
        ),
        dissipation_matrices=(
            matrices
        ),
        effective_radius=np.full(
            count,
            1.5e-3,
            dtype=float,
        ),
        normalization_closure_error=0.0,
        normalization_correction=1.0,
    )


def _geometry(
    center,
):
    return SuperquadricPackageGeometry(
        np.asarray(
            [
                0.012,
                0.010,
                0.009,
            ]
        ),
        exponent_xy=2.0,
        exponent_z=2.0,
        pose=RigidPose(
            np.eye(
                3
            ),
            np.asarray(
                center,
                dtype=float,
            ),
        ),
    )


def _background():
    return HomogeneousThermalMedium(
        conductivity=0.6,
        density=1000.0,
        heat_capacity=4000.0,
        ambient_temperature=293.15,
    )


def _regions(
    first,
    second,
):
    return (
        (
            0,
            first,
            HomogeneousThermalMedium(
                conductivity=0.22,
                density=1200.0,
                heat_capacity=1800.0,
                ambient_temperature=293.15,
            ),
        ),
        (
            1,
            second,
            HomogeneousThermalMedium(
                conductivity=1.4,
                density=900.0,
                heat_capacity=2400.0,
                ambient_temperature=293.15,
            ),
        ),
    )


def _field(
    first,
    second,
    source,
):
    return PreparedMultiThermalInterfaceField(
        source,
        _background(),
        _regions(
            first,
            second,
        ),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        mfs_offset_fraction=0.12,
        stehfest_order=6,
        interface_residual_tolerance=2e-3,
        svd_rcond=1e-10,
    )


def test_multi_package_thermal_interface_has_finite_steady_and_transient_response():
    first = _geometry(
        (
            -0.028,
            0.0,
            0.0,
        )
    )
    second = _geometry(
        (
            0.028,
            0.0,
            0.0,
        )
    )
    source = _source(
        (
            (
                -0.028,
                0.0,
                0.0,
            ),
            (
                0.028,
                0.0,
                0.0,
            ),
        )
    )
    field = _field(
        first,
        second,
        source,
    )
    query = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.0,
            ],
            [
                0.028,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.035,
            ],
        ]
    )
    currents = np.asarray(
        [
            1.0
            + 0.0j
        ]
    )

    steady = field.steady_temperature(
        query,
        currents,
    )
    transient = field.temperature_step(
        query,
        5.0,
        currents,
    )
    assert np.all(
        np.isfinite(
            steady
        )
    )
    assert np.all(
        np.isfinite(
            transient
        )
    )
    assert np.all(
        steady
        > field.medium.ambient_temperature
    )
    assert np.all(
        transient
        > field.medium.ambient_temperature
    )
    assert (
        field.maximum_interface_residual
        <= field.interface_residual_tolerance
    )


def test_multi_package_thermal_interface_is_common_se3_invariant():
    first = _geometry(
        (
            -0.028,
            0.0,
            0.0,
        )
    )
    second = _geometry(
        (
            0.028,
            0.0,
            0.0,
        )
    )
    source_positions = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.0,
            ],
            [
                0.028,
                0.0,
                0.0,
            ],
        ]
    )
    field = _field(
        first,
        second,
        _source(
            source_positions
        ),
    )
    query = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.002,
            ],
            [
                0.0,
                0.0,
                0.030,
            ],
        ]
    )
    currents = np.asarray(
        [
            0.9
            - 0.2j
        ]
    )
    reference_steady = (
        field.steady_temperature(
            query,
            currents,
        )
    )
    reference_step = (
        field.temperature_step(
            query,
            3.0,
            currents,
        )
    )

    rng = np.random.default_rng(
        991
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [
                0.17,
                -0.11,
                0.29,
            ]
        ),
    )
    moved_first = (
        first.transformed(
            common
        )
    )
    moved_second = (
        second.transformed(
            common
        )
    )
    moved_source = _source(
        common.apply(
            source_positions
        )
    )
    moved_field = _field(
        moved_first,
        moved_second,
        moved_source,
    )
    moved_query = common.apply(
        query
    )
    actual_steady = (
        moved_field.steady_temperature(
            moved_query,
            currents,
        )
    )
    actual_step = (
        moved_field.temperature_step(
            moved_query,
            3.0,
            currents,
        )
    )
    assert np.allclose(
        actual_steady,
        reference_steady,
        rtol=2e-6,
        atol=2e-7,
    )
    assert np.allclose(
        actual_step,
        reference_step,
        rtol=2e-5,
        atol=2e-6,
    )
