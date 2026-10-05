import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    DenseMixedConductorTeacher,
    HomogeneousMedium,
    MQSConfig,
    MultiDebyeMaterial,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TabulatedMaterial,
)

COPPER = ConductorMaterial(5.8e7)
CFG = MQSConfig(
    segments_per_turn=10,
    min_segments=12,
    section_degree=1,
    radial_order=3,
    angular_order=12,
    line_order=2,
)


def coil(radius, z=0.0):
    return CoilObject(
        SuperellipseSpiral(
            radius,
            radius,
            0.95,
            0.0,
            0.0,
            conductor_width=1e-3,
            conductor_thickness=1e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.0, 0.0, z]),
            ),
        ),
        COPPER,
    )


def test_mixed_dc_uses_same_finite_equations_and_matches_resistance():
    scene = Scene(
        (coil(0.025),),
        HomogeneousMedium(),
    )
    result = DenseMixedConductorTeacher(
        scene,
        0.0,
        CFG,
    ).solve()
    nseg = max(
        CFG.min_segments,
        CFG.segments_per_turn,
    )
    poly = scene.coils[0].geometry.polyline(
        nseg
    )
    expected = (
        poly.total_length
        / (
            COPPER.conductivity
            * scene.coils[
                0
            ].geometry.cross_section_area
        )
    )
    assert np.isclose(
        result.impedance[0, 0].real,
        expected,
        rtol=5e-10,
    )
    assert (
        abs(
            result.impedance[0, 0].imag
        )
        < 1e-12
    )
    assert (
        result.normalized_residual
        < 1e-10
    )
    assert (
        result.continuity_residual(
            np.array([1.0 + 0j]),
            0.0,
        )
        < 1e-10
    )


def test_mixed_reciprocity_power_and_continuity_at_frequency():
    scene = Scene(
        (
            coil(0.025),
            coil(
                0.02,
                z=0.018,
            ),
        ),
        HomogeneousMedium(),
    )
    f = 20_000.0
    result = DenseMixedConductorTeacher(
        scene,
        f,
        CFG,
    ).solve()
    assert np.allclose(
        result.impedance,
        result.impedance.T,
        rtol=2e-8,
        atol=2e-9,
    )
    currents = np.array(
        [
            1.1 - 0.3j,
            -0.4 + 0.7j,
        ]
    )
    assert np.isclose(
        result.port_power(currents),
        result.conductor_power(currents),
        rtol=2e-7,
        atol=1e-10,
    )
    assert (
        result.continuity_residual(
            currents,
            2 * np.pi * f,
        )
        < 1e-10
    )
    assert (
        result.normalized_residual
        < 1e-10
    )


def test_mixed_rejects_closed_coincident_terminals():
    closed = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.025,
            1.0,
            0.0,
            0.0,
            conductor_width=1e-3,
            conductor_thickness=1e-3,
        ),
        COPPER,
    )
    scene = Scene(
        (closed,),
        HomogeneousMedium(),
    )
    try:
        DenseMixedConductorTeacher(
            scene,
            10_000.0,
            CFG,
        ).solve()
    except ValueError as exc:
        assert (
            "terminal" in str(exc)
            or "charge nodes" in str(exc)
        )
    else:
        raise AssertionError(
            "closed coincident terminals must be rejected"
        )



def test_mixed_loss_channels_are_psd_and_close_port_dissipation():
    scene = Scene(
        (
            coil(0.025),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        HomogeneousMedium(),
    )
    result = DenseMixedConductorTeacher(
        scene,
        20_000.0,
        CFG,
    ).solve()
    channels = result.coil_dissipation_matrices()
    assert channels.shape == (
        2,
        2,
        2,
    )
    for channel in channels:
        assert np.allclose(
            channel,
            channel.conj().T,
            atol=2e-9,
        )
        assert (
            np.min(
                np.linalg.eigvalsh(channel)
            )
            >= -2e-9
        )
    total = np.sum(
        channels,
        axis=0,
    )
    physical = 0.5 * (
        result.impedance
        + result.impedance.conj().T
    )
    assert np.allclose(
        total,
        physical,
        rtol=2e-6,
        atol=2e-8,
    )
    currents = np.array(
        [1.0 + 0.2j, -0.4 + 0.3j]
    )
    assert np.isclose(
        np.sum(
            result.coil_power(currents)
        ),
        result.conductor_power(currents),
        rtol=2e-7,
        atol=1e-10,
    )


def test_mixed_local_dissipation_matrix_is_psd():
    scene = Scene(
        (coil(0.025),),
        HomogeneousMedium(),
    )
    teacher = DenseMixedConductorTeacher(
        scene,
        10_000.0,
        CFG,
    )
    result = teacher.solve()
    matrix = teacher.local_dissipation_matrix(
        result,
        0,
        0.5,
        (0.0, 0.0),
    )
    assert np.allclose(
        matrix,
        matrix.conj().T,
        atol=1e-10,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(matrix)
        )
        >= -1e-10
    )



def test_mixed_lossy_background_adds_passive_environment_channel_and_closes_power():
    scene = Scene(
        (
            coil(0.025),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        HomogeneousMedium(
            relative_permittivity=3.0,
            relative_permeability=1.0,
            conductivity=1e-4,
        ),
    )
    result = DenseMixedConductorTeacher(
        scene,
        40_000.0,
        CFG,
    ).solve()
    channels = result.dissipation_channels()
    assert channels.shape == (
        3,
        2,
        2,
    )
    background = channels[-1]
    assert np.allclose(
        background,
        background.conj().T,
        atol=2e-9,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                background
            )
        )
        >= -2e-9
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        0.5
        * (
            result.impedance
            + result.impedance.conj().T
        ),
        rtol=3e-6,
        atol=3e-8,
    )
    currents = np.array(
        [
            1.0 + 0.2j,
            -0.4 + 0.3j,
        ]
    )
    assert result.background_power(
        currents
    ) >= -1e-10
    assert np.isclose(
        result.port_power(
            currents
        ),
        result.dissipated_power(
            currents
        ),
        rtol=3e-6,
        atol=3e-9,
    )
    assert (
        result.port_power(
            currents
        )
        >= result.conductor_power(
            currents
        )
        - 1e-10
    )


def test_mixed_debye_background_adds_frequency_loss_channel_without_dc_conductivity():
    frequency = 80_000.0
    medium = DebyeMaterial(
        relative_permittivity_static=30.0,
        relative_permittivity_infinite=5.0,
        relaxation_time=2.0e-6,
        relative_permeability=1.0,
        conductivity=0.0,
    )
    assert (
        medium.conductivity
        == 0.0
    )
    assert (
        medium.loss_conductivity(
            frequency
        )
        > 0.0
    )
    scene = Scene(
        (
            coil(
                0.025
            ),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        medium,
    )
    result = DenseMixedConductorTeacher(
        scene,
        frequency,
        CFG,
    ).solve()
    channels = (
        result.dissipation_channels()
    )
    assert channels.shape == (
        3,
        2,
        2,
    )
    background = channels[
        -1
    ]
    assert np.allclose(
        background,
        background.conj().T,
        atol=3e-9,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                background
            )
        )
        >= -3e-9
    )
    assert (
        np.linalg.norm(
            background
        )
        > 0.0
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        0.5
        * (
            result.impedance
            + result.impedance.conj().T
        ),
        rtol=5e-6,
        atol=5e-8,
    )


def test_multi_debye_background_is_passive_and_closes_mixed_power():
    frequency = 120_000.0
    medium = MultiDebyeMaterial(
        relative_permittivity_infinite=3.0,
        relaxation_strengths=(
            4.0,
            10.0,
        ),
        relaxation_times=(
            2.0e-7,
            5.0e-6,
        ),
        conductivity=0.0,
    )
    assert (
        medium.loss_conductivity(
            frequency
        )
        > 0.0
    )
    scene = Scene(
        (
            coil(
                0.025
            ),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        medium,
    )
    result = DenseMixedConductorTeacher(
        scene,
        frequency,
        CFG,
    ).solve()
    channels = result.dissipation_channels()
    assert channels.shape == (
        3,
        2,
        2,
    )
    assert (
        np.linalg.norm(
            channels[
                -1
            ]
        )
        > 0.0
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        0.5
        * (
            result.impedance
            + result.impedance.conj().T
        ),
        rtol=6e-6,
        atol=6e-8,
    )


class _CustomBackgroundMaterial:
    def __init__(
        self,
    ):
        self._delegate = MultiDebyeMaterial(
            relative_permittivity_infinite=2.5,
            relaxation_strengths=(
                3.0,
                7.0,
            ),
            relaxation_times=(
                3.0e-7,
                4.0e-6,
            ),
            conductivity=0.0,
        )
        self.relative_permeability = (
            self._delegate.relative_permeability
        )
        self.conductivity = (
            self._delegate.conductivity
        )

    @property
    def permeability(
        self,
    ):
        return self._delegate.permeability

    def complex_permittivity(
        self,
        frequency_hz,
    ):
        return self._delegate.complex_permittivity(
            frequency_hz
        )

    def relative_permittivity_at(
        self,
        frequency_hz,
    ):
        return self._delegate.relative_permittivity_at(
            frequency_hz
        )

    def loss_conductivity(
        self,
        frequency_hz,
    ):
        return self._delegate.loss_conductivity(
            frequency_hz
        )


def test_custom_material_response_protocol_runs_mixed_reference():
    frequency = 90_000.0
    medium = _CustomBackgroundMaterial()
    scene = Scene(
        (
            coil(
                0.025
            ),
        ),
        medium,
    )
    result = DenseMixedConductorTeacher(
        scene,
        frequency,
        CFG,
    ).solve()
    assert (
        result.background_dissipation_matrix
        is not None
    )
    assert (
        np.linalg.norm(
            result.background_dissipation_matrix
        )
        > 0.0
    )
    assert np.allclose(
        np.sum(
            result.dissipation_channels(),
            axis=0,
        ),
        0.5
        * (
            result.impedance
            + result.impedance.conj().T
        ),
        rtol=6e-6,
        atol=6e-8,
    )


def test_tabulated_background_runs_mixed_reference_inside_declared_frequency_table():
    frequency = 100_000.0
    medium = TabulatedMaterial(
        frequencies_hz=(
            20_000.0,
            100_000.0,
            500_000.0,
        ),
        relative_permittivity_real=(
            10.0,
            7.0,
            4.0,
        ),
        loss_conductivity_values=(
            1.0e-5,
            2.5e-4,
            1.0e-4,
        ),
    )
    scene = Scene(
        (
            coil(
                0.025
            ),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        medium,
    )
    result = DenseMixedConductorTeacher(
        scene,
        frequency,
        CFG,
    ).solve()
    channels = result.dissipation_channels()
    assert channels.shape == (
        3,
        2,
        2,
    )
    assert (
        np.linalg.norm(
            channels[
                -1
            ]
        )
        > 0.0
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        0.5
        * (
            result.impedance
            + result.impedance.conj().T
        ),
        rtol=6e-6,
        atol=6e-8,
    )
    assert (
        medium.loss_conductivity(
            0.0
        )
        == 0.0
    )
    try:
        medium.relative_permittivity_at(
            1_000_000.0
        )
    except ValueError as exc:
        assert (
            "outside the tabulated material domain"
            in str(
                exc
            )
        )
    else:
        raise AssertionError(
            "tabulated material must not extrapolate outside its frequency domain"
        )


def test_conductive_homogeneous_background_runs_true_dc_environment_current_formulation():
    scene = Scene(
        (
            coil(
                0.025
            ),
            coil(
                0.020,
                z=0.018,
            ),
        ),
        HomogeneousMedium(
            relative_permittivity=3.0,
            relative_permeability=1.0,
            conductivity=2.0e-3,
        ),
    )
    result = DenseMixedConductorTeacher(
        scene,
        0.0,
        CFG,
    ).solve()

    assert (
        result.node_environment_current
        is not None
    )
    assert (
        np.linalg.norm(
            result.node_environment_current
        )
        > 0.0
    )
    assert np.allclose(
        result.node_charge,
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        result.background_dissipation_matrix
        is not None
    )
    currents = np.asarray(
        [
            1.0 + 0.0j,
            -0.35 + 0.0j,
        ]
    )
    assert (
        result.continuity_residual(
            currents,
            0.0,
        )
        < 1e-9
    )
    channels = result.dissipation_channels()
    total = 0.5 * (
        result.impedance
        + result.impedance.conj().T
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        total,
        rtol=8e-6,
        atol=8e-8,
    )
    for channel in channels:
        assert (
            np.min(
                np.linalg.eigvalsh(
                    0.5
                    * (
                        channel
                        + channel.conj().T
                    )
                )
            )
            >= -2e-8
        )
    assert (
        np.linalg.norm(
            result.impedance.imag
        )
        < 1e-10
    )
