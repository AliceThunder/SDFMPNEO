import numpy as np
import pytest

from sdfmpneo_vnext import (
    ChannelResolvedCurrentEnvelope,
    ChannelResolvedVoltageEnvelope,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    Scene,
    StructuredPortPrediction,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    ThermalNodeProperties,
    build_lumped_channel_thermal_model,
)


class _TwoChannelArtifact:
    def __init__(self):
        self.calls = 0

    def predict_structured(
        self,
        scene,
        frequency_hz,
    ):
        self.calls += 1
        assert len(
            scene.packages
        ) == 1
        reference_sigma = 5.8e7
        resistance = (
            0.20
            * reference_sigma
            / scene.coils[
                0
            ].material.conductivity
        )
        impedance = np.array(
            [
                [
                    resistance
                    + 0.15j
                ]
            ],
            dtype=complex,
        )
        channels = np.array(
            [
                [
                    [
                        0.75
                        * resistance
                    ]
                ],
                [
                    [
                        0.25
                        * resistance
                    ]
                ],
            ],
            dtype=complex,
        )
        return StructuredPortPrediction(
            impedance,
            channels,
        )


def _scene():
    coil = CoilObject(
        SuperellipseSpiral(
            0.02,
            0.018,
            0.7,
            0.001,
            0.001,
            conductor_width=8e-4,
            conductor_thickness=6e-4,
        ),
        ConductorMaterial(
            5.8e7,
            resistance_temperature_coefficient=0.00393,
        ),
        "coil",
    )
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.03, 0.026, 0.008]
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=3.0,
            thermal_conductivity=0.25,
            density=1200.0,
            heat_capacity=1400.0,
        ),
        "package",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
        (package,),
    )


def _thermal():
    return build_lumped_channel_thermal_model(
        (
            ThermalNodeProperties(
                6.0,
                0.10,
            ),
            ThermalNodeProperties(
                12.0,
                0.06,
            ),
        ),
        mutual_conductance=np.array(
            [
                [0.0, 0.03],
                [0.03, 0.0],
            ]
        ),
        ambient_temperature=293.15,
    )


def test_channel_resolved_current_envelope_separates_ports_channels_and_states():
    artifact = _TwoChannelArtifact()
    envelope = ChannelResolvedCurrentEnvelope(
        _scene(),
        40_000.0,
        _thermal(),
        artifact,
        coil_temperature_indices=np.array(
            [0]
        ),
        coupling_tolerance=1e-9,
        max_coupling_iterations=20,
    )
    cold = envelope.step(
        np.zeros(2),
        np.array(
            [2.0 + 0j]
        ),
        0.0,
    )
    hot = envelope.step(
        np.zeros(2),
        np.array(
            [2.0 + 0j]
        ),
        20.0,
    )
    assert hot.converged
    assert hot.channel_power.shape == (
        2,
    )
    assert np.all(
        hot.channel_power
        > 0.0
    )
    assert hot.temperatures.shape == (
        2,
    )
    assert np.all(
        hot.temperatures
        > 293.15
    )
    assert (
        hot.impedance[
            0,
            0,
        ].real
        > cold.impedance[
            0,
            0,
        ].real
    )
    assert artifact.calls > 2


def test_channel_resolved_voltage_envelope_reduces_current_when_coil_heats():
    envelope = ChannelResolvedVoltageEnvelope(
        _scene(),
        40_000.0,
        _thermal(),
        _TwoChannelArtifact(),
        external_impedance=np.array(
            [[0.05 + 0j]]
        ),
        coil_temperature_indices=np.array(
            [0]
        ),
        coupling_tolerance=1e-9,
        max_coupling_iterations=20,
    )
    cold = envelope.step(
        np.zeros(2),
        np.array(
            [1.0 + 0j]
        ),
        0.0,
    )
    hot = envelope.step(
        np.zeros(2),
        np.array(
            [1.0 + 0j]
        ),
        30.0,
    )
    assert hot.converged
    assert (
        abs(
            hot.currents[
                0
            ]
        )
        < abs(
            cold.currents[
                0
            ]
        )
    )


def test_channel_resolved_envelope_rejects_thermal_source_channel_mismatch():
    thermal = build_lumped_channel_thermal_model(
        (
            ThermalNodeProperties(
                5.0,
                0.1,
            ),
        ),
    )
    envelope = ChannelResolvedCurrentEnvelope(
        _scene(),
        40_000.0,
        thermal,
        _TwoChannelArtifact(),
        coil_temperature_indices=np.array(
            [0]
        ),
    )
    with pytest.raises(
        ValueError,
        match="channel count",
    ):
        envelope.step(
            np.zeros(1),
            np.array(
                [1.0 + 0j]
            ),
            1.0,
        )
