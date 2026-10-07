from types import SimpleNamespace

import numpy as np

import sdfmpneo_vnext.tensor_spatial_reference as tensor_spatial_reference
from sdfmpneo_vnext.prediction import StructuredPortPrediction


def test_reciprocalized_integrated_joule_matrix_remains_psd():
    transfer = np.asarray(
        [
            [1.0 + 0.4j, 0.2 - 0.7j],
            [-0.3 + 0.8j, 0.9 + 0.1j],
            [0.5 - 0.2j, -0.6 + 0.3j],
        ],
        dtype=complex,
    )
    raw = transfer.conj().T @ transfer

    projected = tensor_spatial_reference.reciprocalize_dissipation_matrices(raw)

    assert np.allclose(projected, projected.T, rtol=0.0, atol=0.0)
    assert np.max(np.abs(projected.imag)) == 0.0
    assert np.min(np.linalg.eigvalsh(projected.real)) >= -1e-12


def test_common_energy_congruence_preserves_local_complex_structure():
    local = np.asarray(
        [[2.0, 0.2 + 0.4j], [0.2 - 0.4j, 1.2]],
        dtype=complex,
    )
    raw_total = np.asarray(
        [[4.0, 0.5 + 0.3j], [0.5 - 0.3j, 2.5]],
        dtype=complex,
    )
    target = tensor_spatial_reference.reciprocalize_dissipation_matrices(
        raw_total
    )
    transform = tensor_spatial_reference._energy_congruence(
        raw_total,
        target,
    )

    corrected_total = tensor_spatial_reference._apply_energy_transform(
        raw_total,
        transform,
    )
    corrected_local = tensor_spatial_reference._apply_energy_transform(
        local,
        transform,
    )

    assert np.allclose(corrected_total, target, rtol=1e-11, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(corrected_local)) >= -1e-12
    assert abs(corrected_local[0, 1].imag) > 1e-6


def test_tensor_energy_truth_replaces_structurally_mismatched_port_loss(monkeypatch):
    old_environment = np.asarray(
        [[1.0e-12, 0.0], [0.0, 2.0e-12]],
        dtype=complex,
    )
    conductor = np.asarray(
        [[[2.0, 0.1 + 0.2j], [0.1 - 0.2j, 1.0]]],
        dtype=complex,
    )
    old_channels = np.concatenate(
        (conductor, old_environment[None, :, :]),
        axis=0,
    )
    old_impedance = np.asarray(
        [[2.1 + 4.0j, 0.2 + 0.5j], [0.2 + 0.5j, 1.2 + 3.0j]],
        dtype=complex,
    )
    prediction = StructuredPortPrediction(
        old_impedance,
        old_channels,
        ("coil:0", "electric_environment:aggregate"),
    )

    mixed_result = SimpleNamespace(
        coil_dissipation_matrices=lambda: conductor,
    )
    result = SimpleNamespace(
        impedance=old_impedance,
        prediction=prediction,
        dielectric_dissipation_matrix=old_environment,
        tensor_electric_transmission=object(),
        mixed_result=mixed_result,
    )
    scene = SimpleNamespace(
        packages=(object(),),
        coils=(object(), object()),
    )
    teacher = SimpleNamespace(
        scene=scene,
        frequency_hz=100_000.0,
    )

    package_channel = np.asarray(
        [[[1.5e-3, 2.0e-4 + 1.0e-4j], [2.0e-4 - 1.0e-4j, 8.0e-4]]],
        dtype=complex,
    )
    background_channel = np.asarray(
        [[3.0e-4, 1.0e-4 - 0.5e-4j], [1.0e-4 + 0.5e-4j, 2.0e-4]],
        dtype=complex,
    )
    environment = package_channel[0] + background_channel

    monkeypatch.setattr(
        tensor_spatial_reference,
        "_integrated_tensor_environment",
        lambda *args, **kwargs: (
            package_channel,
            background_channel,
            environment,
        ),
    )

    calibration = tensor_spatial_reference.prepare_tensor_spatial_reference_adaptive(
        teacher,
        result,
        volume_axial_order=6,
        volume_radial_order=4,
        volume_azimuthal_order=16,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=0.35,
        maximum_quadrature_refinements=4,
    )

    prediction = StructuredPortPrediction(
        calibration.target_impedance,
        calibration.target_dissipation_channels,
        ("coil:0", "electric_environment:aggregate"),
    )
    assert calibration.raw_closure_target_exceeded
    assert calibration.power_closure_error < 1e-8
    assert prediction.power_closure_error() < 1e-8
    assert prediction.reciprocity_defect() < 1e-12
    assert np.min(
        np.linalg.eigvalsh(calibration.target_dissipation_channels[-1])
    ) >= -1e-12
    assert np.max(
        np.abs(calibration.target_dissipation_channels[-1].imag)
    ) > 0.0
