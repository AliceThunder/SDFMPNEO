from types import SimpleNamespace

import numpy as np

from sdfmpneo_vnext.hybrid_dielectric import DielectricCoupledResult
from sdfmpneo_vnext.prediction import StructuredPortPrediction


def test_coupled_result_keeps_raw_solver_state_separate_from_public_observables():
    raw_impedance = np.asarray(
        [[2.0 + 3.0j, 0.1 + 0.2j], [0.1 + 0.2j, 1.0 + 2.0j]],
        dtype=complex,
    )
    corrected_impedance = np.asarray(
        [[2.2 + 3.0j, 0.15 + 0.2j], [0.15 + 0.2j, 1.3 + 2.0j]],
        dtype=complex,
    )
    channels = np.asarray(
        [
            [[1.0, 0.10 + 0.05j], [0.10 - 0.05j, 0.5]],
            [[0.8, 0.02 - 0.03j], [0.02 + 0.03j, 0.4]],
            [[0.4, 0.03 - 0.02j], [0.03 + 0.02j, 0.4]],
        ],
        dtype=complex,
    )
    prediction = StructuredPortPrediction(
        corrected_impedance,
        channels,
        ("coil:0", "coil:1", "electric_environment:aggregate"),
    )
    raw_state = SimpleNamespace(
        impedance=raw_impedance,
        normalized_residual=2e-12,
    )
    result = DielectricCoupledResult(
        mixed_result=raw_state,
        prediction=prediction,
        dielectric_dissipation_matrix=channels[-1],
        surface_density_transfer=np.zeros((0, 2), dtype=complex),
        effective_potential_matrix=np.zeros((0, 0), dtype=complex),
        raw_potential_reciprocity_defect=0.0,
        surface_residual=0.0,
        source_region_index=np.zeros(0, dtype=int),
        channel_labels=(
            "conductor:first",
            "conductor:second",
            "electric_environment:aggregate",
        ),
    )

    assert np.array_equal(result.mixed_result.impedance, raw_impedance)
    assert np.array_equal(result.impedance, corrected_impedance)
    assert np.array_equal(result.coil_dissipation_matrices(), channels[:2])
    assert np.array_equal(result.dissipation_channels(), channels)
