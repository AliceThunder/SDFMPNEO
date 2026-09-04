import pytest
import torch

from sdfmpneo.contracts import TopologyOperators
from sdfmpneo.topology import validate_topology


def _valid_topology() -> TopologyOperators:
    current_exact = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    current_harmonic = torch.tensor(
        [[1.0], [-1.0], [0.0], [1.0]], dtype=torch.float64
    )
    current_div = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float64)

    velocity_exact = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )
    velocity_harmonic = torch.tensor(
        [[1.0], [-1.0], [0.0], [1.0], [0.0]], dtype=torch.float64
    )
    velocity_div = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    return TopologyOperators(
        curl_current=current_exact,
        harmonic_current=current_harmonic,
        curl_velocity=velocity_exact,
        harmonic_velocity=velocity_harmonic,
        divergence_current=current_div,
        divergence_velocity=velocity_div,
    )


def test_valid_gauge_reduced_topology_passes() -> None:
    validation = validate_topology(_valid_topology())
    assert validation.current_exact_defect < 1.0e-12
    assert validation.current_harmonic_defect < 1.0e-12
    assert validation.velocity_exact_defect < 1.0e-12
    assert validation.velocity_harmonic_defect < 1.0e-12
    assert validation.current_generator_rank == 3
    assert validation.velocity_generator_rank == 3


def test_missing_divergence_is_rejected_in_production_validation() -> None:
    topology = _valid_topology()
    topology.divergence_current = None
    with pytest.raises(ValueError, match="requires current and velocity divergence"):
        validate_topology(topology)


def test_non_solenoidal_generator_is_rejected() -> None:
    topology = _valid_topology()
    topology.curl_current = topology.curl_current.clone()
    topology.curl_current[0, 0] += 0.25
    with pytest.raises(ValueError, match="violates solenoidal"):
        validate_topology(topology)


def test_gauge_redundant_generator_is_rejected() -> None:
    topology = _valid_topology()
    topology.curl_current = torch.cat(
        (topology.curl_current, topology.curl_current[:, :1]), dim=-1
    )
    with pytest.raises(ValueError, match="column-rank deficient"):
        validate_topology(topology)
