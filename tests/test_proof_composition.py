import numpy as np
import pytest

from sdfmpneo.certification import (
    ElectroThermalDomainBounds,
    GeometryDerivativeProof,
    ParameterBox,
    ProofNode,
    compose_electrothermal_error_certificate,
    compose_geometry_physical_lipschitz_proof,
    compose_typed_error_certificate,
)


def test_typed_composer_rejects_incompatible_physical_quantities():
    nodes = {
        "field": ProofNode(
            "field", "em_field", "energy", 1.0, certified=True, provenance="field theorem"
        ),
        "thermal": ProofNode(
            "thermal", "thermal_state", "modal_l2", 2.0, certified=True, provenance="thermal theorem"
        ),
    }
    with pytest.raises(ValueError, match="common accumulation"):
        compose_typed_error_certificate(nodes)


def test_electrothermal_composer_uses_finite_time_residual_gain_before_state_addition():
    residual = {
        "analytic": ProofNode(
            "analytic",
            "thermal_residual",
            "modal_l2",
            2.0,
            certified=True,
            provenance="continuous residual proof",
        )
    }
    state = {
        "thermal_rom": ProofNode(
            "thermal_rom",
            "thermal_state",
            "modal_l2",
            0.25,
            certified=True,
            provenance="spectral tail proof",
        )
    }
    certificate = compose_electrothermal_error_certificate(
        residual,
        kappa=0.0,
        time_horizon=3.0,
        state_nodes=state,
        output_gain=4.0,
    )
    assert certificate.certified
    assert certificate.residual_to_state_gain == 3.0
    assert certificate.propagated_residual_state_error_bound >= 6.0
    assert certificate.total_state_error_bound >= 6.25
    assert certificate.output_error_bound >= 25.0


def test_geometry_proof_composer_uses_full_vector_field_state_bound():
    tbox = ParameterBox(np.array([-1.0]), np.array([1.0]))
    ubox = ParameterBox(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    electrothermal = ElectroThermalDomainBounds(
        thermal_box=tbox,
        operating_box=ubox,
        source_dual_energy_bound=2.0,
        heat_source_jacobian_entry_bounds=np.array([[0.2]]),
        heat_source_jacobian_norm_bound=0.2,
        vector_field_state_jacobian_norm_bound=5.2,
        explicit_operating_jacobian_entry_bounds=np.array([[0.3, 0.4]]),
        explicit_operating_jacobian_norm_bound=0.5,
        contraction_margin_lower_bound=0.8,
    )
    geometry = GeometryDerivativeProof(
        np.array([[0.6, 0.7]]),
        certified=True,
        provenance="affine chart derivative theorem",
    )
    proof = compose_geometry_physical_lipschitz_proof(
        electrothermal,
        geometry,
        provenance="composed continuous-domain proof",
    )
    assert proof.certified
    assert proof.state_jacobian_norm_bound == 5.2
    assert np.array_equal(proof.operating_jacobian_entry_bounds, np.array([[0.3, 0.4]]))
    assert np.array_equal(proof.geometry_jacobian_entry_bounds, np.array([[0.6, 0.7]]))
    assert proof.contraction_margin_lower_bound == 0.8
