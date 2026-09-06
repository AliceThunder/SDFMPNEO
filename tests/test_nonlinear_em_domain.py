import numpy as np

from sdfmpneo.certification import (
    ParameterBox,
    bound_nonlinear_reduced_residual_on_box,
    certify_nonlinear_reduced_residual_domain,
)
from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseEnergyResidualGreedyEMReducer,
    make_morse_auxiliary_physical_pcg_riesz_factory,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def _problem():
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [0.25, 0.25, 0.25],
    ])
    tetrahedra = np.array([
        [4, 1, 2, 3], [0, 4, 2, 3], [0, 1, 4, 3], [0, 1, 2, 4],
    ])
    mesh = TetrahedralComplex3D.build(vertices, tetrahedra)
    copper = np.array([True, True, False, False])
    regions = (
        ConductivityRegion(
            "copper", copper,
            ReciprocalLinearResistivity(5.8e7, 3.93e-3, 293.15),
        ),
        ConductivityRegion("seawater", ~copper, ConstantConductivity(5.0)),
    )
    reference = np.full((mesh.n_tetrahedra, 4), 293.15)
    modes = np.zeros((1, mesh.n_tetrahedra, 4))
    modes[0] = np.array([0.2, 0.4, 0.1, 0.3])
    return NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.full(mesh.n_tetrahedra, 1.0 / (4.0e-7 * np.pi)),
        source_current=tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0])),
        temperature_reference_local=reference,
        thermal_modes_local=modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-10,
    )


def test_nonlinear_box_bound_is_finite_and_shrinks_with_box():
    problem = _problem()
    reducer = SparseEnergyResidualGreedyEMReducer(
        problem,
        riesz_action_factory=make_morse_auxiliary_physical_pcg_riesz_factory(problem),
    )
    model = reducer.build(
        [np.array([-4.0]), np.array([0.0]), np.array([4.0])],
        requested_energy_state_error=1e-5,
    )
    wide = bound_nonlinear_reduced_residual_on_box(
        model, ParameterBox(np.array([-4.0]), np.array([4.0]))
    )
    narrow = bound_nonlinear_reduced_residual_on_box(
        model, ParameterBox(np.array([-1.0]), np.array([1.0]))
    )
    assert np.isfinite(wide.residual_upper_bound)
    assert narrow.residual_upper_bound <= wide.residual_upper_bound
    assert wide.reduced_stability_lower_bound > 0.0


def test_nonlinear_domain_returns_only_proved_statuses():
    problem = _problem()
    reducer = SparseEnergyResidualGreedyEMReducer(
        problem,
        riesz_action_factory=make_morse_auxiliary_physical_pcg_riesz_factory(problem),
    )
    model = reducer.build(
        [np.array([-2.0]), np.array([0.0]), np.array([2.0])],
        requested_energy_state_error=1e-4,
    )
    cert = certify_nonlinear_reduced_residual_domain(
        model,
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
        tolerance=1.0,
        work_budget=32,
    )
    assert cert.status in {"certified", "violated", "indeterminate"}
    if cert.certified:
        assert cert.global_upper_bound <= cert.tolerance
