import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.certification import (
    ParameterBox,
    certify_geometry_joule_derivative_bounds,
    certify_geometry_thermal_operator_bounds,
    certify_trained_geometry_model_finite_domain,
    make_uwpt_mass_residual_physical_proof_factory,
)
from sdfmpneo.geometry_research import GeometryResearchModel
from sdfmpneo.research import demo_research_model
from sdfmpneo.spatial import TaggedTetrahedralMesh
from sdfmpneo.spatial.geometry_chart import AffineTetrahedralGeometryChart
from sdfmpneo.training.research import ResearchTrainingConfig


def _family():
    ref = demo_research_model()
    mesh = ref.core.mesh
    faces = mesh.face_vertices[mesh.boundary_face_indices]
    tags = np.arange(1, len(faces) + 1)
    tagged = TaggedTetrahedralMesh(mesh, np.arange(mesh.n_tetrahedra), faces, tags)
    chart = AffineTetrahedralGeometryChart(
        mesh.vertices, mesh.tetrahedra, mesh.vertices[None, :, :], ("scale",)
    )
    model = GeometryResearchModel(
        ref,
        tagged,
        chart,
        [1.0],
        [0.8],
        [1.2],
        np.array([3.45e6, 3.45e6, 4.1e6, 4.1e6]),
        np.array([400.0, 400.0, 0.6, 0.6]),
        [(1, 2), (3, 4)],
        np.zeros(2),
        np.eye(2),
    )
    model.build_joint_em_basis([[0.0], [2.0]], requested_error=1e-7, anchor_count=1)
    model.graph = ParametricAnalyticEvolutionGraph(
        model.thermal_model.lambdas, ["scale", "current_0", "current_1"]
    )
    base = ResearchTrainingConfig(
        (0.0,), (0.2,), (0.0, 0.0), (3.0, 3.0), 1.0, 1e-5
    )
    model.training_config = model.training_domain(base)
    return model


def test_thermal_provider_contains_exact_uniform_scale_derivatives():
    model = _family()
    # Public box layout [a0, physical scale, current0, current1, time].
    box = ParameterBox(
        np.array([0.05, 0.9, 0.0, 0.0, 0.1]),
        np.array([0.15, 1.1, 0.0, 0.0, 0.2]),
    )
    proof = certify_geometry_thermal_operator_bounds(model, box)
    assert proof.certified
    assert proof.minimum_mass_eigenvalue > 0.0

    # For this chart M_r=s^3 and K_r=s*lambda exactly.
    for scale in (0.9, 1.0, 1.1):
        exact_dm = 3.0 * scale**2
        exact_dk = float(model.thermal_model.lambdas[0])
        assert exact_dm <= proof.mass_geometry_derivative_norm_bounds[0]
        assert exact_dk <= proof.stiffness_geometry_derivative_norm_bounds[0]
        context = model.context([scale])
        assert np.linalg.norm(context.M, 2) <= proof.mass_matrix_norm_bound
        assert np.linalg.norm(context.K, 2) <= proof.stiffness_matrix_norm_bound
        assert np.linalg.eigvalsh(context.M)[0] >= proof.minimum_mass_eigenvalue


def test_joule_provider_is_certified_and_dominates_local_physical_derivatives():
    model = _family()
    box = ParameterBox(
        np.array([0.08, 0.98, 1.8, 0.8, 0.19]),
        np.array([0.12, 1.02, 2.2, 1.2, 0.21]),
    )
    proof = certify_geometry_joule_derivative_bounds(model, box)
    assert proof.certified
    assert np.isfinite(proof.state_jacobian_norm_bound)
    assert np.all(np.isfinite(proof.geometry_jacobian_column_norm_bounds))
    assert np.all(np.isfinite(proof.operating_jacobian_column_norm_bounds))

    state = np.array([0.1])
    geometry = np.array([1.0])
    current = np.array([2.0, 1.0])
    context = model.context(geometry)
    rhs = context.rhs.evaluate(current)
    _, Jq = context.em.heat_source_and_jacobian_for_rhs(state, rhs)
    assert np.linalg.norm(Jq, 2) <= proof.state_jacobian_norm_bound * (1.0 + 1e-10)

    # Regression-only numerical probes: the certificate itself never uses finite
    # differences or samples.  These checks catch missing chain-rule/source terms.
    h = 1e-6
    plus = model.context([geometry[0] + h])
    minus = model.context([geometry[0] - h])
    q_plus = plus.em.heat_source_for_rhs(state, plus.rhs.evaluate(current))
    q_minus = minus.em.heat_source_for_rhs(state, minus.rhs.evaluate(current))
    d_geometry = np.linalg.norm((q_plus - q_minus) / (2.0 * h))
    assert d_geometry <= proof.geometry_jacobian_column_norm_bounds[0] * (1.0 + 1e-7)

    for k in range(2):
        direction = np.zeros(2)
        direction[k] = h
        q_plus = context.em.heat_source_for_rhs(state, context.rhs.evaluate(current + direction))
        q_minus = context.em.heat_source_for_rhs(state, context.rhs.evaluate(current - direction))
        derivative = np.linalg.norm((q_plus - q_minus) / (2.0 * h))
        assert derivative <= proof.operating_jacobian_column_norm_bounds[k] * (1.0 + 1e-7)


def test_default_proof_factory_closes_zero_drive_continuous_geometry_domain():
    model = _family()
    # The homogeneous zero state is exact for every geometry and every finite
    # time, so the theorem providers should let the root continuous box close.
    model.training_config = model.training_domain(
        ResearchTrainingConfig((0.0,), (0.0,), (0.0, 0.0), (0.0, 0.0), 1.0, 1e-5)
    )
    proof_factory = make_uwpt_mass_residual_physical_proof_factory(model)
    physical_box = ParameterBox(
        np.array([0.0, 0.8, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.2, 0.0, 0.0, 1.0]),
    )
    proof = proof_factory(physical_box)
    assert proof.certified

    report = certify_trained_geometry_model_finite_domain(model, work_budget=1)
    assert report.status == "certified"
    assert report.certified
    assert report.unresolved_boxes == 0
    assert report.maximum_vector_residual_bound <= 1e-5
