import numpy as np

from sdfmpneo import (
    GeometryResearchModel,
    ResearchElectroThermalModel,
    ResearchTrainingConfig,
    ResearchTrainingContinuation,
)
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.geometry_research import GeometryResearchModel
from sdfmpneo.research import demo_research_model
from sdfmpneo.spatial import TaggedTetrahedralMesh
from sdfmpneo.spatial.geometry_chart import AffineTetrahedralGeometryChart
from sdfmpneo.training.adaptive_runtime import (
    _config_signature,
    _select_adaptive_validation_points,
)


class _Record:
    def __init__(self, residual):
        self.residual = np.asarray(residual, float)


def test_selective_refinement_promotes_only_actual_tolerance_violations():
    points = np.arange(20.0).reshape(4, 5)
    records = [
        _Record([1e-6, 0.0]),
        _Record([2e-5, 0.0]),
        _Record([0.0, 8e-6]),
        _Record([3e-5, 4e-5]),
    ]
    violating, guards, norms = _select_adaptive_validation_points(
        records, points, tolerance=1e-5
    )
    assert np.array_equal(violating, points[[1, 3]])
    assert np.array_equal(guards, points[[0, 2]])
    assert np.allclose(norms, [1e-6, 2e-5, 8e-6, 5e-5])


def test_continuation_signature_allows_only_node_budget_to_change():
    common = dict(
        initial_lower=(0.0,), initial_upper=(1.0,),
        operating_lower=(0.0,), operating_upper=(2.0,),
        time_horizon=1.0, residual_tolerance=1e-5,
        sample_count=8, validation_count=8,
        max_degree=2,
    )
    a = ResearchTrainingConfig(max_nodes=16, **common)
    b = ResearchTrainingConfig(max_nodes=64, **common)
    c = ResearchTrainingConfig(max_nodes=64, time_min=1e-4,
                               time_sampling="mixed_log", **common)
    assert _config_signature(a) == _config_signature(b)
    assert _config_signature(a) != _config_signature(c)


def _small_geometry_family():
    ref = demo_research_model()
    mesh = ref.core.mesh
    faces = mesh.face_vertices[mesh.boundary_face_indices]
    tags = np.arange(1, len(faces) + 1)
    tagged = TaggedTetrahedralMesh(mesh, np.arange(mesh.n_tetrahedra), faces, tags)
    chart = AffineTetrahedralGeometryChart(
        mesh.vertices, mesh.tetrahedra, mesh.vertices[None, :, :], ("scale",)
    )
    model = GeometryResearchModel(
        ref, tagged, chart, [1.0], [0.8], [1.2],
        np.array([3.45e6, 3.45e6, 4.1e6, 4.1e6]),
        np.array([400.0, 400.0, 0.6, 0.6]),
        [(1, 2), (3, 4)], np.zeros(2), np.eye(2),
    )
    model.build_joint_em_basis([[0.0], [2.0]], requested_error=1e-7, anchor_count=1)
    graph = ParametricAnalyticEvolutionGraph(
        model.thermal_model.lambdas, ["scale", "current_0", "current_1"]
    )
    graph.add_product_response("seed", 0, ["current_0", "current_0"], 0.01)
    model.graph = graph
    model.training_config = model.training_domain(
        ResearchTrainingConfig(
            (0.0,), (2.0,), (0.0, 0.0), (30.0, 30.0),
            1.0, 1e-5, sample_count=4, validation_count=4, max_nodes=8,
        )
    )
    return model


def test_geometry_checkpoint_roundtrips_adaptive_collocation_state(tmp_path):
    model = _small_geometry_family()
    cfg = model.training_config
    points = cfg.points()
    checks = cfg.points(validation=True)
    guards = checks[:2]
    model.training_continuation = ResearchTrainingContinuation(
        points,
        checks[2:],
        guards,
        3,
        _config_signature(cfg),
    )
    path = model.save(tmp_path / "continued.npz")
    loaded = ResearchElectroThermalModel.load(path)
    state = loaded.training_continuation
    assert state.collocation_epoch == 3
    assert state.config_signature == _config_signature(cfg)
    assert np.array_equal(state.points, points)
    assert np.array_equal(state.checks, checks[2:])
    assert np.array_equal(state.guards, guards)
