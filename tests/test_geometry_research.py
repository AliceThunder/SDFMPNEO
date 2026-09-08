"""Physics and persistence regressions for a single geometry-conditioned DAG."""
import numpy as np
import pytest

from sdfmpneo.research import demo_research_model, ResearchElectroThermalModel
from sdfmpneo.geometry_research import GeometryResearchModel
from sdfmpneo.spatial import TaggedTetrahedralMesh
from sdfmpneo.spatial.geometry_chart import AffineTetrahedralGeometryChart
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from sdfmpneo.training.research import ResearchTrainingConfig, train_research_graph
from sdfmpneo.certification import (
    certify_geometry_mass_residual_equivalence,
    local_geometry_dynamics_diagnostic,
    validate_geometry_trajectory,
)


@pytest.fixture(scope='module')
def family():
    ref = demo_research_model()
    mesh = ref.core.mesh
    faces = mesh.face_vertices[mesh.boundary_face_indices]
    tags = np.arange(1,len(faces)+1)
    tagged = TaggedTetrahedralMesh(mesh, np.arange(mesh.n_tetrahedra), faces, tags)
    chart = AffineTetrahedralGeometryChart(mesh.vertices, mesh.tetrahedra,
                                            mesh.vertices[None,:,:], ('scale',))
    out = GeometryResearchModel(ref, tagged, chart, [1.], [.8], [1.2],
        np.array([3.45e6,3.45e6,4.1e6,4.1e6]), np.array([400.,400.,.6,.6]),
        [(1,2),(3,4)], np.zeros(2), np.eye(2))
    out.build_joint_em_basis([[0.],[2.]], requested_error=1e-7, anchor_count=1)
    return out


def test_geometry_uses_mass_and_stiffness_pullback_and_physical_jacobian(family):
    m = family
    s=1.137  # unseen by the EM anchor set
    c=m.context([s])
    assert np.allclose(c.M, s**3*np.eye(1))
    assert np.allclose(c.K, s*m.thermal_model.lambdas.reshape(1,1))
    assert c.em.V is m.reference.em.V
    a=np.array([.31]);static=np.r_[m.normalize([s]),[21.,8.]]
    F=m.vector_field(a,static)
    q=c.em.heat_source_for_rhs(a,c.rhs.evaluate(static[1:]))
    assert np.allclose(c.M@F+c.K@a,q,rtol=1e-9,atol=1e-12)
    h=1e-5
    J=(m.vector_field(a+h,static)-m.vector_field(a-h,static))/(2*h)
    assert np.allclose(m.evaluate(a,static).vector_field_jacobian[:,0],J,rtol=1e-6,atol=1e-9)
    T=m.reference.temperature(a)
    assert np.allclose(m.project_initial_temperature(T,geometry=[s]),a)


def test_continuous_mass_residual_equivalence_and_mass_contractivity(family):
    m = family
    cert = certify_geometry_mass_residual_equivalence(m)
    # Uniform scale s in [0.8,1.2] gives M_r=s^3 at this one-mode test model.
    assert cert.continuous_geometry_box
    assert cert.minimum_mass_eigenvalue <= .8**3
    assert cert.maximum_mass_eigenvalue >= 1.2**3

    s = 1.137
    state = np.array([.31])
    current = np.array([0.,0.])
    static = np.r_[m.normalize([s]),current]
    physical = m.evaluate(state,static)
    diagnostic = local_geometry_dynamics_diagnostic(
        m, geometry=[s], a=state, operating=current,
        da=physical.vector_field, mass_certificate=cert)
    assert diagnostic.vector_residual_norm < 1e-14
    assert diagnostic.mass_residual_norm < 1e-14
    assert diagnostic.local_mass_contractivity_margin > 0.


def test_geometry_full_em_radau_validation_is_independent_and_exact_for_zero_drive(family):
    m = family
    old_graph, old_config = m.graph, m.training_config
    try:
        m.graph = ParametricAnalyticEvolutionGraph(
            m.thermal_model.lambdas,['scale','current_0','current_1'])
        m.training_config = None
        result = validate_geometry_trajectory(
            m,[0.,.001,.01],geometry=[1.0],a0=[.2],operating=[0.,0.],
            full_electromagnetics=True)
        assert result['used_for_training'] is False
        assert result['reference'] == 'Radau + full sparse EM on queried geometry'
        assert result['maximum_coordinate_error'] < 1e-8
        assert result['maximum_temperature_error'] < 1e-8
    finally:
        m.graph, m.training_config = old_graph, old_config


def test_single_graph_checkpoint_unseen_geometry_and_no_online_reassembly(family,tmp_path,monkeypatch):
    m=family
    graph=ParametricAnalyticEvolutionGraph(m.thermal_model.lambdas,['scale','current_0','current_1'])
    graph.add_product_response('geometric_heating',0,['scale','current_0','current_0'],.01)
    m.graph=graph
    m.training_config=m.training_domain(ResearchTrainingConfig((0.,),(2.,),(0.,0.),(30.,30.),1.,1e-5))
    path=m.save(tmp_path/'family.npz')
    loaded=ResearchElectroThermalModel.load(path)
    assert isinstance(loaded,GeometryResearchModel)
    assert np.array_equal(loaded.reference.em.V,m.reference.em.V)
    assert loaded.em_basis_report==m.em_basis_report
    expected=m.predict(1000.,geometry=[1.137],a0=[.3],operating=[20.,8.],diagnostics=False)
    def forbidden(*args,**kwargs):
        raise AssertionError('online state-only query assembled/retrained a spatial model')
    monkeypatch.setattr(loaded,'context',forbidden)
    for t in [0.,1000.,1e300,np.inf]:
        result=loaded.predict(t,geometry={'scale':1.137},a0=[.3],operating=[20.,8.],diagnostics=False)
        assert np.all(np.isfinite(result['thermal_coordinates']))
        if t==1000.:
            assert np.array_equal(result['temperature_field'],expected['temperature_field'])
        if t==0.:
            assert np.allclose(result['thermal_coordinates'],[.3])
    with pytest.raises(ValueError,match='geometry'):
        loaded.predict(0.,geometry=[1.21],a0=[.3],operating=[20.,8.],diagnostics=False)
    with pytest.raises(ValueError,match='time'):
        loaded.predict(2.,geometry=[1.1],a0=[.3],operating=[20.,8.],diagnostics=False,
                       allow_time_extrapolation=False)


def test_training_learns_geometry_input_with_finite_and_steady_residuals():
    # An independently specified family with a known exact solution verifies the
    # generalized residual path, geometry weights and stationary candidate fit.
    from types import SimpleNamespace
    class Field:
        thermal_model=SimpleNamespace(lambdas=np.array([2.]))
        n_operating=2
        rhs_map=None
        def vector_field(self,a,p):
            return -2*a+(1+p[0])*p[1]**2
        def evaluate(self,a,p):
            return SimpleNamespace(vector_field=self.vector_field(a,p),vector_field_jacobian=np.array([[-2.]]))
    cfg=ResearchTrainingConfig((0.,),(1.,),(-.5,0.),(.5,2.),1000.,1e-8,
           sample_count=12,validation_count=12,max_nodes=2,max_degree=3,
           time_sampling='mixed_log',time_min=1e-5,include_steady_state=True)
    initial=ParametricAnalyticEvolutionGraph([2.],['u0','u1'])
    initial.add_product_response('heating',0,['u1','u1'],.2)
    initial.add_product_response('geometry_heating',0,['u0','u1','u1'],.3)
    graph,report=train_research_graph(Field(),cfg,graph=initial)
    assert report.numerical_tolerance_met
    assert any('u0' in node.parents for node in graph.response_nodes)
    for g,u,t in [(.173,1.32,.61),(-.31,.97,1e6),(.2,1.1,np.inf)]:
        a,da=evaluate_parametric_stable(graph,t,a0=[.41],operating=[g,u])
        equilibrium=(1+g)*u*u/2
        exact=equilibrium+(.41-equilibrium)*np.exp(-2*t)
        assert np.allclose(a,[exact],rtol=1e-8,atol=1e-8)
        assert np.linalg.norm(da-Field().vector_field(a,[g,u]))<1e-8
