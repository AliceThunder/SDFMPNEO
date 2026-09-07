import numpy as np
import pytest

from sdfmpneo import ResearchElectroThermalModel, ResearchTrainingConfig, demo_research_model
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.parametric_realization import evaluate_parametric_stable_with_jacobians


@pytest.fixture(scope='module')
def trained():
    model = demo_research_model()
    # Training must not consume full-order equilibrium snapshots or trajectory labels.
    def forbidden(*args, **kwargs):
        raise AssertionError('dense/full-solution compatibility path was used')
    model.em.problem.solve_full = forbidden
    model.em.problem.operator = forbidden
    model.em.problem.loss_operator = forbidden
    config = ResearchTrainingConfig((0.,),(2.,),(1000.,0.),(3000.,1000.),.25,.002,
                                    sample_count=16,validation_count=16,max_nodes=12,max_degree=2)
    report = model.train(config)
    assert report.numerical_tolerance_met
    assert report.maximum_validation_residual <= config.residual_tolerance
    assert report.final_rms_residual < report.initial_rms_residual * 1e-3
    assert np.all(np.diff(report.objective_history) <= 0.)
    return model


def test_nonlinear_training_checkpoint_and_physical_outputs(trained, tmp_path):
    path=trained.save(tmp_path/'model.npz')
    loaded=ResearchElectroThermalModel.load(path)
    for time in [0., .073, .219]:
        expected=trained.predict(time,a0=[1.],operating=[2000.,500.])
        actual=loaded.predict(time,a0=[1.],operating=[2000.,500.])
        assert np.allclose(actual.thermal_coordinates,expected.thermal_coordinates,rtol=1e-12,atol=1e-12)
        assert np.allclose(actual.impedance.impedance,expected.impedance.impedance,rtol=1e-10,atol=1e-12)
        assert np.allclose(actual.temperature_field,expected.temperature_field,atol=1e-12)
        assert actual.temperature_field.shape==(trained.core.mesh.n_nodes,)
        from sdfmpneo.__main__ import jsonable
        assert jsonable(actual)['maximum_temperature'] == float(np.max(actual.temperature_field))
        boundary=trained.core.thermal_assembly.boundary_nodes
        assert np.all(actual.temperature_field[boundary]==293.15)
        current=np.array([2000.,500.])
        terminal_power=.5*np.real(np.vdot(current,actual.impedance.impedance @ current))
        assert np.isclose(sum(actual.region_losses.values()),terminal_power,rtol=2e-5)
        assert np.isfinite(actual.drive_rhs_residual_dual_norm)
    assert loaded.training_report==trained.training_report
    initial=loaded.predict(0.,a0=[1.],operating=[2000.,500.])
    assert np.allclose(initial.thermal_coordinates,[1.],atol=1e-12)
    assert np.allclose(loaded.project_initial_temperature(initial.temperature_field),[1.],atol=1e-12)


def test_arbitrary_time_dag_inference_needs_no_em_solve(trained,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('EM solve in DAG-only inference')
    monkeypatch.setattr(trained.em,'state_for_rhs',forbidden)
    monkeypatch.setattr(trained.em.problem,'operator_sparse',forbidden)
    for t in [.2,.0,.1,.2]:
        out=trained.predict(t,a0=[.7],operating=[1723.,423.],diagnostics=False)
        assert np.isfinite(out['maximum_temperature'])


def test_independent_full_em_transient_validation(trained):
    for initial,current in [([1.],[2000.,500.]),([.3],[2700.,750.])]:
        result=trained.validate_trajectory([.25,0.,.041,.172,.25],a0=initial,operating=current)
        assert result['used_for_training'] is False
        assert result['maximum_temperature_error'] < 1e-3
        assert np.allclose(result['predicted_coordinates'][0],result['predicted_coordinates'][-1])


def test_training_budget_is_not_convergence():
    model=demo_research_model()
    config=ResearchTrainingConfig((0.,),(2.,),(1000.,0.),(3000.,1000.),.25,1e-10,
                                   sample_count=4,validation_count=4,max_nodes=1,max_degree=2)
    result=model.train(config)
    assert not result.numerical_tolerance_met
    assert result.status=='budget_exhausted'
    assert len(model.graph.response_nodes)==1


def test_nonuniform_reference_cannot_silently_omit_conductive_forcing():
    model = demo_research_model()
    problem = model.core.electromagnetic_problem
    problem.temperature_reference_local += np.array([0., 0., 0., 0., 1.])[model.core.mesh.tetrahedra]
    with pytest.raises(ValueError, match='conductive forcing'):
        ResearchElectroThermalModel(model.core, model.em, model.ports, model.rhs_map)


def test_dag_operating_and_weight_derivatives_at_near_resonance():
    graph=ParametricAnalyticEvolutionGraph([1.,1.-1e-12],['u'])
    graph.add_product_response('first',0,['a0_1','u'],.3)
    graph.add_product_response('second',1,['first','u'],.2)
    a0=np.array([.4,.7]);u=np.array([1.2]);t=.8
    _,_,jac,djac=evaluate_parametric_stable_with_jacobians(graph,t,a0=a0,operating=u)
    h=1e-5
    plus=evaluate_parametric_stable_with_jacobians(graph,t,a0=a0,operating=u+h)
    minus=evaluate_parametric_stable_with_jacobians(graph,t,a0=a0,operating=u-h)
    assert np.allclose(jac[:,0],(plus[0]-minus[0])/(2*h),rtol=1e-7,atol=1e-10)
    assert np.allclose(djac[:,0],(plus[1]-minus[1])/(2*h),rtol=1e-7,atol=1e-10)
    _,_,jac,djac=evaluate_parametric_stable_with_jacobians(graph,t,a0=a0,operating=u,weight_derivatives=True)
    for index in range(2):
        variants=[]
        for sign in [1,-1]:
            clone=ParametricAnalyticEvolutionGraph(graph.lambdas,graph.operating_names)
            for j,node in enumerate(graph.response_nodes):
                clone.add_product_response(node.name,node.target_mode,node.parents,node.weight+sign*h*(j==index))
            variants.append(evaluate_parametric_stable_with_jacobians(clone,t,a0=a0,operating=u))
        assert np.allclose(jac[:,index],(variants[0][0]-variants[1][0])/(2*h),rtol=1e-7,atol=1e-10)
        assert np.allclose(djac[:,index],(variants[0][1]-variants[1][1])/(2*h),rtol=1e-7,atol=1e-10)


def test_fixed_source_and_thermal_forcing_without_operating_parameters():
    from sdfmpneo import CertifiedElectroThermalVectorField
    from sdfmpneo.em import ParametricEMProblem,ReducedEMModel
    from sdfmpneo.thermal import ThermalSpectralModel
    from sdfmpneo.training import AffineOperatingRHSMap
    from sdfmpneo.training.research import train_research_graph
    problem=ParametricEMProblem(np.eye(1,dtype=complex),np.zeros((1,1,1),complex),np.ones(1),np.eye(1),np.ones((1,1,1)))
    thermal=ThermalSpectralModel.build(np.eye(1),np.array([[2.]]))
    field=CertifiedElectroThermalVectorField(thermal,ReducedEMModel(problem,np.eye(1)),
                rhs_map=AffineOperatingRHSMap(np.ones(1),np.empty((1,0))),thermal_forcing=np.array([.5]))
    config=ResearchTrainingConfig((0.,),(1.,),(),(),1.,1e-10,sample_count=4,validation_count=4,max_nodes=2)
    graph,report=train_research_graph(field,config)
    assert report.numerical_tolerance_met
    value=evaluate_parametric_stable_with_jacobians(graph,.7,a0=[.3],operating=[])[0]
    assert np.allclose(value,[.3*np.exp(-1.4)+.75*(1-np.exp(-1.4))],atol=1e-12)


@pytest.fixture
def run_script():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    spec = spec_from_file_location('uwpt_run_script', Path(__file__).resolve().parents[1]/'run.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_script_inference_paths_and_nodal_initial_temperature(trained, tmp_path, monkeypatch, run_script):
    import json
    run = run_script

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    trained.save(tmp_path/'saved'/'model.npz')
    initial = trained.temperature(np.array([.7]))
    np.save(tmp_path/'initial.npy', initial)
    monkeypatch.setattr(run, 'FILES', {
        'model': 'saved/model.npz', 'predictions': 'output/predictions.json',
        'settings_dir': 'output', 'resume_model': None,
    })
    monkeypatch.setattr(run, 'PREDICTION', {
        'a0': [999.], 'operating': [1723., 423.], 'times': [.2, 0., .1],
        'initial_temperature_file': 'initial.npy',
        'state_only': True, 'allow_extrapolation': False,
    })
    assert run.main(['--mode', 'predict']) == 0
    predictions = json.loads((tmp_path/'output'/'predictions.json').read_text())
    assert [p['time'] for p in predictions] == [.2, 0., .1]
    assert np.allclose(predictions[1]['temperature_field'], initial)
    settings = json.loads((tmp_path/'output'/'predict.settings.json').read_text())
    assert np.allclose(settings['effective_a0'], [.7])


def test_run_script_training_preserves_nonconvergence_exit_and_checkpoint(tmp_path, monkeypatch, run_script):
    import json
    run = run_script

    monkeypatch.setattr(run, 'ROOT', tmp_path)
    seed = demo_research_model()
    seed.graph = ParametricAnalyticEvolutionGraph(seed.core.thermal_model.lambdas, ['u0', 'u1'])
    seed.save(tmp_path/'seed.npz')
    monkeypatch.setattr(run, 'FILES', {
        'model': 'results/model.npz', 'predictions': 'unused.json',
        'settings_dir': 'results', 'resume_model': 'seed.npz',
    })
    monkeypatch.setattr(run, 'TRAINING', {
        'initial_lower': [0.], 'initial_upper': [2.],
        'operating_lower': [1000., 0.], 'operating_upper': [3000., 1000.],
        'time_horizon': .25, 'sample_count': 4, 'validation_count': 4,
        'max_nodes': 1, 'residual_tolerance': 1e-10,
    })
    assert run.main(['--mode', 'train']) == 2
    output = tmp_path/'results'
    report = json.loads((output/'training.report.json').read_text())
    assert report['status'] == 'budget_exhausted'
    assert not report['numerical_tolerance_met']
    model = ResearchElectroThermalModel.load(output/'model.npz')
    assert len(model.graph.response_nodes) == 1
