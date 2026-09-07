"""Complete nonlinear training -> save -> reload -> inference -> independent check."""
from pathlib import Path
import json
from sdfmpneo import ResearchElectroThermalModel,ResearchTrainingConfig,demo_research_model
from sdfmpneo.__main__ import jsonable


def main():
    output=Path('results/research_demo')
    output.mkdir(parents=True,exist_ok=True)
    model=demo_research_model()
    config=ResearchTrainingConfig((0.,),(2.,),(1000.,0.),(3000.,1000.),.25,.002,
                                  sample_count=16,validation_count=16,max_nodes=12,max_degree=2)
    report=model.train(config,progress=lambda n,r,m:print(f'nodes={n} rms={r:.6g} maximum={m:.6g}',flush=True))
    model.save(output/'model.npz')
    model=ResearchElectroThermalModel.load(output/'model.npz')
    prediction=model.predict(.2,a0=[1.],operating=[2000.,500.])
    validation=model.validate_trajectory([0,.05,.1,.2,.25],a0=[1.],operating=[2000.,500.])
    result={'training':report,'prediction':prediction,'validation':validation}
    (output/'results.json').write_text(json.dumps(jsonable(result),indent=2,allow_nan=False)+'\n')
    print('temperature max (K):',prediction.maximum_temperature)
    print('independent maximum temperature error (K):',validation['maximum_temperature_error'])
    if not report.numerical_tolerance_met:
        raise SystemExit('training budget exhausted; inspect results.json before using the model')
    print(output/'model.npz')


if __name__=='__main__':
    main()
