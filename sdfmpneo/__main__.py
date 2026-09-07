"""python -m sdfmpneo: train, predict, and independently validate a research model."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
from .research import ResearchElectroThermalModel, demo_research_model, model_from_config
from .training.research import ResearchTrainingConfig


def jsonable(value):
    if is_dataclass(value):
        result = asdict(value)
        if hasattr(value, 'maximum_temperature'):
            result['maximum_temperature'] = value.maximum_temperature
        return jsonable(result)
    if isinstance(value,np.ndarray):
        if np.iscomplexobj(value):
            return {'real':value.real.tolist(),'imag':value.imag.tolist()}
        return value.tolist()
    if isinstance(value,complex):
        return {'real':value.real,'imag':value.imag}
    if isinstance(value,dict):
        return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [jsonable(v) for v in value]
    if isinstance(value,np.generic):
        return value.item()
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    train=commands.add_parser('train',help='fit only the coupled equation residual')
    group=train.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo',action='store_true')
    group.add_argument('--config',type=Path)
    train.add_argument('--output',type=Path,required=True)
    train.add_argument('--max-nodes',type=int)
    train.add_argument('--residual-tolerance',type=float)
    for command in ['predict','validate']:
        sub=commands.add_parser(command)
        sub.add_argument('model',type=Path)
        sub.add_argument('--a0',nargs='+',type=float,required=True)
        sub.add_argument('--operating',nargs='+',type=float,required=True)
        sub.add_argument('--times',nargs='+',type=float,required=True)
        sub.add_argument('--output',type=Path)
        if command=='predict':
            sub.add_argument('--state-only',action='store_true',help='DAG-only inference; skip EM diagnostics')
            sub.add_argument('--allow-extrapolation',action='store_true')
        else:
            sub.add_argument('--rom-reference',action='store_true',help='use EM ROM in the validation integrator')
    args=parser.parse_args(argv)
    if args.command=='train':
        if args.demo:
            model=demo_research_model()
            config=ResearchTrainingConfig((0.,),(2.,),(1000.,0.),(3000.,1000.),.25,.002,
                                           sample_count=16,validation_count=16,max_nodes=12,max_degree=2)
        else:
            model,config=model_from_config(args.config)
        if args.max_nodes is not None or args.residual_tolerance is not None:
            from dataclasses import replace
            config=replace(config,**({} if args.max_nodes is None else {'max_nodes':args.max_nodes}),
                            **({} if args.residual_tolerance is None else {'residual_tolerance':args.residual_tolerance}))
        report=model.train(config,progress=lambda n,r,m:print(f'node={n} rms_residual={r:.6g} max_residual={m:.6g}',flush=True))
        model.save(args.output)
        text=json.dumps(jsonable(report),indent=2,allow_nan=False)
        args.output.with_suffix('.training.json').write_text(text+'\n')
        print(text)
        return 0 if report.numerical_tolerance_met else 2
    model=ResearchElectroThermalModel.load(args.model)
    if args.command=='predict':
        result=[model.predict(t,a0=args.a0,operating=args.operating,diagnostics=not args.state_only,
                               allow_extrapolation=args.allow_extrapolation) for t in args.times]
    else:
        result=model.validate_trajectory(args.times,a0=args.a0,operating=args.operating,
                                         full_electromagnetics=not args.rom_reference)
    text=json.dumps(jsonable(result),indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    else:
        print(text)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
