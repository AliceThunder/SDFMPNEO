"""Train a geometry-free seawater-Z correction on the deployed cold/heat pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import geometry, rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import ZFix, restore as load_z


def init_case(case, atlas, port_self, mutual, source):
    init = atlas.predict(case)
    for model in (port_self, mutual, source):
        if model is not None:
            init = model.apply(case, init)
    return init


def build(specs, plan, atlas, net, ck, base, port_self, mutual,
          source, shape_mix, training=False):
    out = []
    for i, spec in enumerate(specs):
        name = Path(spec['anchor']).stem.split('_')[2]
        case = plan['cases'][name]
        init = init_case(case, atlas, port_self, mutual, source)
        _, feat = geometry(case, init)
        for row in prepare(spec, 32):
            with torch.no_grad():
                pred = rollout(case, atlas, net, ck, row['time'],
                    row['voltage'], base, init_override=init,
                    shape_mix=shape_mix)
            if training:
                truth = np.asarray(row['z6'], float)
                delta = (truth - truth[0]) - (pred['z6'] - pred['z6'][0])
                a = np.asarray(pred['heat_state'][1:], float)
                g = np.repeat(feat[None], len(a), axis=0)
                out.append(dict(
                    g=torch.as_tensor(g, dtype=torch.float32),
                    a=torch.as_tensor(a, dtype=torch.float32),
                    target=torch.as_tensor(delta[1:], dtype=torch.float32),
                    den=max(float(np.linalg.norm(truth-truth[0])), 1e-12),
                    file=row['file']))
            else:
                out.append(dict(case=case, init=init, row=row))
        print('FREE_Z_PREP', name, len(out), flush=True)
    return out


def objective(model, rows, sy):
    vals = []
    regs = []
    for row in rows:
        corr = model(row['g'], row['a']) * sy
        vals.append((corr-row['target']).square().sum()/row['den']**2)
        regs.append(corr.square().sum()/row['den']**2)
    return torch.stack(vals).mean() + .002*torch.stack(regs).mean()


def closed_score(model, rows, atlas, net, ck, base, sy, shape_mix):
    model.eval()
    def tangent(g, a):
        return base(g, a) + model(g, a)*sy
    out = []
    with torch.no_grad():
        for item in rows:
            row = item['row']
            pred = rollout(item['case'], atlas, net, ck,
                row['time'], row['voltage'], tangent,
                init_override=item['init'], shape_mix=shape_mix)
            out.append(metric(pred, row))
    return dict(
        dynamic_z_pct=float(np.mean([x['dynamic_z_rel_L2'] for x in out])*100),
        current_pct=float(np.mean([x['current_rel_L2'] for x in out])*100),
        temp_rmse_K=float(np.mean([x['temperature_rmse_K'] for x in out])),
        cu_tx_K=float(np.mean([x['material_cu_tx_rmse_K'] for x in out])),
        cu_rx_K=float(np.mean([x['material_cu_rx_rmse_K'] for x in out])),
        sea_power_pct=float(np.mean([x['sea_power_rel_L2'] for x in out])*100),
        copper_power_pct=float(np.mean([x['copper_power_rel_L2'] for x in out])*100),
        budget_W=float(max(x['energy_budget_W'] for x in out)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('runs/geomfrac_pose42_plan.json'))
    p.add_argument('--manifest', type=Path, default=Path('runs/geomfrac_s5_v6_42g_manifest.json'))
    p.add_argument('--atlas', type=Path, default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--model', type=Path, default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--base-z-fix-model', type=Path, default=Path('runs/geomfrac_et_v9_ztune.pt'))
    p.add_argument('--port-self-model', type=Path, default=Path('runs/geomfrac_portself_v5.npz'))
    p.add_argument('--mutual-model', type=Path, default=Path('runs/geomfrac_mutual_v6.npz'))
    p.add_argument('--source-model', type=Path, default=Path('runs/geomfrac_src_v6.npz'))
    p.add_argument('--shape-mix', type=float, default=.5)
    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--patience', type=int, default=80)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--out', type=Path, default=Path('runs/geomfrac_et_v10_free_ztune.pt'))
    p.add_argument('--report', type=Path, default=Path('runs/geomfrac_et_v10_free_ztune_dev.json'))
    args = p.parse_args()
    if not 0 <= args.shape_mix <= 1 or args.epochs < 1 or args.eval_every < 1:
        p.error('invalid training schedule or shape mix')

    torch.manual_seed(109)
    torch.set_num_threads(2)
    ck = torch.load(args.model, map_location='cpu', weights_only=True)
    net = GeomFracET(ng=int(ck['ng']), rank=32,
        soe_quad=ck.get('soe_quad', 'trap'),
        heat_dt=float(ck.get('heat_dt', 1.875)),
        heat_z=bool(ck.get('heat_z', False)))
    net.load_state_dict(ck['state'])
    net.eval()
    atlas = GeomInit.load(args.atlas)
    base = load_z(args.base_z_fix_model)
    port_self = PortSelf.load(args.port_self_model) if args.port_self_model else None
    mutual = MutFix.load(args.mutual_model) if args.mutual_model else None
    source = SrcFix.load(args.source_model) if args.source_model else None
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))

    hold_names = {'N05', 'N10', 'N15', 'N20', 'N25', 'N27'}
    train_specs = [x for x in manifest['train']
                   if Path(x['anchor']).stem.split('_')[2] not in hold_names]
    hold_specs = [x for x in manifest['train']
                  if Path(x['anchor']).stem.split('_')[2] in hold_names]
    if len(train_specs) != 36 or len(hold_specs) != 6 or len(manifest['val']) != 5:
        p.error('expected 36 train, 6 internal holdout and 5 frozen dev geometries')

    train = build(train_specs, plan, atlas, net, ck, base,
        port_self, mutual, source, args.shape_mix, training=True)
    dev = build(manifest['val'], plan, atlas, net, ck, base,
        port_self, mutual, source, args.shape_mix)
    hold = build(hold_specs, plan, atlas, net, ck, base,
        port_self, mutual, source, args.shape_mix)

    sy = torch.as_tensor(np.asarray(ck['y_scale'])[:6], dtype=torch.float32)
    model = ZFix(ng=45, sa=np.asarray(ck['a_scale']))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.002)
    best = float('inf')
    best_ep = 0
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []

    def measure(ep):
        nonlocal best, best_ep, saved
        d = closed_score(model, dev, atlas, net, ck, base, sy, args.shape_mix)
        h = closed_score(model, hold, atlas, net, ck, base, sy, args.shape_mix)
        select = .5*(d['dynamic_z_pct']+h['dynamic_z_pct'])
        history.append(dict(epoch=ep, dev=d, hold=h, selection_pct=select))
        if select < best:
            best, best_ep = select, ep
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print('FREE_Z_DEV', ep, 'dev', round(d['dynamic_z_pct'], 4),
              'hold', round(h['dynamic_z_pct'], 4),
              'current', round(d['current_pct'], 4),
              'temp', round(d['temp_rmse_K'], 4),
              'best', round(best, 4), 'best_epoch', best_ep, flush=True)

    measure(0)
    for ep in range(1, args.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = objective(model, train, sy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        opt.step()
        if ep % args.eval_every == 0:
            measure(ep)
            if ep-best_ep >= args.patience:
                break

    model.load_state_dict(saved)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), ng=model.ng,
        sa=torch.as_tensor(np.asarray(ck['a_scale'])), sy=sy,
        base=str(args.base_z_fix_model), base_mode='ztune',
        train_files=[x['file'] for x in train]), args.out)
    report = dict(
        status='geometry-free coupled-state Z tangent selected on dev and holdout',
        train_geometries=len(train_specs), dev_geometries=len(manifest['val']),
        hold_geometries=len(hold_specs), train_histories=len(train),
        dev_histories=len(dev), hold_histories=len(hold),
        baseline_dev=history[0]['dev'], baseline_hold=history[0]['hold'],
        best_epoch=best_ep, best_selection_pct=best,
        best=next(x for x in history if x['epoch']==best_ep),
        history=history, base_model=str(args.base_z_fix_model),
        atlas=str(args.atlas), heat_model=str(args.model),
        port_self_model=str(args.port_self_model),
        mutual_model=str(args.mutual_model), source_model=str(args.source_model),
        shape_mix=args.shape_mix, checkpoint=str(args.out))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('FREE_Z_FINAL', best_ep, round(best, 4), flush=True)


if __name__ == '__main__':
    main()
