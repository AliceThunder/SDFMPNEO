"""Cold lead geometry selection preserves every rejected raw sample."""
from examples.bfzi_cold_lead_guard import apply
from examples.bfzi_cold_plan import sample


def test_rejected_leads_leave_cold_splits_only():
    plan=dict(train=['C0001','C0002'],val=['C0003'],
              test=['C0004'],
              cases={f'C{i:04d}':{'alpha1':.85,'alpha2':.9}
                     for i in range(1,5)})
    report={k:dict(clear=k not in ('C0002','C0003'))
            for k in plan['cases']}
    removed=apply(plan,report)
    assert removed==dict(train=['C0002'],val=['C0003'],test=[])
    assert plan['train']==['C0001']
    assert plan['val']==[]
    assert plan['test']==['C0004']
    assert {r['name'] for r in plan['excluded_leads']}=={'C0002','C0003'}
    assert len(plan['cases'])==4
    second=apply(plan,report)
    assert not any(second.values())
    assert len(plan['excluded_leads'])==2


def test_geom_sampler_accept_filter():
    pred=lambda g: g['alpha1']>=.84 and g['alpha2']>=.84
    x=sample(179,n=40,accept=pred)
    y=sample(179,n=40,accept=pred)
    assert x==y
    assert len(x['cases'])==40
    assert all(pred(g) for g in x['cases'].values())
