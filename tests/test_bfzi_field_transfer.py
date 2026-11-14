import numpy as np
import pytest
from sdfmpneo_bfzi.field_transfer import owner,shared,norm,compare,project


def cells(bounds):
    out=[]
    for lo,hi in bounds:
        lo=np.array(lo,float);hi=np.array(hi,float)
        q=np.array([lo+(hi-lo)*.25,lo+(hi-lo)*.75])
        out.append(dict(cell_min=lo,cell_max=hi,
            quadrature_points=q,quadrature_weights=np.full(2,np.prod(hi-lo)/2)))
    return out


def test_shared_samples_exact_piecewise_fields():
    a=cells([([0,0,0],[.5,1,1]),([.5,0,0],[1,1,1])])
    b=cells([([0,0,0],[.25,1,1]),([.25,0,0],[.5,1,1]),([.5,0,0],[1,1,1])])
    q,w,ia,ib=shared(a,b)
    assert np.isclose(w.sum(),1)
    assert np.all((ia==0)==(q[:,0]<.5))
    assert np.allclose(np.array([3.,8.])[ia],np.array([3.,3.,8.])[ib])
    assert np.isclose(norm(np.ones((len(q),3)),w),np.sqrt(3))


def test_shared_quadrature_rejects_ambiguous_and_missing():
    a=cells([([0,0,0],[.5,1,1])])
    with pytest.raises(ValueError,match='outside'):
        owner(a,np.array([[.75,.3,.3]]))
    with pytest.raises(ValueError,match='ambiguous'):
        owner(a+a,np.array([[.3,.3,.3]]))


def test_field_compare_removes_drive_amp_and_phase():
    w=np.array([1.,2.])
    ref=np.array([[1.+1j,0,0],[2.+0j,1j,0]])
    a=np.array([ref,ref*(2j)])
    b=np.array([ref,ref*3])
    res=compare(a,b,w)
    assert res['raw'][0]==0
    assert res['raw'][1]>0.5
    assert np.isclose(res['shape'][1],0,atol=1e-12)
    assert res['thermal'][1] is None


def test_original_wet_fv_face_tie_uses_deterministic_owner():
    r=cells([([0,0,0],[.5,1,1]),([.5,0,0],[1,1,1])])
    q=np.array([[.5,.2,.3],[1.5,.2,.3]])
    assert owner(r,q,allow_face=True,allow_outside=True).tolist()==[0,-1]


def test_shared_overlap_projection_conserves_integrated_field():
    a=cells([([0,0,0],[.5,1,1]),([.5,0,0],[1,1,1])])
    b=cells([([0,0,0],[.25,1,1]),([.25,0,0],[.5,1,1]),([.5,0,0],[1,1,1])])
    _,w,ia,ib=shared(a,b)
    fine=np.array([3+1j,3+1j,8-2j])
    coarse,vol=project(w,ia,ib,2,3,fine)
    assert np.allclose(coarse,[3+1j,8-2j])
    assert np.allclose(vol,[.5,.5])
    assert np.allclose(vol@coarse,np.dot(w,fine[ib]))
