"""Cross-geometry BFZI sea-Z tangent transfer in aligned physical heat modes.

Uses only ORIGINAL BFZI cold thermal bases and EM teachers. P03/P07 dynamic
histories never participate in training or hyperparameter selection.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from .geom_memory import geom_feat
from .geom_tangent import SeaTangent, basis
from .geom_heat import pack


def rotate(xyz, vol, modes, query_xyz, query_modes):
    """Orthogonal Procrustes rotation from a native sea basis to reference.

    Input reference coordinates/vol/modes; target world-space FV centers/modes.
    The returned Q makes native_modes @ Q comparable across BFZI geometries.
    """
    xyz=np.asarray(xyz,float)
    vol=np.asarray(vol,float)
    modes=np.asarray(modes,float)
    pts=np.asarray(query_xyz,float)
    vals=np.asarray(query_modes,float)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or pts.ndim!=2 or
            pts.shape[1]!=3 or modes.shape!=(len(xyz),vals.shape[1]) or
            vals.shape[0]!=len(pts) or vol.shape!=(len(xyz),)):
        raise ValueError("physical BFZI heat-mode reference dimensions differ")
    if np.any(vol<=0):
        raise ValueError("positive BFZI FV volumes required")
    dist,idx=cKDTree(pts).query(xyz,k=min(8,len(pts)))
    dist=np.asarray(dist).reshape(len(xyz),-1)
    idx=np.asarray(idx).reshape(len(xyz),-1)
    wt=1./np.maximum(dist,1e-9)**2
    wt/=wt.sum(axis=1,keepdims=True)
    b=np.einsum('ij,ijk->ik',wt,vals[idx])
    b[dist[:,0]<1e-10]=vals[idx[dist[:,0]<1e-10,0]]
    C=b.T@(vol[:,None]*modes)
    u,_,vh=np.linalg.svd(C,full_matrices=False)
    return u@vh


def vec(jac):
    return np.concatenate([pack(j) for j in np.asarray(jac,complex)])


def mat(v):
    a=np.asarray(v,float).reshape(-1,6)
    z=np.zeros((len(a),2,2),complex)
    z[:,0,0]=a[:,0]+1j*a[:,1]
    z[:,0,1]=z[:,1,0]=a[:,2]+1j*a[:,3]
    z[:,1,1]=a[:,4]+1j*a[:,5]
    return z


def feat(head):
    if head.pose is None:
        raise ValueError("40D BFZI relative SE(3) pose required")
    return geom_feat(head.geom,head.z0,head.pose)


def label(ref_head,ref_tan,head,tan):
    """Return matched-frame physical Jacobian and its native mode rotation."""
    if len(tan.jac)!=len(ref_tan.jac):
        raise ValueError("cross-geometry BFZI tangent ranks must match")
    Q=rotate(ref_head.xyz,ref_head.vol,ref_tan.modes,
             head.xyz,tan.modes)
    return vec(np.einsum('ij,iab->jab',Q,tan.jac)),Q


def neighbors(x,y,xi,scale,exclude=None):
    d=((x-xi)/scale)**2
    dist=np.mean(d,axis=1)
    if exclude is not None:
        dist[exclude]=np.inf
    w=1/np.maximum(dist,.01)
    w/=w.sum()
    return w@y


class Resid(nn.Module):
    def __init__(self,nout,hidden=12):
        super().__init__()
        self.a=nn.Linear(40,hidden)
        self.b=nn.Linear(hidden,nout)

    def forward(self,x):
        return self.b(torch.tanh(self.a(x)))


def fit_net(x,y,epochs=350):
    torch.manual_seed(37)
    model=Resid(y.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.03)
    tx=torch.as_tensor(x,dtype=torch.float32)
    ty=torch.as_tensor(y,dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss=(model(tx)-ty).square().mean()
        loss.backward()
        opt.step()
    return model


class TangentTransfer:
    """Tiny NN residual around geometry-neighbor physical Jacobian interpolation.

    Every saved direction is in the same reference heat-mode frame. Inference
    needs the target geometry's C/H cold ROM but zero temperature-perturbed EM
    solves. The original FV J/q, exact copper model and 5-material ROM remain.
    """
    def __init__(self,ref_head,ref_tan):
        self.xyz=ref_head.xyz.copy()
        self.vol=ref_head.vol.copy()
        self.modes=ref_tan.modes.copy()
        self.rank=len(ref_tan.jac)

    @classmethod
    def train(cls,examples,epochs=350):
        """examples: list of (name, GeomHeat, SeaTangent); P03/P07 excluded."""
        names=[n.upper() for n,_,_ in examples]
        if len(set(names))!=len(names) or len(names)<4 or any(
                n in ('P03','P07') for n in names):
            raise ValueError("need >=4 distinct training-only BFZI geometries")
        _,ref_head,ref_tan=examples[0]
        obj=cls(ref_head,ref_tan)
        obj.names=names
        obj.x=np.stack([feat(head) for _,head,_ in examples])
        obj.y=np.stack([label(ref_head,ref_tan,head,tan)[0]
                        for _,head,tan in examples])
        obj.xavg=obj.x.mean(axis=0)
        obj.xscale=np.maximum(np.std(obj.x,axis=0),.2)
        obj.yscale=np.maximum(np.std(obj.y,axis=0),
                              np.max(np.abs(obj.y),axis=0)*.02+1e-15)
        obj.outscale=np.maximum(np.max(np.abs(obj.y),axis=0),1e-15)
        # Folded geometry validation chooses the residual weight using only
        # training geometries; no held-out P03/P07 EM or dynamic histories.
        errs=np.zeros((len(examples),4))
        choices=np.array([0.,.25,.5,1.])
        for i in range(len(examples)):
            idx=np.arange(len(examples))!=i
            a=obj.x[idx]; b=obj.y[idx]
            scale=np.maximum(np.std(a,axis=0),.2)
            avg=a.mean(axis=0)
            bas=np.stack([neighbors(a,b,a[j],scale,exclude=j)
                          for j in range(len(a))])
            yn=np.maximum(np.max(np.abs(b),axis=0),1e-15)
            net=fit_net((a-avg)/scale,(b-bas)/yn,epochs)
            with torch.no_grad():
                res=net(torch.tensor(((obj.x[i]-avg)/scale)[None],
                                     dtype=torch.float32)).numpy()[0]*yn
            base=neighbors(a,b,obj.x[i],scale)
            for k,c in enumerate(choices):
                errs[i,k]=np.linalg.norm((base+c*res-obj.y[i])/obj.outscale)
        obj.cv=np.mean(errs,axis=0)
        obj.weight=float(choices[np.argmin(obj.cv)])
        bas=np.stack([neighbors(obj.x,obj.y,obj.x[i],obj.xscale,
                                exclude=i) for i in range(len(obj.x))])
        obj.net=fit_net((obj.x-obj.xavg)/obj.xscale,
                        (obj.y-bas)/obj.yscale,epochs)
        return obj

    def predict(self,head,rom):
        """Compute target physical modes cold; predict BFZI sea tangent (0 EM)."""
        if len(rom.phi)!=len(head.cap):
            raise ValueError("target heat ROM geometry mismatch")
        native=basis(head.vol,rom.phi[:head.ns],self.rank)
        if native.shape[1]!=self.rank:
            raise ValueError("target heat ROM has insufficient sea modes")
        Q=rotate(self.xyz,self.vol,self.modes,head.xyz,native)
        x=feat(head)
        base=neighbors(self.x,self.y,x,self.xscale)
        with torch.no_grad():
            inp=torch.tensor(((x-self.xavg)/self.xscale)[None],
                             dtype=torch.float32)
            corr=self.net(inp).numpy()[0]*self.yscale
        jac=mat(base+self.weight*corr)
        tan=SeaTangent(native@Q,jac,head.vol,head.sea0,head.geom)
        tan.method="neural-residual"
        tan.nsolve=0
        tan.amplitude=0.
        return tan

    def save(self,path):
        d={k:getattr(self,k) for k in (
            'xyz','vol','modes','rank','x','y','xavg','xscale',
            'yscale','outscale','weight','cv','names')}
        for k,val in self.net.state_dict().items():
            d['net_'+k]=val.detach().numpy()
        np.savez_compressed(Path(path),**d)

    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            obj=cls.__new__(cls)
            for k in ('xyz','vol','modes','rank','x','y','xavg','xscale',
                      'yscale','outscale','weight','cv','names'):
                v=d[k]
                setattr(obj,k,v.item() if k in ('rank','weight') else v.copy())
            obj.net=Resid(obj.y.shape[1])
            state={k:torch.as_tensor(d['net_'+k].copy())
                   for k in obj.net.state_dict()}
            obj.net.load_state_dict(state)
            obj.net.eval()
        return obj
