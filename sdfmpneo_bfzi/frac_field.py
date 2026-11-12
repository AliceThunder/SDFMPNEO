"""Compact temperature-conditioned BFZI seawater J and Joule field decoder.

The decoder learns *unit-port* seawater currents and the normalized local
weak-form Joule distribution from independent original BFZI snapshots.
At inference, fields use the neural thermal prediction and circuit currents.
No Green solve or original BFZI checkout is required.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def complex_z(a):
    x=np.asarray(a,float)
    return x[...,0]+1j*x[...,1]


def z6(z):
    return np.stack([
        z[...,0,0].real,z[...,0,0].imag,
        z[...,0,1].real,z[...,0,1].imag,
        z[...,1,1].real,z[...,1,1].imag],axis=-1)


def zmat(v):
    v=np.asarray(v,float)
    a=v[0]+1j*v[1]
    b=v[2]+1j*v[3]
    c=v[4]+1j*v[5]
    return np.array([[a,b],[b,c]],complex)


class SeaHead:
    def __init__(self,vol,basis,sc,j0,q0,z0,jfit,qfit,zfit,
                 *,ambient=293.15,alpha=.02):
        self.vol=np.asarray(vol,float)
        self.basis=np.asarray(basis,float)
        self.sc=np.asarray(sc,float)
        self.j0=np.asarray(j0,complex)
        self.q0=np.asarray(q0,float)
        self.z0=np.asarray(z0,float)
        self.jfit=np.asarray(jfit,complex)
        self.qfit=np.asarray(qfit,float)
        self.zfit=np.asarray(zfit,float)
        self.amb=float(ambient)
        self.alpha=float(alpha)

    @classmethod
    def fit(cls,files,rank=6,reg=1e-5):
        temps=[];currents=[];heats=[];power=[];zs=[]
        vol=None
        for file in files:
            path=Path(file)
            with np.load(path) as d:
                if vol is None:
                    vol=np.asarray(d["sea_vol_m3"],float)
                elif not np.allclose(vol,d["sea_vol_m3"]):
                    raise ValueError("sea geometry mismatch across original BFZI snapshots")
                temps.append(np.asarray(d["Tsea_K"],float))
                currents.append(np.asarray(d["Jsea_unit_A_m2_per_A"],complex))
                heats.append(np.asarray(d["qsea_W_m3"],float))
            r=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            power.append([a["Psea_port_W"] for a in r["trace"]])
            zs.append([z6(complex_z(a["Zsea_ohm"])) for a in r["trace"]])
        temp=np.concatenate(temps)
        unit=np.concatenate(currents)
        q=np.concatenate(heats)
        p=np.concatenate(power)
        z=np.concatenate(zs)
        if unit.shape[1:]!=(len(vol),2,3) or temp.shape!=q.shape:
            raise ValueError("original BFZI sea field dimensions mismatch")
        amb=float(np.min(temp))
        delta=temp-amb
        _,s,vh=np.linalg.svd(delta,full_matrices=False)
        good=np.count_nonzero(s>max(s[0]*1e-10,1e-12))
        rank=min(int(rank),int(good))
        if rank<1:
            raise ValueError("temperature snapshots must contain heating")
        basis=vh[:rank].T
        feat=delta@basis
        sc=np.maximum(np.std(feat,axis=0),1e-6)
        F=feat/sc
        ridge=float(reg)*max(np.trace(F.T@F)/rank,1e-8)
        eye=np.eye(rank)
        def fit(y,mask=None):
            A=F if mask is None else F[mask]
            B=y if mask is None else y[mask]
            return np.linalg.solve(A.T@A+ridge*eye,A.T@B)
        cold=np.where(np.max(abs(delta),axis=1)<1e-8)[0]
        if len(cold)==0:
            raise ValueError("cold original BFZI snapshot required")
        j0=np.mean(unit[cold],axis=0)
        z0=np.mean(z[cold],axis=0)
        ratio=1+.02*delta
        dj=(unit-ratio[:,:,None,None]*j0).reshape(len(temp),-1)
        jfit=fit(dj)
        zfit=fit(z-z0)
        on=p>1e-10
        if not np.any(on):
            raise ValueError("nonzero port Joule power required")
        shape=q[on]/p[on,None]
        q0=np.mean((q[cold]/p[cold,None]),axis=0)
        q0=np.maximum(q0,0.)
        q0/=q0@vol
        qfit=fit(shape-q0,mask=on)
        return cls(vol,basis,sc,j0,q0,z0,jfit,qfit,zfit,
                   ambient=amb)

    def features(self,sea_t):
        t=np.asarray(sea_t,float)
        if t.shape!=(len(self.vol),):
            raise ValueError("predicted seawater temperature dimensions mismatch")
        return ((t-self.amb)@self.basis)/self.sc

    def at(self,sea_t,amps):
        t=np.asarray(sea_t,float)
        I=np.asarray(amps,complex)
        if I.shape!=(2,):
            raise ValueError("two port currents required")
        f=self.features(t)
        ratio=1+self.alpha*(t-self.amb)
        ju=(ratio[:,None,None]*self.j0
            +(f@self.jfit).reshape(self.j0.shape))
        J=np.einsum("npc,p->nc",ju,I,optimize=True)
        z=zmat(self.z0+f@self.zfit)
        psea=max(float(.5*np.vdot(I,z@I).real),0.)
        shape=np.maximum(self.q0+f@self.qfit,0.)
        total=float(shape@self.vol)
        if total<=0:
            raise ValueError("unphysical sea heat shape")
        q=psea*shape/total
        return J,q,z

    def save(self,path):
        np.savez_compressed(path,vol=self.vol,basis=self.basis,sc=self.sc,
            j0=self.j0,q0=self.q0,z0=self.z0,
            jfit=self.jfit,qfit=self.qfit,zfit=self.zfit,
            ambient=self.amb,alpha=self.alpha)

    @classmethod
    def load(cls,path):
        with np.load(path) as d:
            return cls(d["vol"],d["basis"],d["sc"],
                       d["j0"],d["q0"],d["z0"],
                       d["jfit"],d["qfit"],d["zfit"],
                       ambient=float(d["ambient"]),
                       alpha=float(d["alpha"]))
