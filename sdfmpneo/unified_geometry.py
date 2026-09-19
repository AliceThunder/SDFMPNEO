"""Analytic geometry for the unified fixed-background UWPT solver."""
from __future__ import annotations
from dataclasses import dataclass
import copy,json
import numpy as np


def _v3(x,name):
    x=np.asarray(x,float).reshape(-1)
    if x.shape!=(3,) or np.any(~np.isfinite(x)): raise ValueError(f"{name} must be a finite 3-vector")
    return x


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    angles: np.ndarray
    def __post_init__(self):
        object.__setattr__(self,"translation",_v3(self.translation,"translation"))
        object.__setattr__(self,"angles",_v3(self.angles,"angles"))
    @property
    def rotation(self):
        r,p,y=self.angles; cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
        Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],float)
        Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],float)
        Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],float)
        return Rz@Ry@Rx
    def apply(self,p): return np.asarray(p,float)@self.rotation.T+self.translation
    def inverse(self,p): return (np.asarray(p,float)-self.translation)@self.rotation


@dataclass(frozen=True)
class CoilGeometry:
    name: str
    shape: str
    conductor_width: float
    conductor_thickness: float
    pose: Pose
    turns: float=1.0
    outer_half_size: float=0.02
    pitch: float=0.002
    corner_radius: float|None=None
    control_points: np.ndarray|None=None

    def __post_init__(self):
        if self.shape not in {"circle","rounded_square","polyline","spline"}: raise ValueError("unsupported coil shape")
        if min(float(self.conductor_width),float(self.conductor_thickness))<=0: raise ValueError("conductor dimensions must be positive")
        if self.shape in {"circle","rounded_square"}:
            if min(float(self.turns),float(self.outer_half_size),float(self.pitch))<=0: raise ValueError("spiral dimensions must be positive")
            # pitch is the radial centerline spacing between successive turns.
            # Once the chart reaches one full turn, pitch <= conductor width
            # makes adjacent copper strips touch or overlap even though the
            # centerline itself is finite/nondegenerate.
            if float(self.turns) >= 1.0 and float(self.pitch) <= float(self.conductor_width):
                raise ValueError("spiral pitch must exceed conductor_width to prevent turn overlap")
            if self.outer_half_size-self.pitch*self.turns<=0.5*self.conductor_width: raise ValueError("spiral collapses")
            if self.shape=="rounded_square" and (
                self.corner_radius is None
                or self.corner_radius<=self.pitch*self.turns+0.5*self.conductor_width
                or self.corner_radius>=self.outer_half_size
            ):
                raise ValueError("invalid corner_radius")
        else:
            q=np.asarray(self.control_points,float)
            if q.ndim!=2 or q.shape[1]!=3 or len(q)<2 or np.any(~np.isfinite(q)):
                raise ValueError("custom coil needs finite control_points (n>=2,3)")
            if np.sum(np.linalg.norm(np.diff(q,axis=0),axis=1))<=0:
                raise ValueError("zero-length centerline")
            object.__setattr__(self,"control_points",q)

    @classmethod
    def from_mapping(cls,name,m):
        d=dict(m); pose=Pose(d.pop("translation",[0,0,0]),d.pop("angles",[0,0,0]))
        return cls(name=name,pose=pose,**d)

    @staticmethod
    def _rounded(theta,half,radius):
        c=np.abs(np.cos(theta)); s=np.abs(np.sin(theta)); center=half-radius; tiny=np.finfo(float).tiny
        tv=half/np.maximum(c,tiny); th=half/np.maximum(s,tiny); b=center*(c+s)
        disc=np.maximum(b*b-(2*center*center-radius*radius),0)
        return np.where(tv*s<=center,tv,np.where(th*c<=center,th,b+np.sqrt(disc)))

    def centerline(self,spacing):
        spacing=float(spacing)
        if not np.isfinite(spacing) or spacing<=0: raise ValueError("spacing must be finite and positive")
        if self.shape in {"polyline","spline"}:
            q=np.asarray(self.control_points,float); seg=np.linalg.norm(np.diff(q,axis=0),axis=1); dist=np.r_[0,np.cumsum(seg)]
            if dist[-1]<=0: raise ValueError("zero-length centerline")
            count=max(2,int(np.ceil(dist[-1]/spacing))+1); s=np.linspace(0,dist[-1],count)
            if self.shape=="spline" and len(q)>=4:
                from scipy.interpolate import CubicSpline
                local=np.column_stack([CubicSpline(dist,q[:,k],bc_type="natural")(s) for k in range(3)])
            else:
                local=np.column_stack([np.interp(s,dist,q[:,k]) for k in range(3)])
            return self.pose.apply(local)
        end=2*np.pi*self.turns
        estimated=max(2*np.pi*self.outer_half_size*self.turns,spacing)
        n=max(16,int(np.ceil(estimated/spacing))+1)
        th=np.linspace(0,end,n); off=self.pitch*th/(2*np.pi); half=self.outer_half_size-off
        radial=half if self.shape=="circle" else self._rounded(th,half,float(self.corner_radius)-off)
        return self.pose.apply(np.column_stack([radial*np.cos(th),radial*np.sin(th),np.zeros_like(th)]))

    def length(self,spacing):
        points=self.centerline(spacing)
        return float(np.sum(np.linalg.norm(np.diff(points,axis=0),axis=1)))

    def to_mapping(self):
        d={
            "shape":self.shape,"turns":float(self.turns),"outer_half_size":float(self.outer_half_size),
            "pitch":float(self.pitch),"conductor_width":float(self.conductor_width),
            "conductor_thickness":float(self.conductor_thickness),
            "corner_radius":None if self.corner_radius is None else float(self.corner_radius),
            "translation":self.pose.translation.tolist(),"angles":self.pose.angles.tolist(),
        }
        if self.control_points is not None: d["control_points"]=np.asarray(self.control_points).tolist()
        return d


@dataclass(frozen=True)
class PackageGeometry:
    half_extent: np.ndarray
    pose: Pose
    def __post_init__(self):
        h=_v3(self.half_extent,"half_extent")
        if np.any(h<=0): raise ValueError("package half_extent must be positive")
        object.__setattr__(self,"half_extent",h)
    def contains(self,p): return np.all(np.abs(self.pose.inverse(p))<=self.half_extent,axis=-1)
    def to_mapping(self):
        return {"half_extent":self.half_extent.tolist(),"translation":self.pose.translation.tolist(),"angles":self.pose.angles.tolist()}


@dataclass(frozen=True)
class UnifiedUWPTGeometry:
    coils: tuple
    packages: tuple
    def __post_init__(self):
        if not self.coils or len(self.coils)!=len(self.packages): raise ValueError("one package is required per coil")
        if len({c.name for c in self.coils})!=len(self.coils): raise ValueError("coil names must be unique")
    @property
    def n_ports(self): return len(self.coils)
    @classmethod
    def from_mapping(cls,m):
        d=dict(m)
        if "coils" in d:
            coils=tuple(
                CoilGeometry.from_mapping(str(v.get("name",f"port_{i}")),{k:x for k,x in v.items() if k!="name"})
                for i,v in enumerate(d["coils"])
            )
            pv=list(d.get("packages",[]))
            if len(pv)!=len(coils): raise ValueError("packages must match coils")
            packages=tuple(
                PackageGeometry(v["half_extent"],Pose(v.get("translation",c.pose.translation),v.get("angles",c.pose.angles)))
                for v,c in zip(pv,coils)
            )
            return cls(coils,packages)
        tx=CoilGeometry.from_mapping("tx",d["transmitter"]); rx=CoilGeometry.from_mapping("rx",d["receiver"])
        common=d.get("package_half_extent",[0.02,0.02,0.004])
        return cls(
            (tx,rx),
            (
                PackageGeometry(d.get("tx_package_half_extent",common),tx.pose),
                PackageGeometry(d.get("rx_package_half_extent",common),rx.pose),
            ),
        )
    def to_mapping(self):
        return {"coils":[dict(name=c.name,**c.to_mapping()) for c in self.coils],"packages":[p.to_mapping() for p in self.packages]}
    def canonical_json(self): return json.dumps(self.to_mapping(),sort_keys=True,separators=(",",":"),allow_nan=False)


def _sample(value,rule,rng):
    if not isinstance(rule,dict): return copy.deepcopy(value)
    if "choices" in rule:
        q=list(rule["choices"])
        if not q: raise ValueError("sampling choices cannot be empty")
        return copy.deepcopy(q[int(rng.integers(len(q)))])
    if "bounds" not in rule: return copy.deepcopy(value)
    b=np.asarray(rule["bounds"],float)
    if b.shape==(2,): return float(rng.uniform(*b))
    if b.shape==(3,2): return rng.uniform(b[:,0],b[:,1]).tolist()
    raise ValueError("sampling bounds must be (2,) or (3,2)")


def sample_geometry(base,sampling,rng):
    out=copy.deepcopy(dict(base))
    for section,rules in dict(sampling or {}).items():
        if section in out and isinstance(out[section],dict) and isinstance(rules,dict):
            for key,rule in rules.items():
                if key in out[section]: out[section][key]=_sample(out[section][key],rule,rng)
        elif section in out:
            out[section]=_sample(out[section],rules,rng)
    return out


__all__=["Pose","CoilGeometry","PackageGeometry","UnifiedUWPTGeometry","sample_geometry"]
