"""BFZI geometry-to-response labels, with explicit unverified reference status.

A record stores the *gauge-independent* complex TWO-port physical raw mode
coefficients (not the rotating four-dimensional SVD coordinates). That
makes records from varying geometry and rigid poses directly comparable
provided they use the same deterministic mode topology/order.

These are DATA CONTRACTS for a future neural FAST model; this module
neither trains a model nor certifies the finite-domain MQS reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .sea_snapshot import BFZISnapshotROM, _scene_signature
from .source import BFZISources
from .sea_basis import SeaCurrentBasis
from .geometry import BFZICoil, RigidPose
from .scene import BFZIAssembly, BfziScene, InsulatingBox, Seawater
from sdfmpneo_vnext.scene import ConductorMaterial


SCHEMA_VERSION = 1


def geometry_features(sources: BFZISources, *, max_return_waypoints: int = 8):
    """Lossless fixed-width features for two current BFZI coil/box/source inputs.

    Coordinates are in the world frame; the full two 3x3 rigid rotations,
    two coil descriptions, enclosures, return-waypoint routes, conductor
    properties, seawater conductivity, frequency and outer domain are
    represented. The count+padding makes varying return routes explicit;
    no geometry is silently truncated or assumed symmetric.
    """
    if (not isinstance(max_return_waypoints,int)
            or isinstance(max_return_waypoints,bool)
            or max_return_waypoints < 2):
        raise ValueError("max_return_waypoints must be an integer >= 2")
    s=sources.scene
    values=[
        sources.omega/(2*np.pi),
        s.seawater.conductivity,
        s.outer_radius,
        *s.outer_center,
    ]
    for a in s.assemblies:
        n=len(a.return_waypoints)
        if n > max_return_waypoints:
            raise ValueError("return route exceeds feature capacity")
        coil=a.coil
        material=a.material
        values.extend((
            coil.turns, coil.outer_half_size, coil.corner_radius,
            coil.pitch, coil.alpha, coil.winding,
            *a.pose.center, *a.pose.rotation.reshape(-1),
            *a.enclosure.half_extents, a.wire_radius,
            material.conductivity, material.relative_permeability,
            material.reference_temperature,
            material.resistance_temperature_coefficient,
            n,
        ))
        route=np.zeros((max_return_waypoints,3))
        route[:n]=np.asarray(a.return_waypoints)
        values.extend(route.ravel())
    features=np.asarray(values,dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("BFZI features must be finite")
    return features


def scene_from_geometry_features(features, *, max_return_waypoints: int = 8):
    """Reconstruct the exact BFZI physical scene and frequency from a label.

    Enables a genuinely geometry-held-out field/Joule test against the
    *stored* current-mode label without re-solving the sea MQS reference.
    Complete line-source quadrature order is not stored in schema v1:
    callers must use the order originally used for label generation.
    """
    if (not isinstance(max_return_waypoints, int)
            or isinstance(max_return_waypoints, bool)
            or max_return_waypoints < 2):
        raise ValueError("max_return_waypoints must be an integer >=2")
    vector=np.asarray(features,dtype=float)
    expected=6+2*(27+3*max_return_waypoints)
    if (vector.shape!=(expected,) or not np.isfinite(vector).all()):
        raise ValueError(
            f"geometry feature vector must contain {expected} finite scalars"
        )
    offset=0

    def take(count):
        nonlocal offset
        part=vector[offset:offset+count]
        offset+=count
        return part

    def integer(value,name):
        if not np.isfinite(value) or abs(value-round(value))>1e-8:
            raise ValueError(f"{name} must be integer-valued")
        return int(round(value))

    frequency,conductivity,outer_radius=take(3)
    center=tuple(take(3))
    assemblies=[]
    for port in range(2):
        turns,outer_half,corner,pitch,alpha,winding=take(6)
        coil=BFZICoil(
            integer(turns,"turns"),outer_half,corner,pitch,
            alpha,integer(winding,"winding"),
        )
        pose_center=take(3)
        rotation=take(9).reshape(3,3)
        half_extents=tuple(take(3))
        wire_radius,metal_sigma,mu_r,reference_t,alpha_r=take(5)
        count=integer(take(1)[0],"return waypoint count")
        if not 2<=count<=max_return_waypoints:
            raise ValueError("invalid number of return waypoints")
        waypoint_block=take(3*max_return_waypoints).reshape(
            max_return_waypoints,3,
        )
        if np.any(waypoint_block[count:]!=0):
            raise ValueError("nonzero padding in stored return route")
        assemblies.append(BFZIAssembly(
            coil=coil,
            pose=RigidPose(rotation,pose_center),
            enclosure=InsulatingBox(half_extents),
            return_waypoints=tuple(map(tuple,waypoint_block[:count])),
            wire_radius=float(wire_radius),
            material=ConductorMaterial(
                float(metal_sigma),float(mu_r),
                float(alpha_r),float(reference_t),
            ),
        ))
    if offset!=expected:
        raise RuntimeError("feature schema was not fully decoded")
    return BfziScene(
        assemblies[0],assemblies[1],Seawater(conductivity),
        outer_radius,center,
    ),float(frequency)


@dataclass(frozen=True)
class BFZIResponseLabel:
    """One independent geometry/pose/frequency reference response.

    `raw_unit_modes` has shape (n_physical_modes,2), complex: physical curl
    basis coefficients for each unit phasor port. Unlike four-SVD-mode
    projections it has no gauge/sign/rotation ambiguity across samples.
    `impedance_seawater` is the physical 2x2 symmetric sea reaction.

    "provisional" is mandatory until spatial, magnetic and domain
    convergence gates are independently validated for the dataset.
    """

    features: np.ndarray
    raw_unit_modes: np.ndarray
    impedance_seawater: np.ndarray
    scene_signature: str
    reference_status: str = "provisional"

    @classmethod
    def from_snapshot(
        cls, snapshot: BFZISnapshotROM, *,
        max_return_waypoints: int = 8,
    ) -> "BFZIResponseLabel":
        if snapshot.signature != _scene_signature(
            snapshot.sources, snapshot.basis,
        ):
            raise ValueError("snapshot geometry/source mismatch")
        features=geometry_features(
            snapshot.sources,max_return_waypoints=max_return_waypoints,
        )
        raw=snapshot.raw_projection @ snapshot.unit_modes
        return cls(features, raw, snapshot.impedance_seawater.copy(),
                   snapshot.signature)

    def __post_init__(self):
        f=np.asarray(self.features,dtype=float)
        c=np.asarray(self.raw_unit_modes,dtype=complex)
        z=np.asarray(self.impedance_seawater,dtype=complex)
        if (f.ndim!=1 or c.ndim!=2 or c.shape[1]!=2
                or z.shape!=(2,2) or
                not all(np.isfinite(x).all() for x in (f,c,z))
                or not isinstance(self.scene_signature,str)
                or len(self.scene_signature)!=64
                or self.reference_status!="provisional"):
            raise ValueError("invalid or unvalidated BFZI reference label")
        if not np.allclose(z,z.T,rtol=1e-9,atol=1e-12):
            raise ValueError("BFZI reaction matrix must be reciprocal")
        if np.linalg.eigvalsh(z.real)[0]<-1e-10:
            raise ValueError("BFZI reaction matrix must be passive")


def save_response_labels(path,labels):
    """Save several independently generated geometries; never imply certification.

    Rejects repeated signatures (train/validation leakage) and mismatched
    physical basis/feature dimensions. A real trainer must still hold out
    *whole geometries*, not random query points from the same geometry.
    """
    samples=tuple(labels)
    if not samples or not all(isinstance(v,BFZIResponseLabel) for v in samples):
        raise ValueError("at least one BFZIResponseLabel is required")
    if len({s.scene_signature for s in samples})!=len(samples):
        raise ValueError("duplicate geometry/source signatures in labels")
    if len({s.features.tobytes() for s in samples})!=len(samples):
        raise ValueError("duplicate geometry features in labels")
    nfeat=samples[0].features.shape
    nraw=samples[0].raw_unit_modes.shape
    if any(s.features.shape!=nfeat or s.raw_unit_modes.shape!=nraw
           for s in samples):
        raise ValueError("mixed feature width or modal topology in labels")
    np.savez_compressed(
        path,
        schema_version=np.array(SCHEMA_VERSION,dtype=np.int64),
        reference_status=np.array("provisional"),
        geometry_features=np.stack([s.features for s in samples]),
        raw_unit_modes=np.stack([s.raw_unit_modes for s in samples]),
        impedance_seawater=np.stack([s.impedance_seawater for s in samples]),
        scene_signatures=np.array([s.scene_signature for s in samples]),
    )


def load_response_labels(path) -> tuple[BFZIResponseLabel,...]:
    """Load numeric samples without pickle; enforce unverified status."""
    with np.load(path,allow_pickle=False) as data:
        if (int(data["schema_version"])!=SCHEMA_VERSION
                or str(data["reference_status"])!="provisional"):
            raise ValueError("unsupported BFZI labels schema/reference status")
        x=np.array(data["geometry_features"],copy=True)
        y=np.array(data["raw_unit_modes"],copy=True)
        z=np.array(data["impedance_seawater"],copy=True)
        signatures=np.array(data["scene_signatures"],copy=True)
    if (x.ndim!=2 or y.ndim!=3 or y.shape[0]!=len(x)
            or z.shape!=(len(x),2,2)
            or signatures.shape!=(len(x),)):
        raise ValueError("corrupt BFZI dataset shapes")
    result=tuple(
        BFZIResponseLabel(x[i],y[i],z[i],str(signatures[i]))
        for i in range(len(x))
    )
    if len({s.scene_signature for s in result})!=len(result):
        raise ValueError("duplicate geometry/source signatures in labels")
    return result
