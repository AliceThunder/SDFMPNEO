"""Exact physical-source contract for independent BFZI two-port comparisons.

Exports the oriented, closed WORLD-frame polygonal current paths actually
used by the first-order Whitney edge-load operator. This is a source
GEOMETRY/phasor manifest, NOT an Elmer CoilSolver .sif or a finite-radius
copper-current distribution. Independent solvers must match source physics
before comparing impedance and E/B/J/q.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .source_reference import polygonal_port_vertices


def closed_port_source_manifest(scene, frequency_hz, *, panels_per_corner=16):
    """Return a reproducible physical-unit and winding-direction contract."""
    frequency_hz=float(frequency_hz)
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("Source manifest requires finite positive frequency")
    if type(panels_per_corner) is not int or panels_per_corner < 2:
        raise ValueError("panels_per_corner must be an integer >= 2")

    records=[]
    for port_id,assembly in enumerate(scene.assemblies,1):
        path=polygonal_port_vertices(
            assembly,panels_per_corner=panels_per_corner)
        if (path.ndim!=2 or path.shape[1]!=3 or
                not np.isfinite(path).all() or len(path)<4 or
                not np.array_equal(path[0],path[-1])):
            raise ValueError("Port must be a finite exactly closed 3D polyline")
        lengths=np.linalg.norm(np.diff(path,axis=0),axis=1)
        if np.any(lengths<=0):
            raise ValueError("Port contains a zero-length current segment")
        h=np.asarray(assembly.enclosure.half_extents,float)
        local=(path-np.asarray(assembly.pose.center,float))@(
            assembly.pose.rotation)
        if np.any(np.abs(local)+float(assembly.wire_radius)
                  >=h[None,:]+1e-12):
            raise ValueError(
                "Closed physical source, including wire radius, must "
                "stay fully inside its own sealed enclosure")
        coil=assembly.coil
        records.append({
            "port_id":port_id,
            "source_type":"unit_ampere_peak_closed_polygonal_filament",
            "amplitude_peak_A":[1.,0.],
            "weight_per_closed_path":1.,
            "world_vertices_m":path.tolist(),
            "number_of_segments":int(len(lengths)),
            "physical_chord_length_m":float(lengths.sum()),
            "path_displacement_norm_m":float(np.linalg.norm(
                np.diff(path,axis=0).sum(axis=0))),
            "wire_radius_m":float(assembly.wire_radius),
            "copper_conductivity_S_m":float(
                assembly.material.conductivity),
            "coil":{
                "turns":int(coil.turns),
                "outer_half_size_m":float(coil.outer_half_size),
                "corner_radius_m":float(coil.corner_radius),
                "pitch_m":float(coil.pitch),
                "corner_shrink_alpha":float(coil.alpha),
                "winding":int(coil.winding),
                "panels_per_corner":int(panels_per_corner),
                "return_waypoints_local_m":[list(row)
                                            for row in assembly.return_waypoints],
            },
            "enclosure":{
                "center_world_m":np.asarray(
                    assembly.pose.center,float).tolist(),
                "rotation_local_to_world":np.asarray(
                    assembly.pose.rotation,float).tolist(),
                "half_extents_local_m":h.tolist()
            },
        })

    result={
        "contract_version":"BFZI_D2_PHYSICAL_CLOSED_PORT_SOURCE_V1",
        "time_phasor":"exp(+i*omega*t)",
        "frequency_hz":frequency_hz,
        "source_current_basis":"port_1=(1+0j,0), port_2=(0,1+0j); peak phasors",
        "ports":records,
        "seawater_conductivity_S_m":float(scene.seawater.conductivity),
        "exterior":{
            "shape":"finite_cube",
            "center_m":list(scene.outer_center),
            "half_width_m":float(scene.outer_radius),
            "A_boundary":"n_cross_A_total=0",
        },
        "Whitney_excitation_and_measurement":
            "L(e,p)=integral_closed_Gamma_p w_e dot dl; "
            "Z_material uses same oriented source-path functional "
            "and same-domain zero-conductivity subtraction",
        "important_limitations":[
            "Filament path is a distributional line source; finite copper "
            "cross-section, its skin/proximity currents and copper Joule "
            "loss are not solved by this D2 reference.",
            "This JSON does NOT implement an Elmer CoilSolver source. "
            "An independent Elmer reference must explicitly implement a "
            "physics-equivalent impressed current distribution; replacing "
            "the source by a different finite-volume coil is not an "
            "identical-source comparison.",
            "MSH material physical groups 1/2/3 define seawater/tx dry/rx dry; "
            "surface 101 defines artificial total-A Dirichlet boundary.",
        ]
    }
    canonical=json.dumps(result,sort_keys=True,separators=(",",":"),
                         ensure_ascii=False)
    result["sha256_of_contract_without_hash"] = hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()
    return result


def write_closed_port_source_manifest(scene, path, frequency_hz, *,
                                     panels_per_corner=16):
    result=closed_port_source_manifest(
        scene,frequency_hz,panels_per_corner=panels_per_corner)
    target=Path(path)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                      encoding="utf-8")
    return result
