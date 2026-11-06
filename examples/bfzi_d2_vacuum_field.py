"""Independent D2 vacuum B-field benchmark against exact finite Biot-Savart.

    python -m examples.bfzi_d2_vacuum_field --levels 4 6 8

Reports actual raw 3D Nedelec curl(A) errors for an off-wire polygonal
closed port. No paper-value matching; this is a direct-method field gate.
Primary-secondary reconstruction analytically restores the vacuum source
but must NOT be taken as an independent test of the material reaction.
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import TetraMesh, solve_aphi
from sdfmpneo_bfzi.direct3d_edge_source import closed_edge_load
from sdfmpneo_bfzi.direct3d_probe import (
    sample_cell_fields, closed_polyline_magnetic,
)


def graded_vacuum_mesh(n):
    """Same finite outer cube [-.4,.4]^3; refine central [-.15,.15]^3."""
    if not isinstance(n,int) or n < 2:
        raise ValueError("Central refinement n must be at least 2")
    axis=np.r_[[-.4,-.25,-.15],
                np.linspace(-.15,.15,n+1)[1:-1],
                [.15,.25,.4]]
    dims=(len(axis)-1,)*3
    points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    points=points.reshape(-1,3)
    def index(i,j,k):
        return (i*(dims[1]+1)+j)*(dims[2]+1)+k
    six=((0,1,3,7),(0,3,2,7),(0,2,6,7),
         (0,6,4,7),(0,4,5,7),(0,5,1,7))
    cells=[]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                corner=(index(i,j,k),index(i+1,j,k),index(i,j+1,k),
                        index(i+1,j+1,k),index(i,j,k+1),
                        index(i+1,j,k+1),index(i,j+1,k+1),
                        index(i+1,j+1,k+1))
                cells.extend([corner[v] for v in local] for local in six)
    return TetraMesh(points,np.asarray(cells,int),np.zeros(len(cells)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--levels',nargs='+',type=int,default=[4,6,8])
    p.add_argument('--frequency-hz',type=float,default=1000.)
    args=p.parse_args()
    path=np.array([
        [-.065,-.065,0.],[.065,-.065,0.],
        [.065,.065,0.],[-.065,.065,0.],[-.065,-.065,0.]])
    probes=np.array([[.025,.012,.17],[-.04,.021,.18],
                     [.1,.04,.16],[0.,0.,.18]])
    exact=closed_polyline_magnetic(path)(probes)
    output=[]
    for n in args.levels:
        mesh=graded_vacuum_mesh(n)
        load=closed_edge_load(mesh,[[(1.,path)]])
        def zero(x):
            return np.zeros_like(x)
        solution=solve_aphi(mesh,args.frequency_hz,[zero],
                            edge_source_load=load)
        actual=sample_cell_fields(solution,probes)['B_secondary_T'][0]
        relative=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
        output.append({
            'central_cells_per_axis':n,
            'n_tetrahedra':len(mesh.tetrahedra),
            'vacuum_curl_A_vs_BiotSavart_relative_l2':relative,
            'Bz_FEM_T':[float(x.real) for x in actual[:,2]],
            'Bz_BiotSavart_T':[float(x) for x in exact[:,2]],
        })
    print(json.dumps({
        'status':'D2 vacuum spatial error measured; not D2 PASS',
        'outer_cube_half_extent_m':.4,
        'note':'Raw FEM curl(A), NOT vacuum-corrected B. No conductive domain.',
        'levels':output,
    },indent=2))


if __name__=='__main__':
    main()
