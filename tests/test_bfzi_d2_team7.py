"""Minimal TEAM-7 entry checks; never assert experimental agreement on smoke mesh."""
import numpy as np
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import solve_aphi
from sdfmpneo_bfzi.direct3d_team7 import (
    team7_geo, racetrack_vertices, team7_stranded_source,
    official_probe_lines, coarse_team7_tetra_mesh, graded_team7_smoke_mesh,
)
from sdfmpneo_bfzi.direct3d_probe import sample_cell_fields


def test_official_team7_geometry_and_source():
    geo=team7_geo()
    assert '0.294,0.294,0.019' in geo
    assert '0.018,0.018' in geo
    assert 'BooleanDifference' in geo and 'BooleanFragments' in geo
    assert 'Mesh.SaveAll' not in geo  # physical id is needed by MSH2 importer
    path=racetrack_vertices(.0125,.1,corners=8)
    assert np.array_equal(path[0],path[-1])
    assert np.min(np.linalg.norm(np.diff(path,axis=0),axis=1))>0
    probes=official_probe_lines()
    assert_allclose(probes['A1_B1_Bz'][0],[0.,.072,.034])
    assert_allclose(probes['A2_B2_Bz'][-1],[.288,.144,.034])
    assert_allclose(probes['A3_B3_Jy_top'][0],[0.,.072,.019])


def test_team7_first_3d_solve_and_probe_power():
    mesh=coarse_team7_tetra_mesh()
    assert mesh.tetrahedra.shape[0]==750
    assert np.count_nonzero(mesh.sigma)==48
    A,B=team7_stranded_source(2,2,5)
    for f in (50,200):
        solution=solve_aphi(mesh,f,[A])
        assert solution.power_matrix[0,0].real>0
        assert_allclose(solution.power_matrix,solution.power_from_fields,
                        rtol=1e-9,atol=1e-6)
        probes=official_probe_lines()
        for name,points in probes.items():
            fields=sample_cell_fields(solution,points,
                                     incident_magnetic=[B],
                                     prefer_conductor=('Jy_' in name),
                                     nearest=256)
            assert np.all(np.isfinite(fields['B_total_T']))
            if 'Jy_' in name:
                assert np.any(fields['sigma_S_m']>0)


def test_team7_conforming_plate_geometry_stable_on_refinement():
    """Both physical conductor volume and drilled eccentric void are fixed."""
    reference_volume = (.294**2 - .108**2)*.019
    for level, total, metal in ((1,750,48),(2,2304,384),
                                (3,5082,1296),(4,9408,3072)):
        m = graded_team7_smoke_mesh(level)
        assert len(m.tetrahedra)==total
        assert int(np.count_nonzero(m.sigma))==metal
        coords = m.nodes[m.tetrahedra]
        transform = (coords[:,1:] - coords[:,0,None,:]).transpose(0,2,1)
        volume = np.abs(np.linalg.det(transform))/6
        assert_allclose(volume[m.sigma>0].sum(),reference_volume,
                        rtol=0,atol=2e-18)
