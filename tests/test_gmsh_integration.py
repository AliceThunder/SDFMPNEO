"""Actual CAD/mesh regression, enabled when the optional Gmsh runtime is installed."""
import numpy as np
import pytest
try:
    import gmsh
except (ImportError,OSError):
    gmsh=None
from sdfmpneo.spatial import SpiralCoilGeometry,RigidPose,UnderwaterWPTGeometry
from sdfmpneo.spatial.gmsh_pipeline import mesh_underwater_wpt_geometry


@pytest.mark.skipif(gmsh is None,reason='optional Gmsh runtime unavailable')
@pytest.mark.parametrize('shape',['circle','rounded_square'])
def test_real_cad_mesh_has_conforming_material_interfaces_and_terminal_groups(tmp_path,shape):
    kwargs={'corner_radius':.006} if shape=='rounded_square' else {}
    tx=SpiralCoilGeometry(shape,.5,.015,.002,.001,.001,**kwargs)
    rx=SpiralCoilGeometry(shape,.5,.015,.002,.001,.001,pose=RigidPose(np.array([0.,0.,.01])),**kwargs)
    result=mesh_underwater_wpt_geometry(UnderwaterWPTGeometry(tx,rx,np.array([.019,.019,.003]),.006),
                                       tmp_path/'coil.msh',geometry_tolerance=.0005,mesh_size=.01)
    tagged=result.tagged_mesh
    assert set(tagged.tetra_physical_tags)=={101,102,201,202,301}
    assert np.array_equal(tagged.mesh.boundary_nodes(),tagged.boundary_nodes(2001))
    for tag in [1001,1002,1003,1004]:
        assert len(tagged.boundary_nodes(tag))>0
    assert np.all(tagged.mesh.volumes>0)
