"""Actual UWPT conductor/package/seawater mesh family (optional Gmsh)."""
import numpy as np
import pytest


def test_default_uwpt_geometry_box_and_independent_solid_controls(tmp_path):
    try:
        import gmsh  # noqa: F401
    except (ImportError,OSError):
        pytest.skip('optional Gmsh runtime unavailable')
    import run
    from sdfmpneo.spatial import read_gmsh_v22_ascii
    from sdfmpneo.spatial.uwpt_family import uwpt_geometry_chart
    from sdfmpneo.spatial.geometry_chart import _tet_jacobian
    path=tmp_path/'uwpt.msh'
    run.generate_mesh(path)
    tagged=read_gmsh_v22_ascii(path)
    chart,reference=uwpt_geometry_chart(tagged,run.TRANSMITTER,run.RECEIVER,
                                        run.PHYSICAL_TAGS,run.GEOMETRY_FAMILY['parameters'])
    bounds=np.array([v*np.asarray(spec['relative']) if 'relative' in spec else spec['bounds']
                     for v,spec in zip(reference,run.GEOMETRY_FAMILY['parameters'].values())])
    cert=chart.certify_box(bounds[:,0]-reference,bounds[:,1]-reference)
    assert cert.certified_nondegenerate
    assert tagged.mesh.n_tetrahedra>100
    rng=np.random.default_rng(33)
    for g in [bounds[:,0],bounds[:,1],*rng.uniform(bounds[:,0],bounds[:,1],size=(3,len(reference)))]:
        vertices=chart.vertices(g-reference)
        assert min(np.linalg.det(_tet_jacobian(vertices,t)) for t in chart.tetrahedra)>0
    # Receiver translation moves its conductor rigidly, leaving TX conductor fixed.
    rx=np.unique(tagged.mesh.tetrahedra[tagged.tetra_mask(run.PHYSICAL_TAGS['rx_copper'])])
    tx=np.unique(tagged.mesh.tetrahedra[tagged.tetra_mask(run.PHYSICAL_TAGS['tx_copper'])])
    p=np.zeros(len(reference));p[chart.parameter_names.index('rx_gap')]=.0001
    change=chart.vertices(p)-chart.reference_vertices
    assert np.allclose(change[rx],[0,0,.0001])
    assert np.allclose(change[tx],0)
    # A seawater-radius change preserves all conductor coordinates.
    p[:]=0;p[chart.parameter_names.index('seawater_radius')]=.0001
    change=chart.vertices(p)-chart.reference_vertices
    assert np.allclose(change[np.union1d(tx,rx)],0)
