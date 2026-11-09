"""D2 method-level verification of PHYSICAL closed-port Galerkin excitation.

This tests 3D A-phi field construction and its dual source/voltage mapping,
not absolute TEAM-7 experiment accuracy, infinite ocean, or copper self Z.
"""
import numpy as np
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import box_tetra_mesh, TetraMesh
from sdfmpneo_bfzi.direct3d_edge_source import (
    closed_edge_load, elemental_current_edge_load,
    source_nodal_conservation)
from sdfmpneo_bfzi.direct3d_galerkin import (
    solve_closed_ports,solve_elemental_current_ports)
from sdfmpneo_bfzi.direct3d_team7 import (
    graded_team7_smoke_mesh, team7_stranded_source,
)


def _rectangle(x):
    return np.array([
        [x-.05,-.05,0.],[x+.05,-.05,0.],
        [x+.05,.05,0.],[x-.05,.05,0.],[x-.05,-.05,0.],
    ])


def test_source_is_closed_and_obeys_reversal():
    m = box_tetra_mesh([[-.3,.3]]*3,[3,3,3],lambda x:0.)
    loop = _rectangle(0.)
    a = closed_edge_load(m,[[(1.,loop)]])
    b = closed_edge_load(m,[[(1.,loop[::-1])]])
    assert_allclose(a,-b,rtol=0,atol=1e-12)
    assert np.linalg.norm(a)>0
    assert source_nodal_conservation(m,a)[0]<1e-10


def test_two_real_ports_reciprocity_and_complex_phase_power():
    mesh = box_tetra_mesh([[-.3,.3]]*3,[5,5,5],
                          lambda x:5. if x[2]>.1 else 0.)
    ports = [[(1.,_rectangle(-.07))],[(1.,_rectangle(.08))]]
    r = solve_closed_ports(mesh,100000.,ports)
    assert_allclose(r.material_increment_z,
                    r.material_increment_z.T,
                    rtol=1e-9,atol=1e-11)
    assert abs(r.material_increment_z[0,1])>1e-9
    for current in ((1.,0.),(0.,1.),(1.,1.),(1.,-1.),(1.,1j)):
        i = np.asarray(current,complex)
        assert r.conductor_power(i)>=0
        assert_allclose(r.port_power(i),r.conductor_power(i),
                        rtol=1e-8,atol=1e-12)


def test_same_finite_domain_vacuum_reference_has_no_material_loss():
    mesh = box_tetra_mesh([[-.3,.3]]*3,[4,4,4],lambda x:0.)
    r = solve_closed_ports(mesh,120000.,[[(1.,_rectangle(0.))]])
    assert_allclose(r.material_increment_z,0.,rtol=0,atol=1e-12)
    assert_allclose(r.conductor_power(np.array([1.])),0.,atol=1e-12)
    assert r.vacuum_port_z[0,0].imag>0


def test_real_team7_metal_current_50_and_200Hz():
    # Coarse TEAM-7 is only a reference *method* smoke, not measured fit.
    m = graded_team7_smoke_mesh(1)
    _, _, strands = team7_stranded_source(
        2,2,5,return_strands=True)
    for f in (50,200):
        r = solve_closed_ports(m,f,[strands])
        assert r.port_power(np.array([1.]))>0.
        assert_allclose(r.port_power(np.array([1.])),
                        r.conductor_power(np.array([1.])),
                        rtol=1e-9,atol=1e-9)
        assert_allclose(r.material_increment_z,
                        r.material_increment_z.T,atol=1e-11)
        # The analytically known coil source is reconstructed independently
        # for spatial E/Joule. It is deliberately NOT forced to equal the
        # coarse Galerkin port power: that gap detects source-space error.
        A_source, _, _ = team7_stranded_source(
            2,2,5,return_strands=True)
        q = r.analytic_source_corrected_joule(f,[A_source])
        assert q.shape == (1,1)
        assert np.isfinite(q).all() and q[0,0].real >= 0
        assert_allclose(q,q.conj().T,rtol=1e-12,atol=1e-12)


def test_tree_gauge_matches_mixed_physical_fields_on_two_real_ports():
    """Gauge DOFs differ; physical 3-D E/B, port and Joule must not."""
    mesh=box_tetra_mesh([[-.3,.3]]*3,[3,3,3],
                        lambda x:5. if x[2]>.1 else 0.)
    strands=[[(1.,_rectangle(-.07))],[(1.,_rectangle(.08))]]
    fast=solve_closed_ports(mesh,200000.,strands,gauge='tree')
    mixed=solve_closed_ports(mesh,200000.,strands,gauge='mixed')
    assert fast.medium.n_gauge_dofs==0
    assert mixed.medium.n_gauge_dofs>0
    assert_allclose(fast.material_increment_z,mixed.material_increment_z,
                    atol=3e-10,rtol=1e-8)
    # In the MQS A-phi model phi is solved ONLY inside conducting cells.
    # In sigma=0 cells, -i*omega*A by itself is gauge dependent: comparing
    # those values would erroneously test the gauge rather than a physical E.
    # Conductive E=-i*omega*A-grad(phi) must agree pointwise.
    wet=mesh.sigma>0
    assert np.any(wet) and np.any(~wet)
    assert_allclose(fast.medium.electric_at_quadrature[:,wet],
                    mixed.medium.electric_at_quadrature[:,wet],
                    atol=3e-8,rtol=1e-8)
    assert_allclose(fast.medium.magnetic_secondary,
                    mixed.medium.magnetic_secondary,
                    atol=5e-11,rtol=1e-8)
    for currents in ((1.,0.),(0.,1.),(1.,1j)):
        assert_allclose(fast.conductor_power(currents),
                        mixed.conductor_power(currents),
                        atol=1e-10,rtol=1e-8)
        assert_allclose(fast.port_power(currents),
                        fast.conductor_power(currents),
                        atol=1e-10,rtol=1e-8)


def test_elemental_current_whitney_source_exact_on_oriented_tetra():
    """Analytic constant-J volume integral of each Nedelec basis, 2 ports."""
    nodes=np.array([[0.,0.,0.],[1.,0.,0.],
                    [0.,1.,0.],[0.,0.,1.]])
    density=np.array([[[.7,-.2,.9]],[[.1,.8,-.4]]])
    mesh=TetraMesh(nodes,np.array([[0,1,2,3]]),np.zeros(1))
    load=elemental_current_edge_load(mesh,density)
    grads=np.array([[-1.,-1.,-1.],[1.,0.,0.],
                    [0.,1.,0.],[0.,0.,1.]])
    pairs=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    expected=np.array([
        [np.dot(J,(grads[j]-grads[i]))/24
         for i,j in pairs] for J in density[:,0]])
    assert_allclose(load,expected,rtol=0,atol=1e-15)
    # Flipping tetra vertex storage must not flip globally oriented edges.
    reversed_mesh=TetraMesh(
        nodes,np.array([[1,0,2,3]]),np.zeros(1))
    # _topology assigns consecutive edge indices in FIRST-SEEN order.
    # Reordering local tetra vertices changes that enumeration without
    # changing the physical/global orientation (always min-node -> max-node).
    from sdfmpneo_bfzi.direct3d_aphi import _topology
    original_edges,*_=_topology(mesh)
    permuted_edges,*_=_topology(reversed_mesh)
    aligned=elemental_current_edge_load(reversed_mesh,density)
    aligned=aligned[:,[permuted_edges[key] for key in original_edges]]
    assert_allclose(aligned,load,rtol=0,atol=1e-15)


def test_complex_volume_source_requires_separate_terminal_functional():
    """Do not report a phase-dependent pseudo-Z for complex source basis."""
    import pytest
    mesh=TetraMesh(np.array([[0.,0.,0.],[1.,0.,0.],
                             [0.,1.,0.],[0.,0.,1.]]),
                   np.array([[0,1,2,3]]),np.zeros(1))
    with pytest.raises(ValueError,match='Complex elemental source basis'):
        solve_elemental_current_ports(
            mesh,100000.,np.array([[[1j,0.,0.]]]))



def test_nonconserving_volume_source_is_refused_before_aphi():
    """A valid Whitney integral of arbitrary J is NOT a closed-coil source."""
    import pytest
    mesh=TetraMesh(np.array([[0.,0.,0.],[1.,0.,0.],
                             [0.,1.,0.],[0.,0.,1.]]),
                   np.array([[0,1,2,3]]),np.zeros(1))
    J=np.array([[[1.,0.,0.]]])
    L=elemental_current_edge_load(mesh,J)
    assert_allclose(source_nodal_conservation(mesh,L),[1.],
                    rtol=0,atol=1e-14)
    with pytest.raises(ValueError,match="charge-conserving"):
        solve_elemental_current_ports(mesh,100000.,J)
