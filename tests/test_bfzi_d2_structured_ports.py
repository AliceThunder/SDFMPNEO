"""Real closed BFZI two-port A-phi integration on conforming dry/wet geometry.

This is the bounded axis-aligned D2 end-to-end entry, NOT a high-fidelity
rotated-SE3 / open-seawater convergence certificate.
"""
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.direct3d_structured_bfzi import axis_aligned_bfzi_tetra_mesh
from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports
from sdfmpneo_bfzi.source_reference import polygonal_port_vertices


def test_two_real_continuous_bfzi_closed_ports_on_conforming_3d_mesh():
    scene=replace(build_demo_scene(),outer_radius=.32)
    mesh=axis_aligned_bfzi_tetra_mesh(
        scene,near_padding_m=.01,far_step_m=.30,box_layers=1)
    ports=[[(1.,polygonal_port_vertices(
        assembly,panels_per_corner=2))]
        for assembly in scene.assemblies]
    assert all(np.array_equal(p[0][1][0],p[0][1][-1]) for p in ports)
    reference=solve_closed_ports(mesh,200000.,ports)
    delta=reference.material_increment_z
    assert delta.shape==(2,2) and np.isfinite(delta).all()
    assert_allclose(delta,delta.T,rtol=1e-8,atol=2e-10)
    assert np.linalg.norm(delta)>0
    for currents in ((1.,0.),(0.,1.),(1.,1.),(1.,-1.),(1.,1j)):
        I=np.asarray(currents,complex)
        joule=reference.conductor_power(I)
        assert joule>=0
        assert_allclose(reference.port_power(I),joule,rtol=1e-7,atol=1e-10)
    # The source-space Galerkin identity is just an internal consistency
    # check; it is NOT a proof of corrected spatial E/B/J/q accuracy.
    q=reference.joule_density(np.array([1.,1j]))
    assert np.all(np.isfinite(q)) and np.max(q)>0
