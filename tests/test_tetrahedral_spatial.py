import numpy as np

from sdfmpneo.spatial import TetrahedralComplex3D, assemble_weighted_nedelec_mass


def centered_tetrahedral_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ]
    )
    tetrahedra = np.array(
        [
            [4, 1, 2, 3],
            [0, 4, 2, 3],
            [0, 1, 4, 3],
            [0, 1, 2, 4],
        ],
        dtype=int,
    )
    return TetrahedralComplex3D.build(vertices, tetrahedra)


def test_tetrahedral_complex_exact_topology_and_tree_cotree():
    mesh = centered_tetrahedral_mesh()
    assert mesh.n_nodes == 5
    assert mesh.n_edges == 10
    assert mesh.n_faces == 10
    assert mesh.n_tetrahedra == 4
    assert mesh.topology_defect() == 0.0
    assert (mesh.curl @ mesh.grad).nnz == 0

    tree, cotree = mesh.tree_cotree_edges()
    assert tree.size == mesh.n_nodes - 1
    assert cotree.size == mesh.n_edges - mesh.n_nodes + 1
    R = mesh.gauge_basis()
    assert R.shape == (mesh.n_edges, cotree.size)
    assert np.allclose((R.T @ R).toarray(), np.eye(cotree.size))


def test_tetrahedral_p1_thermal_has_one_positive_interior_degree_of_freedom():
    mesh = centered_tetrahedral_mesh()
    thermal = mesh.assemble_p1_thermal(
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        homogeneous_dirichlet_boundary=True,
    )
    assert np.array_equal(thermal.boundary_nodes, np.array([0, 1, 2, 3]))
    assert np.array_equal(thermal.free_nodes, np.array([4]))
    assert thermal.M.shape == (1, 1)
    assert thermal.K.shape == (1, 1)
    assert thermal.M[0, 0] > 0.0
    assert thermal.K[0, 0] > 0.0


def test_exact_weighted_nedelec_mass_reduces_to_constant_mass_for_unit_p1_factors():
    mesh = centered_tetrahedral_mesh()
    coefficient = np.array([1.2, 0.7, 2.0, 1.5])
    reference = mesh.assemble_edge_mass(coefficient).toarray()

    no_factor = assemble_weighted_nedelec_mass(
        mesh,
        scale_tetra=coefficient,
    ).toarray()
    one_factor = assemble_weighted_nedelec_mass(
        mesh,
        scale_tetra=coefficient,
        p1_factors=np.ones((1, mesh.n_tetrahedra, 4)),
    ).toarray()
    two_factors = assemble_weighted_nedelec_mass(
        mesh,
        scale_tetra=coefficient,
        p1_factors=np.ones((2, mesh.n_tetrahedra, 4)),
    ).toarray()

    assert np.allclose(no_factor, reference, rtol=2e-13, atol=2e-13)
    assert np.allclose(one_factor, reference, rtol=2e-13, atol=2e-13)
    assert np.allclose(two_factors, reference, rtol=2e-13, atol=2e-13)
    assert np.allclose(one_factor, one_factor.T, rtol=0.0, atol=2e-13)
