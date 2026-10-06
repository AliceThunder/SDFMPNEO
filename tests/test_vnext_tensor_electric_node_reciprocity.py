import numpy as np

from sdfmpneo_vnext.electric_tensor import (
    _laplace_kernel,
    _node_potential_kernel,
)


def test_tensor_node_potential_is_reciprocal_with_unequal_node_radii():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.011, 0.002, -0.001],
            [0.003, 0.013, 0.004],
        ],
        dtype=float,
    )
    radii = np.asarray([1.0e-4, 4.0e-4, 2.0e-4], dtype=float)
    coefficient = np.diag(
        np.asarray(
            [
                2.0 + 0.10j,
                4.0 + 0.20j,
                7.0 + 0.40j,
            ],
            dtype=complex,
        )
    )

    potential = _node_potential_kernel(
        points,
        radii,
        points,
        radii,
        coefficient,
    )

    assert np.allclose(potential, potential.T, rtol=2e-13, atol=2e-13)

    # Finite node radius is a self-term model.  Distinct-node interactions
    # must remain the unregularized tensor Green interaction.
    off_diagonal = _laplace_kernel(
        points[:1],
        points[1:],
        coefficient,
    )[0]
    assert np.allclose(
        potential[0, 1:],
        off_diagonal[0],
        rtol=2e-13,
        atol=2e-13,
    )

    for index in range(len(points)):
        self_term = _laplace_kernel(
            points[index : index + 1],
            points[index : index + 1],
            coefficient,
            source_radius=radii[index : index + 1],
        )[0][0, 0]
        assert np.allclose(
            potential[index, index],
            self_term,
            rtol=2e-13,
            atol=2e-13,
        )
