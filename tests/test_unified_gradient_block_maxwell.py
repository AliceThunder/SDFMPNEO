import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_accurate_residual import accurate_residual_vector
from sdfmpneo.unified_background import BackgroundContext
from sdfmpneo.unified_compensated_field import field_parts
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo.unified_gradient_block_maxwell import (
    _edge_mass_diagonal,
    build_gradient_block,
    compose_block_preconditioner,
    gradient_operator,
    source_terminal_divergence,
)
from sdfmpneo.unified_refined_gradient_projection import refined_gradient_projection
from sdfmpneo.unified_transverse_ilu import (
    build_transverse_ilu,
    build_transverse_stabilized_matrix,
)
import sdfmpneo.unified_certified_local_solve as local_solver


def _background():
    materials = {
        "wire": {
            "electrical_conductivity": 5.8e7,
            "relative_permittivity": 1.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
        "package": {
            "electrical_conductivity": 0.0,
            "relative_permittivity": 3.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
        "sea": {
            "electrical_conductivity": 5.0,
            "relative_permittivity": 80.0,
            "relative_permeability": 1.0,
            "thermal_conductivity": 1.0,
            "volumetric_heat_capacity": 1.0,
        },
    }
    return OpenBoundaryBackground(
        np.linspace(-0.03, 0.03, 5),
        np.linspace(-0.03, 0.03, 5),
        np.linspace(-0.03, 0.03, 5),
        frequency_hz=1.0e5,
        materials=materials,
        coil_materials=("wire",),
        package_materials=("package",),
        seawater_material="sea",
        ambient_temperature=293.15,
    )


def _sea_context(bg):
    fractions = {
        "wire": np.zeros(bg.n_cells),
        "package": np.zeros(bg.n_cells),
        "sea": np.ones(bg.n_cells),
    }
    return BackgroundContext(None, fractions, np.zeros((bg.n_edges, 1)), (np.ones(bg.n_cells) / bg.n_cells,))


def test_cartesian_gradient_is_exactly_curl_free_and_scalar_block_factors():
    bg = _background()
    context = _sea_context(bg)
    G = gradient_operator(bg, gauge_fixed=True)
    CG = (bg.curl @ G).tocsr()
    CG.eliminate_zeros()
    assert CG.nnz == 0 or np.max(np.abs(CG.data)) <= 1e-14

    block = build_gradient_block(bg, context)
    assert block.scalar_dofs == G.shape[1]
    assert block.topology_error <= 1e-14


def test_open_path_source_has_balanced_nonzero_terminal_divergence():
    bg = _background()
    source = np.zeros(bg.n_edges)
    j = 2
    k = 2
    for i in range(bg.nx):
        source[bg.edge_maps[0][(i, j, k)]] = 1.0 / bg.dx[i]

    q, net_error, moment = source_terminal_divergence(bg, source)
    expected = np.array([bg.nx, 0.0, 0.0])
    # source[e] * edge_length == 1 for every path edge, so the first moment is
    # the number of oriented edge steps along x.
    assert np.linalg.norm(q) > 0.0
    assert net_error <= 1e-14
    assert np.allclose(moment, expected, rtol=0.0, atol=1e-13)


def test_transverse_stabilization_vanishes_on_compatible_transverse_space():
    bg = _background()
    context = _sea_context(bg)
    A = bg.em_operator(context, None)
    block = build_gradient_block(bg, context)
    G = block.gradient
    d = _edge_mass_diagonal(bg, context)

    rng = np.random.default_rng(17)
    value = rng.normal(size=bg.n_edges) + 1j * rng.normal(size=bg.n_edges)
    scalar_rhs = np.asarray(G.T @ (d * value), complex).reshape(-1)
    phi = np.asarray(block.factor.solve(scalar_rhs), complex).reshape(-1)
    transverse = np.asarray(value - G @ phi, complex).reshape(-1)
    constraint = np.linalg.norm(G.T @ (d * transverse))
    reference = max(np.linalg.norm(G.T @ (d * value)), np.finfo(float).tiny)
    assert constraint / reference <= 1e-10

    augmented, stats = build_transverse_stabilized_matrix(
        A,
        bg,
        context,
        block,
        stabilization_factor=3e-2,
    )
    added_action = np.asarray((augmented - A) @ transverse, complex).reshape(-1)
    physical_action = np.asarray(A @ transverse, complex).reshape(-1)
    assert stats["augmentation_gain"] >= 1.0
    assert np.linalg.norm(added_action) / max(np.linalg.norm(physical_action), np.finfo(float).tiny) <= 1e-9


def test_gradient_block_preconditioner_certifies_original_maxwell_equation():
    bg = _background()
    context = _sea_context(bg)
    A = bg.em_operator(context, None)
    G = gradient_operator(bg, gauge_fixed=True)
    rng = np.random.default_rng(7)
    phi = rng.normal(size=G.shape[1])
    transverse = rng.normal(size=bg.n_edges) + 1j * rng.normal(size=bg.n_edges)
    x_true = np.asarray(G @ phi, complex) + 1e-3 * transverse
    rhs = A @ x_true

    block = build_gradient_block(bg, context)
    edge_M, _stats = build_transverse_ilu(
        A,
        bg,
        context,
        block,
        drop_tol=1e-3,
        fill_factor=8.0,
        stabilization_factor=3e-2,
    )
    M = compose_block_preconditioner(A, edge_M, block, post_correct=True)
    x, info = local_solver._lgmres(
        A,
        rhs,
        x0=np.zeros(bg.n_edges, dtype=complex),
        M=M,
        rtol=1e-11,
        maxiter=20,
        inner_m=24,
    )
    residual = np.linalg.norm(rhs - A @ x) / np.linalg.norm(rhs)
    assert info == 0
    assert residual <= 1e-10


def test_direct_compatible_split_recovers_small_transverse_component():
    bg = _background()
    context = _sea_context(bg)
    A = bg.em_operator(context, None)
    block = build_gradient_block(bg, context)
    G = block.gradient
    d = _edge_mass_diagonal(bg, context)
    mass = sp.diags(d, format="csr")
    rng = np.random.default_rng(29)

    # Build a genuinely D-transverse field, then bury it under a longitudinal
    # component eight orders of magnitude larger.  The physical RHS is assembled
    # from the exact compatible split so the test probes the solve decomposition,
    # not cancellation in a synthetic full-field matvec.
    candidate = rng.normal(size=bg.n_edges) + 1j * rng.normal(size=bg.n_edges)
    scalar_rhs = np.asarray(G.T @ (d * candidate), complex).reshape(-1)
    correction = np.asarray(block.factor.solve(scalar_rhs), complex).reshape(-1)
    transverse_true = np.asarray(candidate - G @ correction, complex).reshape(-1)
    phi = 1.0e8 * (rng.normal(size=G.shape[1]) + 1j * rng.normal(size=G.shape[1]))
    longitudinal_true = np.asarray(G @ phi, complex).reshape(-1)
    rhs = np.asarray(d * longitudinal_true + A @ transverse_true, complex).reshape(-1)

    longitudinal, projection = refined_gradient_projection(
        bg,
        block,
        rhs,
        relative_tolerance=5e-13,
        maximum_refinements=5,
    )
    transverse_rhs, _diag = accurate_residual_vector(
        mass,
        longitudinal,
        rhs,
        target_relative=5e-13,
    )
    transverse, _lu = local_solver._direct_solve(A, transverse_rhs)
    residual, _ = accurate_residual_vector(
        A,
        transverse,
        transverse_rhs,
        target_relative=1e-11,
    )
    relative_residual = np.linalg.norm(residual) / np.linalg.norm(transverse_rhs)

    assert projection["relative_residual"] <= 5e-13
    assert relative_residual <= 1e-10
    assert np.linalg.norm(transverse - transverse_true) / np.linalg.norm(transverse_true) <= 1e-7
    high, low = field_parts(longitudinal)
    assert np.linalg.norm(high + low) / np.linalg.norm(transverse) >= 1e6


def test_production_open_boundary_class_is_not_transverse_source_patched():
    assert OpenBoundaryBackground.terminal_model == "distributed_terminal_contact_with_charge_balance"
    assert OpenBoundaryBackground.source_model == "stranded_rectangular_cross_section_gauss3_terminal_contact"
    assert not bool(getattr(OpenBoundaryBackground, "_transverse_source_projection_installed", False))
