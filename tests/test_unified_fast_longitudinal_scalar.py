import types

import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_fast_scalar_solve as fast_scalar
import sdfmpneo.unified_global_longitudinal_reference as longitudinal
import sdfmpneo.unified_terminal_dissipative_defect as terminal_defect


def _laplacian_3d(nx, ny, nz):
    def one(n):
        return sp.diags(
            (-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)),
            (-1, 0, 1),
            format="csr",
        )

    Ix = sp.eye(nx, format="csr")
    Iy = sp.eye(ny, format="csr")
    Iz = sp.eye(nz, format="csr")
    return (
        sp.kron(sp.kron(one(nx), Iy), Iz, format="csr")
        + sp.kron(sp.kron(Ix, one(ny)), Iz, format="csr")
        + sp.kron(sp.kron(Ix, Iy), one(nz), format="csr")
    ).tocsr().astype(complex)


def test_two_level_scalar_reaches_true_residual_without_direct_fallback(monkeypatch):
    parent_axis = np.array([0.0, 1.0, 2.0, 3.0])
    fine_axis = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    parent = types.SimpleNamespace(x=parent_axis, y=parent_axis, z=parent_axis)
    patch = types.SimpleNamespace(x=fine_axis, y=fine_axis, z=fine_axis)

    n = (len(fine_axis) - 2) ** 3
    A = _laplacian_3d(len(fine_axis) - 2, len(fine_axis) - 2, len(fine_axis) - 2)
    rhs = np.linspace(0.25, 1.25, n).astype(complex)
    x0 = np.zeros(n, complex)

    monkeypatch.setattr(fast_scalar, "_DIRECT_THRESHOLD", 1)

    def forbidden_direct(*args, **kwargs):
        raise AssertionError("two-level scalar unexpectedly fell back to direct solve")

    monkeypatch.setattr(fast_scalar, "_direct_solve", forbidden_direct)
    field, residual, label = fast_scalar.solve_refined(
        A,
        rhs,
        parent=parent,
        patch=patch,
        x0=x0,
    )
    true_residual = np.linalg.norm(rhs - A @ field) / np.linalg.norm(rhs)
    assert label == "two-level-lgmres"
    assert residual <= 1e-9
    assert true_residual <= 1e-9


def test_production_installs_fast_scalar_paths_for_v34_truth():
    import sdfmpneo
    from sdfmpneo import unified_model
    from sdfmpneo import unified_runtime

    assert unified_model.FORMAT_VERSION == 36
    assert unified_runtime._CACHE_FORMAT == 39
    assert longitudinal._fast_reactive_scalar_installed is True
    assert longitudinal._longitudinal_state_cache_installed is True
    assert terminal_defect._fast_terminal_dissipative_scalar_installed is True
    import sdfmpneo.unified_longitudinal_patch_consistency as consistency\n    assert consistency._lightweight_scalar_patch_installed is True\n    assert terminal_defect._terminal_component_lock_installed is True
    assert "production_parent_piecewise_constant_complex_mass_v1" in longitudinal._MODEL
    assert not bool(getattr(longitudinal, "_shared_terminal_dissipative_installed", False))
