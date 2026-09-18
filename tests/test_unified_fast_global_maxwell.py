import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_certified_local_solve as local_solver
import sdfmpneo.unified_fast_global_maxwell as fast_global
from sdfmpneo.unified_fast_global_maxwell import solve_multi_rhs


class _Background:
    background_config = {
        "linear_solver": {
            "relative_residual_tolerance": 1e-10,
            "direct_max_dofs": 1,
            "iterative_maxiter": 16,
            "iterative_inner_m": 24,
            "ilu_drop_tolerance": 1e-3,
            "ilu_fill_factor": 4.0,
            "ilu_strong_drop_tolerance": 1e-4,
            "ilu_strong_fill_factor": 8.0,
            "ilu_shift_factor": 3e-2,
            "ilu_strong_shift_factor": 1e-1,
        }
    }


def test_global_multi_rhs_iterative_solver_shares_policy_and_certifies_true_residual():
    n = 320
    A = sp.diags(
        (
            -np.ones(n - 1),
            (4.0 + 0.15j) * np.ones(n),
            -np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    x0 = np.linspace(0.1, 1.0, n) + 1j * np.linspace(-0.2, 0.3, n)
    x1 = np.sin(np.linspace(0.0, np.pi, n)) + 0.2j * np.cos(np.linspace(0.0, np.pi, n))
    truth = np.column_stack((x0, x1))
    B = A @ truth

    X, residual, history = solve_multi_rhs(_Background(), A, B, local_solver)

    assert X.shape == truth.shape
    assert history
    assert residual <= 1e-10
    scale = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    actual = np.linalg.norm(B - A @ X, axis=0) / scale
    assert float(np.max(actual)) <= 1e-10



def test_tensor_port_truth_tags_operator_for_compatible_solver(monkeypatch):
    class PhysicsGate:
        pass

    class TensorSurrogate:
        pass

    class Background:
        def em_operator(self, context, temperature):
            assert temperature is None
            return sp.eye(4, format="csr", dtype=complex)

        def rhs_matrix(self, context):
            return np.ones((4, 2), complex)

    seen = {}

    def fake_solve_multi_rhs(background, A, B, local_solver_module):
        seen["background"] = background
        seen["A"] = A
        seen["B"] = np.asarray(B)
        seen["local_solver"] = local_solver_module
        return np.ones_like(B), 1e-12, ()

    monkeypatch.setattr(fast_global, "solve_multi_rhs", fake_solve_multi_rhs)
    fast_global.install(PhysicsGate, TensorSurrogate, local_solver)

    background = Background()
    context = object()
    X, residual = TensorSurrogate._solve_port_fields(background, context)

    assert X.shape == (4, 2)
    assert residual == 1e-12
    A = seen["A"]
    assert A._sdfmpneo_background is background
    assert A._sdfmpneo_context is context
    assert A._sdfmpneo_mqs is False



def test_global_solver_honors_run_iterative_defect_aliases():
    class Background:
        background_config = {
            "linear_solver": {
                "iterative_defect_steps": 3,
                "iterative_defect_maxiter": 16,
                "iterative_defect_inner_m": 21,
                "iterative_defect_start_residual": 5e-6,
            }
        }

    cfg = fast_global._cfg(Background())
    assert cfg["defect_steps"] == 3
    assert cfg["defect_maxiter"] == 16
    assert cfg["defect_inner_m"] == 21
    assert cfg["defect_start_residual"] == 5e-6
