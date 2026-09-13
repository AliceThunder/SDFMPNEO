"""Solution-label-free training for the polynomial neural Maxwell preconditioner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import copy

import numpy as np

from .unified_maxwell import _fgmres, _minimum_residual_guess
from .unified_neural_operator import (
    FEATURE_SCHEMA,
    EdgeMultiscaleConfig,
    build_edge_residual_operator,
    minimum_residual_jacobi_from_action,
    neural_correction,
    operator_feature_statistics,
    polynomial_state_from_operator_action,
    safe_diagonal_values,
    sparse_row_statistics,
)


@dataclass(frozen=True)
class MaxwellTrainingConfig:
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    patience: int = 24
    validation_interval: int = 2
    min_relative_improvement: float = 5e-4
    benchmark_samples_per_split: int = 6
    seed: int = 17
    dtype: str = "float32"

    def __post_init__(self):
        if min(
            self.epochs,
            self.batch_size,
            self.patience,
            self.validation_interval,
            self.benchmark_samples_per_split,
        ) < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0.0 <= self.min_relative_improvement < 1.0:
            raise ValueError("min_relative_improvement must lie in [0, 1)")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MaxwellTrainingReport:
    epochs_completed: int
    best_epoch: int
    best_validation_residual_loss: float
    train_residual_loss: float
    validation_residual_loss: float
    test_residual_loss: float
    stopped_early: bool
    device: str
    training_config: dict
    network_config: dict
    residual_vectors_per_operator: int
    baseline_train_residual_loss: float
    baseline_validation_residual_loss: float
    baseline_test_residual_loss: float
    solver_benchmark: dict


def _operator_coefficients(background, context, state):
    """Exact matrix-free coefficients plus O(n_edges) sparse row statistics."""
    sigma, eps, mu_inv, _, _, _ = background.cell_properties(context, state, em=True)
    h2 = np.asarray(background.face_cell_hodge @ mu_inv, float).ravel()
    hs = np.asarray(background.edge_cell_hodge @ sigma, float).ravel()
    he = np.asarray(background.edge_cell_hodge @ eps, float).ravel()
    diagonal_term = -background.omega**2 * he + 1j * background.omega * hs
    A = background.em_operator(context, state)
    diagonal = safe_diagonal_values(A.diagonal())
    row_abs_sum, row_nnz = sparse_row_statistics(A)
    return h2, np.asarray(diagonal_term, complex), diagonal, row_abs_sum, row_nnz


def _apply_numpy(curl, h2, diagonal_term, X):
    value = np.asarray(X, complex)
    vector = value.ndim == 1
    if vector:
        value = value[:, None]
    curl_value = curl @ value
    result = (
        curl.T @ (h2[:, None] * curl_value)
        + diagonal_term[:, None] * value
    )
    return np.asarray(result[:, 0] if vector else result, complex)


def _minimum_residual_walk(curl, h2, diagonal_term, diagonal, initial, steps):
    """Generate stable intermediate residuals using label-free 1-D MR updates."""
    r = np.asarray(initial, complex).reshape(-1).copy()
    out = []
    tiny = np.finfo(float).tiny
    for _ in range(int(steps)):
        if np.any(~np.isfinite(r)):
            break
        out.append(r.copy())
        z = r / diagonal
        w = _apply_numpy(curl, h2, diagonal_term, z)
        denom = float(np.vdot(w, w).real)
        if not np.isfinite(denom) or denom <= tiny:
            break
        alpha = np.vdot(w, r) / denom
        trial = r - alpha * w
        if np.any(~np.isfinite(trial)):
            break
        r = trial
    return out


def _baseline_residual_bank(curl, h2, diagonal_term, diagonal, B, steps, rng):
    """Port, mixed-port and intermediate physical residuals without solution labels."""
    B = np.asarray(B, complex)
    columns = []
    for p in range(B.shape[1]):
        columns.extend(
            _minimum_residual_walk(
                curl, h2, diagonal_term, diagonal, B[:, p], steps
            )
        )
    if B.shape[1] > 1:
        n_mixed = max(1, int(steps) // 2)
        for _ in range(n_mixed):
            weights = rng.normal(size=B.shape[1]) + 1j * rng.normal(size=B.shape[1])
            norm = float(np.linalg.norm(weights))
            if norm <= 0 or not np.isfinite(norm):
                continue
            mixed = np.asarray(B @ (weights / norm), complex)
            columns.extend(
                _minimum_residual_walk(
                    curl, h2, diagonal_term, diagonal, mixed, steps
                )
            )
    if not columns:
        raise FloatingPointError(
            "could not generate finite Maxwell residual training vectors"
        )
    bank = np.column_stack(columns)
    if np.any(~np.isfinite(bank)):
        raise FloatingPointError("Maxwell residual training bank is non-finite")
    return bank


def _torch_sparse_complex(torch, matrix, device):
    coo = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack([coo.row, coo.col]), dtype=torch.long, device=device
    )
    values = torch.as_tensor(coo.data, dtype=torch.complex128, device=device)
    return torch.sparse_coo_tensor(
        indices, values, size=matrix.shape, device=device
    ).coalesce()


def _torch_apply(torch, curl_t, h2, diagonal_term, Z):
    curl_value = torch.sparse.mm(curl_t, Z)
    weighted = h2[:, None] * curl_value
    return (
        torch.sparse.mm(curl_t.transpose(0, 1), weighted)
        + diagonal_term[:, None] * Z
    )


def _directional_residual_loss(torch, R_t, W):
    """Best 1-D residual attainable from a proposed preconditioned direction."""
    tiny = torch.finfo(torch.float64).tiny
    image_norm2 = torch.sum(torch.abs(W) ** 2, dim=0).clamp_min(tiny)
    correlation = torch.sum(torch.conj(W) * R_t, dim=0)
    alpha = correlation / image_norm2
    after = R_t - W * alpha[None, :]
    numerator = torch.sum(torch.abs(after) ** 2, dim=0)
    denominator = torch.sum(torch.abs(R_t) ** 2, dim=0).clamp_min(tiny)
    return torch.mean(numerator / denominator)


def _system_loss(
    torch,
    model,
    curl_t,
    curl,
    h2,
    diagonal_term,
    diagonal,
    row_abs_sum,
    row_nnz,
    R,
    *,
    device,
    network_dtype,
):
    apply_operator = lambda Z: _apply_numpy(curl, h2, diagonal_term, Z)
    features, baseline, basis, _ = polynomial_state_from_operator_action(
        diagonal,
        R,
        apply_operator,
        polynomial_order=model.polynomial_order,
        row_abs_sum=row_abs_sum,
        row_nnz=row_nnz,
    )
    x = torch.as_tensor(features, dtype=network_dtype, device=device)
    y = model(x).to(dtype=torch.float64)
    coefficients = torch.complex(y[..., 0], y[..., 1])
    basis_t = torch.as_tensor(basis, dtype=torch.complex128, device=device)
    baseline_t = torch.as_tensor(baseline, dtype=torch.complex128, device=device)
    learned = torch.einsum("nmk,mk->nm", basis_t, coefficients.to(torch.complex128))
    Z = baseline_t + learned

    h2_t = torch.as_tensor(h2, dtype=torch.complex128, device=device)
    d_t = torch.as_tensor(diagonal_term, dtype=torch.complex128, device=device)
    R_t = torch.as_tensor(R, dtype=torch.complex128, device=device)
    W = _torch_apply(torch, curl_t, h2_t, d_t, Z)
    return _directional_residual_loss(torch, R_t, W)


def _baseline_loss(curl, h2, diagonal_term, diagonal, R):
    apply_operator = lambda Z: _apply_numpy(curl, h2, diagonal_term, Z)
    baseline, _ = minimum_residual_jacobi_from_action(
        diagonal, R, apply_operator
    )
    W = _apply_numpy(curl, h2, diagonal_term, baseline)
    tiny = np.finfo(float).tiny
    image_norm2 = np.maximum(np.sum(np.abs(W) ** 2, axis=0), tiny)
    correlation = np.sum(np.conj(W) * R, axis=0)
    alpha = correlation / image_norm2
    after = R - W * alpha[None, :]
    ratio = np.sum(np.abs(after) ** 2, axis=0) / np.maximum(
        np.sum(np.abs(R) ** 2, axis=0), tiny
    )
    return float(np.mean(ratio))


def _subset_for_benchmark(ids, maximum):
    ids = np.asarray(ids, dtype=int)
    if ids.size <= maximum:
        return ids
    positions = np.linspace(0, ids.size - 1, int(maximum), dtype=int)
    return ids[positions]


def _solve_with_preconditioner(
    A,
    B,
    precondition,
    *,
    tolerance,
    max_iterations,
    restart,
):
    B = np.asarray(B, complex)
    direction = np.asarray(precondition(B), complex)
    X = _minimum_residual_guess(A, B, direction)
    denominator = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    residual = B - A @ X
    iterations = []
    restarts = []
    for port in range(B.shape[1]):
        initial = float(np.linalg.norm(residual[:, port]) / denominator[port])
        if initial <= tolerance:
            iterations.append(0)
            restarts.append(0)
            continue
        correction, count, restart_count = _fgmres(
            A,
            residual[:, port],
            lambda r: precondition(r),
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
        )
        X[:, port] += correction
        iterations.append(int(count))
        restarts.append(int(restart_count))
    final = np.linalg.norm(B - A @ X, axis=0) / denominator
    return iterations, restarts, final


def _distribution(values):
    arr = np.asarray(values, float)
    if arr.size == 0:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
    }


def _method_summary(iterations, restarts, system_times, final_residuals, tolerance):
    final = np.asarray(final_residuals, float)
    return {
        "rhs_count": int(len(iterations)),
        "iterations": _distribution(iterations),
        "restarts": _distribution(restarts),
        "wall_time_s": {
            "total": float(np.sum(system_times)),
            "median_per_system": float(np.median(system_times)) if system_times else None,
            "p90_per_system": float(np.quantile(system_times, 0.9)) if system_times else None,
        },
        "success_rate": float(np.mean(final <= tolerance)) if final.size else 0.0,
        "maximum_final_relative_residual": float(np.max(final)) if final.size else None,
    }


def _benchmark_split(
    background,
    dataset,
    ids,
    model,
    *,
    tolerance,
    max_iterations,
    restart,
    sample_limit,
):
    chosen = _subset_for_benchmark(ids, sample_limit)
    raw = {
        "baseline": {"iterations": [], "restarts": [], "times": [], "final": []},
        "neural": {"iterations": [], "restarts": [], "times": [], "final": []},
    }
    model.eval()
    for idx in chosen:
        context = background.geometry_context(
            dataset.geometries[int(idx)], assemble_thermal=False
        )
        state = dataset.states[int(idx)]
        A = background.em_operator(context, state)
        B = np.asarray(background.rhs_matrix(context), complex)
        stats = operator_feature_statistics(A)
        diagonal = stats[0]

        def physical(value):
            R = np.asarray(value, complex)
            vector = R.ndim == 1
            if vector:
                R = R[:, None]
            Z, _ = minimum_residual_jacobi_from_action(
                diagonal, R, lambda V: A @ V
            )
            return Z[:, 0] if vector else Z

        def learned(value):
            R = np.asarray(value, complex)
            vector = R.ndim == 1
            if vector:
                R = R[:, None]
            Z = neural_correction(model, A, R, operator_stats=stats)
            return Z[:, 0] if vector else Z

        for name, precondition in (("baseline", physical), ("neural", learned)):
            start = perf_counter()
            iterations, restarts, final = _solve_with_preconditioner(
                A,
                B,
                precondition,
                tolerance=tolerance,
                max_iterations=max_iterations,
                restart=restart,
            )
            elapsed = perf_counter() - start
            raw[name]["iterations"].extend(iterations)
            raw[name]["restarts"].extend(restarts)
            raw[name]["times"].append(elapsed)
            raw[name]["final"].extend(map(float, final))

    baseline = _method_summary(
        raw["baseline"]["iterations"],
        raw["baseline"]["restarts"],
        raw["baseline"]["times"],
        raw["baseline"]["final"],
        tolerance,
    )
    neural = _method_summary(
        raw["neural"]["iterations"],
        raw["neural"]["restarts"],
        raw["neural"]["times"],
        raw["neural"]["final"],
        tolerance,
    )
    b_iter = baseline["iterations"]["median"]
    n_iter = neural["iterations"]["median"]
    b_time = baseline["wall_time_s"]["total"]
    n_time = neural["wall_time_s"]["total"]
    return {
        "system_count": int(len(chosen)),
        "baseline_mr_jacobi": baseline,
        "neural_polynomial": neural,
        "median_iteration_speedup": (
            float(b_iter / n_iter) if b_iter is not None and n_iter not in {None, 0.0} else None
        ),
        "wall_time_speedup": float(b_time / n_time) if n_time > 0 else None,
    }


def train_maxwell_accelerator(
    background,
    dataset,
    *,
    network_settings=None,
    training_settings=None,
    device="cuda",
    monitor=None,
    checkpoint_path=None,
    benchmark_settings=None,
):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("install sdfmpneo[neural] to train the unified model") from exc

    cfg = MaxwellTrainingConfig(**(training_settings or {}))
    net_cfg = EdgeMultiscaleConfig(**dict(network_settings or {}))
    train_ids = dataset.indices("train")
    validation_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    if min(len(train_ids), len(validation_ids), len(test_ids)) < 1:
        raise ValueError("operator dataset must contain train/validation/test samples")

    actual_device = str(device)
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动退回 CPU。", flush=True)
        actual_device = "cpu"
    network_dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    model = build_edge_residual_operator(background, net_cfg).to(
        device=actual_device, dtype=network_dtype
    )
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fused=actual_device.startswith("cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

    systems = []
    total = len(dataset.geometries)
    for i, (geometry, state) in enumerate(zip(dataset.geometries, dataset.states)):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        h2, diagonal_term, diagonal, row_abs_sum, row_nnz = _operator_coefficients(
            background, context, state
        )
        B = np.asarray(background.rhs_matrix(context), complex)
        systems.append((h2, diagonal_term, diagonal, row_abs_sum, row_nnz, B))
        if i == 0 or (i + 1) % max(1, total // 20) == 0 or i + 1 == total:
            print(
                f"缓存 Maxwell matrix-free 系数……{100 * (i + 1) / total:5.1f}% "
                f"({i + 1}/{total})",
                flush=True,
            )

    curl_t = _torch_sparse_complex(torch, background.curl, actual_device)
    residual_cache = [None] * total

    def residual_bank(index):
        index = int(index)
        h2, diagonal_term, diagonal, row_abs_sum, row_nnz, B = systems[index]
        R = residual_cache[index]
        if R is None:
            rng = np.random.default_rng(int(dataset.seed) + 104729 * (index + 1))
            R = _baseline_residual_bank(
                background.curl,
                h2,
                diagonal_term,
                diagonal,
                B,
                dataset.residual_steps,
                rng,
            )
            residual_cache[index] = R
        return h2, diagonal_term, diagonal, row_abs_sum, row_nnz, R

    def system_loss(index):
        h2, diagonal_term, diagonal, row_abs_sum, row_nnz, R = residual_bank(index)
        return _system_loss(
            torch,
            model,
            curl_t,
            background.curl,
            h2,
            diagonal_term,
            diagonal,
            row_abs_sum,
            row_nnz,
            R,
            device=actual_device,
            network_dtype=network_dtype,
        )

    def evaluate(ids):
        model.eval()
        values = []
        with torch.no_grad():
            for idx in ids:
                values.append(float(system_loss(int(idx)).detach().cpu()))
        return float(np.mean(values)) if values else float("inf")

    def evaluate_baseline(ids):
        values = []
        for idx in ids:
            h2, diagonal_term, diagonal, _, _, R = residual_bank(idx)
            values.append(
                _baseline_loss(
                    background.curl, h2, diagonal_term, diagonal, R
                )
            )
        return float(np.mean(values)) if values else float("inf")

    baseline_train = evaluate_baseline(train_ids)
    baseline_validation = evaluate_baseline(validation_ids)
    baseline_test = evaluate_baseline(test_ids)
    print(
        "MR-Jacobi directional baseline："
        f"train={baseline_train:.5g}  val={baseline_validation:.5g}  "
        f"test={baseline_test:.5g}",
        flush=True,
    )

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    start = 0
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_reference = float("inf")
    stale = 0
    checkpoint_identity = {
        "network_config": net_cfg.to_dict(),
        "training_config": cfg.to_dict(),
        "n_edges": int(background.n_edges),
        "sample_count": int(total),
        "residual_steps": int(dataset.residual_steps),
        "operator_representation": "matrix_free_curl_hodge",
        "feature_schema": FEATURE_SCHEMA,
        "training_objective": "fgmres_polynomial_directional_residual_v2",
    }
    if checkpoint is not None and checkpoint.is_file():
        try:
            try:
                saved = torch.load(
                    checkpoint, map_location=actual_device, weights_only=False
                )
            except TypeError:
                saved = torch.load(checkpoint, map_location=actual_device)
            if saved.get("identity") != checkpoint_identity:
                raise ValueError(
                    "training configuration, feature schema, or polynomial model changed"
                )
            model.load_state_dict(saved["network"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["epoch"])
            best = float(saved["best"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            patience_reference = float(saved.get("patience_reference", best))
            stale = int(saved.get("stale", 0))
            print(f"恢复 polynomial Maxwell 训练：epoch={start}", flush=True)
        except Exception as exc:
            print(f"训练检查点不兼容，重新训练 polynomial smoother：{exc}", flush=True)
            checkpoint.unlink(missing_ok=True)

    last_train = float("inf")
    last_validation = float("inf")
    completed = start
    for epoch in range(start, cfg.epochs):
        if monitor is not None:
            monitor.checkpoint()
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(train_ids)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        pending_losses = []
        for position, idx in enumerate(order):
            if monitor is not None:
                monitor.checkpoint()
            loss = system_loss(int(idx))
            pending_losses.append(loss)
            losses.append(float(loss.detach().cpu()))
            flush = len(pending_losses) >= cfg.batch_size or position + 1 == len(order)
            if flush:
                torch.stack(pending_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_losses.clear()

        last_train = float(np.mean(losses)) if losses else float("inf")
        completed = epoch + 1
        validate = (
            completed == 1
            or completed % cfg.validation_interval == 0
            or completed == cfg.epochs
        )
        if validate:
            last_validation = evaluate(validation_ids)
            if last_validation < best:
                best = last_validation
                best_epoch = completed
                best_state = copy.deepcopy(model.state_dict())
            if not np.isfinite(patience_reference):
                patience_reference = last_validation
                stale = 0
            else:
                required = cfg.min_relative_improvement * max(
                    abs(patience_reference), np.finfo(float).tiny
                )
                if last_validation <= patience_reference - required:
                    patience_reference = last_validation
                    stale = 0
                else:
                    stale += cfg.validation_interval

        print(
            f"训练 Maxwell neural polynomial smoother……"
            f"{100 * completed / cfg.epochs:5.1f}%  epoch={completed}/{cfg.epochs}  "
            f"train={last_train:.5g}  val={last_validation:.5g}",
            flush=True,
        )
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="neural_training",
                    epoch=completed,
                    epoch_total=cfg.epochs,
                    train_loss=last_train,
                    validation_loss=(
                        None if not np.isfinite(last_validation) else last_validation
                    ),
                    baseline_validation_loss=baseline_validation,
                )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "identity": checkpoint_identity,
                    "epoch": completed,
                    "network": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience_reference": patience_reference,
                    "stale": stale,
                },
                temporary,
            )
            temporary.replace(checkpoint)
        if stale >= cfg.patience:
            print(
                "validation directional loss 已长期无有效相对改进，提前停止。",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    test_loss = evaluate(test_ids)
    train_loss = evaluate(train_ids)
    validation_loss = evaluate(validation_ids)

    bench = dict(benchmark_settings or {})
    tolerance = float(bench.get("residual_tolerance", 1e-7))
    max_iterations = int(bench.get("max_iterations", 200))
    restart = int(bench.get("restart", 40))
    print(
        f"benchmark FGMRES：tol={tolerance:.1e}  max_iter={max_iterations}  restart={restart}",
        flush=True,
    )
    solver_benchmark = {
        "validation": _benchmark_split(
            background,
            dataset,
            validation_ids,
            model,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
            sample_limit=cfg.benchmark_samples_per_split,
        ),
        "test": _benchmark_split(
            background,
            dataset,
            test_ids,
            model,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
            sample_limit=cfg.benchmark_samples_per_split,
        ),
    }
    for split in ("validation", "test"):
        result = solver_benchmark[split]
        base = result["baseline_mr_jacobi"]
        neural = result["neural_polynomial"]
        print(
            f"{split} FGMRES benchmark："
            f"iterations median {base['iterations']['median']} -> {neural['iterations']['median']}  "
            f"p90 {base['iterations']['p90']} -> {neural['iterations']['p90']}  "
            f"wall speedup={result['wall_time_speedup']:.3g}x",
            flush=True,
        )

    _, _, _, _, _, first_R = residual_bank(0)
    report = MaxwellTrainingReport(
        completed,
        best_epoch,
        float(best),
        float(train_loss),
        float(validation_loss),
        float(test_loss),
        completed < cfg.epochs,
        actual_device,
        cfg.to_dict(),
        net_cfg.to_dict(),
        int(first_R.shape[1]),
        float(baseline_train),
        float(baseline_validation),
        float(baseline_test),
        solver_benchmark,
    )
    return model, report


__all__ = [
    "MaxwellTrainingConfig",
    "MaxwellTrainingReport",
    "train_maxwell_accelerator",
]
