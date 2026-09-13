"""Solution-label-free training for the sparse message-passing Maxwell neural solver."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import copy

import numpy as np

from .unified_maxwell import _fgmres, _neural_rollout
from .unified_neural_operator import (
    FEATURE_SCHEMA,
    EdgeMultiscaleConfig,
    build_edge_residual_operator,
    neural_correction,
    operator_feature_statistics,
)


@dataclass(frozen=True)
class MaxwellTrainingConfig:
    epochs: int = 120
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    lr_decay_factor: float = 0.5
    lr_plateau_patience: int = 3
    minimum_learning_rate: float = 5e-4
    patience: int = 20
    validation_interval: int = 2
    min_relative_improvement: float = 1e-3
    benchmark_samples_per_split: int = 4
    seed: int = 17
    dtype: str = "float32"

    def __post_init__(self):
        counts = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.lr_plateau_patience,
            self.patience,
            self.validation_interval,
            self.benchmark_samples_per_split,
        )
        if min(counts) < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0.0 < self.lr_decay_factor < 1.0:
            raise ValueError("lr_decay_factor must lie in (0, 1)")
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("minimum_learning_rate must lie in (0, learning_rate]")
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
    effective_batch_size: int
    final_learning_rate: float
    learning_rate_reductions: int
    solver_benchmark: dict


def _operator_coefficients(background, context, state):
    sigma, eps, mu_inv, _, _, _ = background.cell_properties(context, state, em=True)
    h2 = np.asarray(background.face_cell_hodge @ mu_inv, float).ravel()
    hs = np.asarray(background.edge_cell_hodge @ sigma, float).ravel()
    he = np.asarray(background.edge_cell_hodge @ eps, float).ravel()
    diagonal_term = -background.omega**2 * he + 1j * background.omega * hs
    return h2, np.asarray(diagonal_term, complex), operator_feature_statistics(
        background.em_operator(context, state)
    )


def _rhs_residual_bank(B, mixed_count, rng):
    B = np.asarray(B, complex)
    columns = [B[:, p].copy() for p in range(B.shape[1])]
    if B.shape[1] > 1:
        for _ in range(max(1, int(mixed_count))):
            weights = rng.normal(size=B.shape[1]) + 1j * rng.normal(size=B.shape[1])
            norm = float(np.linalg.norm(weights))
            if norm <= 0 or not np.isfinite(norm):
                continue
            columns.append(np.asarray(B @ (weights / norm), complex))
    bank = np.column_stack(columns)
    if np.any(~np.isfinite(bank)):
        raise FloatingPointError("Maxwell residual seed bank is non-finite")
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
    return (
        torch.sparse.mm(curl_t.transpose(0, 1), h2[:, None] * curl_value)
        + diagonal_term[:, None] * Z
    )


def _torch_dynamic_features(
    torch, R, diagonal, node_features, network_dtype
):
    n = max(1, int(R.shape[0]))
    tiny = torch.finfo(torch.float64).tiny
    jacobi = R / diagonal[:, None]
    rscale = torch.sqrt(torch.sum(torch.abs(R) ** 2, dim=0) / n).clamp_min(tiny)
    zscale = torch.sqrt(torch.sum(torch.abs(jacobi) ** 2, dim=0) / n).clamp_min(tiny)
    rn = R / rscale[None, :]
    zn = jacobi / zscale[None, :]
    operator_batch = node_features.to(dtype=network_dtype).unsqueeze(0).expand(
        R.shape[1], -1, -1
    )
    dynamic = torch.cat(
        [
            rn.real.transpose(0, 1).unsqueeze(-1),
            rn.imag.transpose(0, 1).unsqueeze(-1),
            zn.real.transpose(0, 1).unsqueeze(-1),
            zn.imag.transpose(0, 1).unsqueeze(-1),
            operator_batch,
        ],
        dim=-1,
    )
    return dynamic.to(dtype=network_dtype), zscale


def _unrolled_loss(
    torch,
    model,
    curl_t,
    h2_t,
    diagonal_term_t,
    diagonal_t,
    node_features_t,
    coupling_t,
    R0,
    *,
    network_dtype,
):
    residual = R0
    denominator = torch.sum(torch.abs(R0) ** 2, dim=0).clamp_min(
        torch.finfo(torch.float64).tiny
    )
    weighted = residual.real.new_zeros((), dtype=torch.float64)
    weight_sum = 0.0
    final_ratio = None
    for step in range(int(model.solver_steps)):
        dynamic, scale = _torch_dynamic_features(
            torch, residual, diagonal_t, node_features_t, network_dtype
        )
        y = model(dynamic, coupling_t).to(dtype=torch.float64)
        correction = (
            torch.complex(y[..., 0], y[..., 1])
            .transpose(0, 1)
            .to(torch.complex128)
            * scale[None, :]
        )
        residual = residual - _torch_apply(
            torch, curl_t, h2_t, diagonal_term_t, correction
        )
        final_ratio = torch.mean(
            torch.sum(torch.abs(residual) ** 2, dim=0) / denominator
        )
        weight = float(2**step)
        weighted = weighted + weight * final_ratio
        weight_sum += weight
    return weighted / weight_sum, final_ratio


def _subset_for_benchmark(ids, maximum):
    ids = np.asarray(ids, dtype=int)
    return (
        ids
        if ids.size <= maximum
        else ids[np.linspace(0, ids.size - 1, int(maximum), dtype=int)]
    )


def _distribution(values):
    arr = np.asarray(values, float)
    if arr.size == 0:
        return {"median": None, "p90": None, "max": None}
    return {
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(np.max(arr)),
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
    post_neural, iterations, restarts = [], [], []
    final, neural_times, total_times = [], [], []
    model.eval()
    for idx in chosen:
        context = background.geometry_context(
            dataset.geometries[int(idx)], assemble_thermal=False
        )
        A = background.em_operator(context, dataset.states[int(idx)])
        B = np.asarray(background.rhs_matrix(context), complex)
        graph = operator_feature_statistics(A)
        denominator = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
        start = perf_counter()
        X, residual, relative, _ = _neural_rollout(
            model, A, B, graph, model.solver_steps
        )
        neural_elapsed = perf_counter() - start
        post_neural.extend(map(float, relative))
        for port in range(B.shape[1]):
            if relative[port] <= tolerance:
                iterations.append(0)
                restarts.append(0)
                continue
            residual_norm = max(
                float(np.linalg.norm(residual[:, port])), np.finfo(float).tiny
            )
            correction, count, restart_count = _fgmres(
                A,
                residual[:, port],
                lambda r: neural_correction(
                    model, A, r[:, None], operator_stats=graph
                )[:, 0],
                tolerance=tolerance * denominator[port] / residual_norm,
                max_iterations=max_iterations,
                restart=restart,
            )
            X[:, port] += correction
            iterations.append(int(count))
            restarts.append(int(restart_count))
        total_times.append(perf_counter() - start)
        neural_times.append(neural_elapsed)
        final.extend(map(float, np.linalg.norm(B - A @ X, axis=0) / denominator))
    final_array = np.asarray(final, float)
    return {
        "system_count": int(len(chosen)),
        "rhs_count": int(len(final)),
        "neural_steps": int(model.solver_steps),
        "post_neural_relative_residual": _distribution(post_neural),
        "fgmres_iterations": _distribution(iterations),
        "fgmres_restarts": _distribution(restarts),
        "neural_rollout_time_s": {
            "total": float(np.sum(neural_times)),
            "median_per_system": float(np.median(neural_times)) if neural_times else None,
        },
        "total_solve_time_s": {
            "total": float(np.sum(total_times)),
            "median_per_system": float(np.median(total_times)) if total_times else None,
        },
        "success_rate": (
            float(np.mean(final_array <= tolerance)) if final_array.size else 0.0
        ),
        "maximum_final_relative_residual": (
            float(np.max(final_array)) if final_array.size else None
        ),
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
        raise ImportError(
            "install sdfmpneo[neural] to train the unified model"
        ) from exc

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_decay_factor,
        patience=cfg.lr_plateau_patience,
        threshold=cfg.min_relative_improvement,
        threshold_mode="rel",
        cooldown=0,
        min_lr=cfg.minimum_learning_rate,
    )

    systems = []
    total = len(dataset.geometries)
    for i, (geometry, state) in enumerate(zip(dataset.geometries, dataset.states)):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        h2, diagonal_term, graph = _operator_coefficients(background, context, state)
        B = np.asarray(background.rhs_matrix(context), complex)
        R = _rhs_residual_bank(
            B,
            dataset.residual_steps,
            np.random.default_rng(int(dataset.seed) + 104729 * (i + 1)),
        )
        systems.append((h2, diagonal_term, graph, R))
        if (
            i == 0
            or (i + 1) % max(1, total // 20) == 0
            or i + 1 == total
        ):
            print(
                f"缓存 sparse Maxwell graph……{100 * (i + 1) / total:5.1f}% "
                f"({i + 1}/{total})",
                flush=True,
            )

    curl_t = _torch_sparse_complex(torch, background.curl, actual_device)
    torch_systems = [None] * total

    def torch_system(index):
        index = int(index)
        if torch_systems[index] is None:
            h2, diagonal_term, graph, R = systems[index]
            torch_systems[index] = (
                torch.as_tensor(h2, dtype=torch.complex128, device=actual_device),
                torch.as_tensor(
                    diagonal_term, dtype=torch.complex128, device=actual_device
                ),
                torch.as_tensor(
                    graph.diagonal, dtype=torch.complex128, device=actual_device
                ),
                torch.as_tensor(
                    graph.node_features, dtype=network_dtype, device=actual_device
                ),
                graph.torch_coupling(torch, actual_device, network_dtype),
                torch.as_tensor(R, dtype=torch.complex128, device=actual_device),
            )
        return torch_systems[index]

    def system_losses(index):
        h2_t, d_t, diagonal_t, node_t, coupling_t, R_t = torch_system(index)
        return _unrolled_loss(
            torch,
            model,
            curl_t,
            h2_t,
            d_t,
            diagonal_t,
            node_t,
            coupling_t,
            R_t,
            network_dtype=network_dtype,
        )

    def evaluate(ids):
        model.eval()
        values = []
        with torch.no_grad():
            for idx in ids:
                _, final_loss = system_losses(int(idx))
                values.append(float(final_loss.detach().cpu()))
        return float(np.mean(values)) if values else float("inf")

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    start = 0
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_reference = float("inf")
    stale = 0
    lr_reductions = 0
    checkpoint_identity = {
        "network_config": net_cfg.to_dict(),
        "training_config": cfg.to_dict(),
        "n_edges": int(background.n_edges),
        "sample_count": int(total),
        "residual_seed_count": int(dataset.residual_steps),
        "operator_representation": "sparse_complex_message_graph",
        "feature_schema": FEATURE_SCHEMA,
        "training_objective": "shared_neural_residual_unroll_v1",
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
                    "training configuration or neural graph schema changed"
                )
            model.load_state_dict(saved["network"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start = int(saved["epoch"])
            best = float(saved["best"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            patience_reference = float(saved.get("patience_reference", best))
            stale = int(saved.get("stale", 0))
            lr_reductions = int(saved.get("lr_reductions", 0))
            print(
                f"恢复 sparse neural Maxwell 训练：epoch={start}  "
                f"lr={optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"训练检查点不兼容，重新训练 sparse neural solver：{exc}",
                flush=True,
            )
            checkpoint.unlink(missing_ok=True)

    last_train = float("inf")
    last_validation = float("inf")
    completed = start
    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps
    print(
        f"优化器：physical batch={cfg.batch_size}  "
        f"gradient accumulation={cfg.gradient_accumulation_steps}  "
        f"effective batch={effective_batch_size}  "
        f"lr={cfg.learning_rate:.3e}->{cfg.minimum_learning_rate:.3e}",
        flush=True,
    )

    for epoch in range(start, cfg.epochs):
        if monitor is not None:
            monitor.checkpoint()
        model.train()
        order = np.random.default_rng(cfg.seed + epoch).permutation(train_ids)
        final_losses = []
        update_span = cfg.batch_size * cfg.gradient_accumulation_steps

        for update_start in range(0, len(order), update_span):
            if monitor is not None:
                monitor.checkpoint()
            update_ids = order[update_start : update_start + update_span]
            optimizer.zero_grad(set_to_none=True)
            update_count = len(update_ids)

            for micro_start in range(0, update_count, cfg.batch_size):
                micro_ids = update_ids[micro_start : micro_start + cfg.batch_size]
                objectives = []
                for idx in micro_ids:
                    if monitor is not None:
                        monitor.checkpoint()
                    objective, final_loss = system_losses(int(idx))
                    objectives.append(objective)
                    final_losses.append(float(final_loss.detach().cpu()))
                micro_objective = torch.stack(objectives).mean()
                sample_weight = len(micro_ids) / update_count
                (micro_objective * sample_weight).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        last_train = (
            float(np.mean(final_losses)) if final_losses else float("inf")
        )
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

            previous_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(last_validation)
            current_lr = float(optimizer.param_groups[0]["lr"])
            if current_lr < previous_lr * (1.0 - 1e-12):
                lr_reductions += 1
                stale = 0
                print(
                    f"validation plateau：learning rate {previous_lr:.3e} -> "
                    f"{current_lr:.3e}",
                    flush=True,
                )

        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"训练 sparse neural Maxwell solver……"
            f"{100 * completed / cfg.epochs:5.1f}%  "
            f"epoch={completed}/{cfg.epochs}  train={last_train:.5g}  "
            f"val={last_validation:.5g}  lr={current_lr:.3e}",
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
                        None
                        if not np.isfinite(last_validation)
                        else last_validation
                    ),
                    learning_rate=current_lr,
                    effective_batch_size=effective_batch_size,
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
                    "scheduler": scheduler.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                    "patience_reference": patience_reference,
                    "stale": stale,
                    "lr_reductions": lr_reductions,
                },
                temporary,
            )
            temporary.replace(checkpoint)
        if stale >= cfg.patience:
            print(
                "validation unrolled residual 已在当前最低学习率附近长期无有效相对改进，提前停止。",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    train_loss = evaluate(train_ids)
    validation_loss = evaluate(validation_ids)
    test_loss = evaluate(test_ids)
    bench = dict(benchmark_settings or {})
    tolerance = float(bench.get("residual_tolerance", 1e-7))
    max_iterations = int(bench.get("max_iterations", 200))
    restart = int(bench.get("restart", 40))
    print(
        f"benchmark sparse neural + FGMRES：neural_steps={model.solver_steps}  "
        f"tol={tolerance:.1e}  max_iter={max_iterations}",
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
    return model, MaxwellTrainingReport(
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
        int(systems[0][3].shape[1]),
        int(effective_batch_size),
        float(optimizer.param_groups[0]["lr"]),
        int(lr_reductions),
        solver_benchmark,
    )


__all__ = [
    "MaxwellTrainingConfig",
    "MaxwellTrainingReport",
    "train_maxwell_accelerator",
]
