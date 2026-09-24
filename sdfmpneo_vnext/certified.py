from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import gmres

from .certification import (
    PortCertificate,
    certify_port_result,
)
from .em import (
    DenseMQSTeacher,
    MQSConfig,
    MQSResult,
)
from .mixed import DenseMixedConductorTeacher, MixedResult
from .matrix_free import MatrixFreeMixedOperator
from .scene import Scene


@dataclass(frozen=True)
class CertifiedPortResult:
    status: str
    impedance: np.ndarray
    result: object
    port_certificate: PortCertificate
    initial_residual: float
    final_residual: float
    correction_iterations: tuple[int, ...]
    algebraic_certified: bool
    discretization_certified: bool
    used_reference_fallback: bool
    operator_backend: str = "dense"
    relative_observable_correction: float = float("inf")
    fast_domain_valid: bool = False

    @property
    def certified(self) -> bool:
        return self.status in (
            "CERTIFIED",
            "CORRECTED_OUT_OF_FAST_DOMAIN",
        )


def _mode_metadata(
    teacher: DenseMQSTeacher,
):
    segment_coils = np.asarray(
        [
            segment.coil
            for segment
            in teacher._segments
        ],
        dtype=int,
    )
    mode_segments = np.empty(
        teacher._n_modes,
        dtype=int,
    )
    for index, segment in enumerate(
        teacher._segments
    ):
        mode_segments[
            segment.mode_slice
        ] = index
    return (
        segment_coils,
        mode_segments,
    )


def _make_result(
    teacher: DenseMQSTeacher,
    R,
    L,
    C,
    B,
    coefficients,
    multipliers,
) -> MQSResult:
    impedance = B.T @ multipliers
    (
        segment_coils,
        mode_segments,
    ) = _mode_metadata(
        teacher
    )
    return MQSResult(
        impedance,
        coefficients,
        multipliers,
        R,
        L,
        C,
        B,
        segment_coils,
        mode_segments,
    )


def _scaled_row_weights(
    matrix,
):
    row_norm = np.max(
        np.abs(matrix),
        axis=1,
    )
    return 1.0 / np.maximum(
        row_norm,
        1e-30,
    )


def _scaled_residual(
    matrix,
    rhs,
    solution,
    row_weights,
) -> float:
    residual = rhs - matrix @ solution
    numerator = np.linalg.norm(
        row_weights * residual
    )
    denominator = max(
        np.linalg.norm(
            row_weights * rhs
        ),
        1.0,
    )
    return float(
        numerator / denominator
    )


def _uniform_fast_lift(
    teacher: DenseMQSTeacher,
    C,
    B,
    fast_impedance,
):
    m = C.shape[1]
    ns, n_ports = B.shape
    coefficients = np.zeros(
        (m, n_ports),
        dtype=complex,
    )
    multipliers = np.zeros(
        (ns, n_ports),
        dtype=complex,
    )

    coil_segments = {}
    for segment_index, segment in enumerate(
        teacher._segments
    ):
        coil_segments.setdefault(
            int(segment.coil),
            [],
        ).append(
            segment_index
        )
        first = segment.mode_slice.start
        moment = C[
            segment_index,
            first,
        ]
        if abs(moment) <= 1e-14:
            raise RuntimeError(
                "constant conductor mode has zero terminal-current moment"
            )
        for port in range(
            n_ports
        ):
            coefficients[
                first,
                port,
            ] = (
                B[
                    segment_index,
                    port,
                ]
                / moment
            )

    for coil, indices in coil_segments.items():
        lengths = np.asarray(
            [
                teacher._segments[
                    index
                ].length
                for index
                in indices
            ],
            dtype=float,
        )
        weights = lengths / np.sum(
            lengths
        )
        index_array = np.asarray(
            indices,
            dtype=int,
        )
        for port in range(
            n_ports
        ):
            multipliers[
                index_array,
                port,
            ] = (
                weights
                * fast_impedance[
                    coil,
                    port,
                ]
            )

    return (
        coefficients,
        multipliers,
    )


def certify_mqs_ports(
    scene: Scene,
    frequency_hz: float,
    artifact,
    *,
    config: MQSConfig | None = None,
    convergence_report=None,
    algebraic_tolerance: float = 1e-9,
    correction_rtol: float = 1e-10,
    correction_restart: int = 40,
    correction_maxiter: int = 80,
    allow_reference_fallback: bool = True,
    fast_domain_correction_limit: float = 0.20,
) -> CertifiedPortResult:
    """Lift a FAST model into the physical MQS KKT space and certify it.

    This certifies the assembled discrete operator. Full CERTIFIED status
    additionally requires a converged basis/quadrature report supplied by
    the caller.
    """
    if (
        algebraic_tolerance <= 0.0
        or correction_rtol <= 0.0
        or correction_restart < 1
        or correction_maxiter < 1
        or fast_domain_correction_limit < 0.0
    ):
        raise ValueError(
            "invalid certification tolerances"
        )
    if not hasattr(
        artifact,
        "predict_structured",
    ):
        raise TypeError(
            "artifact must expose predict_structured"
        )

    teacher = DenseMQSTeacher(
        scene,
        frequency_hz,
        config or MQSConfig(),
    )
    R, L, C, B = teacher.assemble()
    A = (
        R.astype(complex)
        + 1j
        * teacher.omega
        * L
    )
    m = A.shape[0]
    ns = C.shape[0]
    K = np.block(
        [
            [
                A,
                -C.T.astype(complex),
            ],
            [
                C.astype(complex),
                np.zeros(
                    (ns, ns),
                    dtype=complex,
                ),
            ],
        ]
    )
    rhs_matrix = np.vstack(
        (
            np.zeros(
                (
                    m,
                    B.shape[1],
                ),
                dtype=complex,
            ),
            B.astype(complex),
        )
    )

    fast_prediction = artifact.predict_structured(
        scene,
        frequency_hz,
    )
    fast_impedance = np.asarray(
        fast_prediction.impedance,
        dtype=complex,
    )
    if fast_impedance.shape != (
        B.shape[1],
        B.shape[1],
    ):
        raise ValueError(
            "FAST artifact returned wrong impedance shape"
        )

    (
        coefficients,
        multipliers,
    ) = _uniform_fast_lift(
        teacher,
        C,
        B,
        fast_impedance,
    )
    lift = np.vstack(
        (
            coefficients,
            multipliers,
        )
    )
    row_weights = _scaled_row_weights(
        K
    )

    initial_residuals = [
        _scaled_residual(
            K,
            rhs_matrix[:, port],
            lift[:, port],
            row_weights,
        )
        for port in range(
            B.shape[1]
        )
    ]
    initial_residual = float(
        max(initial_residuals)
    )

    correction_iterations = []
    corrected = lift.copy()
    correction_failed = False

    if (
        initial_residual
        > algebraic_tolerance
    ):
        for port in range(
            B.shape[1]
        ):
            counter = {
                "iterations": 0
            }

            def callback(_):
                counter[
                    "iterations"
                ] += 1

            solution, info = gmres(
                K,
                rhs_matrix[:, port],
                x0=lift[:, port],
                rtol=correction_rtol,
                atol=0.0,
                restart=min(
                    correction_restart,
                    K.shape[0],
                ),
                maxiter=(
                    correction_maxiter
                ),
                callback=callback,
                callback_type="pr_norm",
            )
            correction_iterations.append(
                int(
                    counter[
                        "iterations"
                    ]
                )
            )
            corrected[:, port] = (
                solution
            )
            if info != 0:
                correction_failed = True
    else:
        correction_iterations = [
            0
            for _ in range(
                B.shape[1]
            )
        ]

    final_residuals = [
        _scaled_residual(
            K,
            rhs_matrix[:, port],
            corrected[:, port],
            row_weights,
        )
        for port in range(
            B.shape[1]
        )
    ]
    final_residual = float(
        max(final_residuals)
    )

    result = _make_result(
        teacher,
        R,
        L,
        C,
        B,
        corrected[:m],
        corrected[m:],
    )
    port_certificate = certify_port_result(
        result,
        residual_tolerance=(
            algebraic_tolerance
        ),
        reciprocity_tolerance=1e-8,
        power_tolerance=1e-7,
        passivity_tolerance=1e-10,
    )
    algebraic_certified = bool(
        not correction_failed
        and final_residual
        <= algebraic_tolerance
        and port_certificate.certified
    )

    discretization_certified = bool(
        convergence_report
        is not None
        and getattr(
            convergence_report,
            "converged",
            False,
        )
    )

    relative_observable_correction = float(
        np.linalg.norm(
            result.impedance
            - fast_impedance
        )
        / max(
            np.linalg.norm(
                result.impedance
            ),
            1e-30,
        )
    )
    fast_domain_valid = bool(
        relative_observable_correction
        <= fast_domain_correction_limit
    )

    used_reference_fallback = False
    if not algebraic_certified:
        if not allow_reference_fallback:
            return CertifiedPortResult(
                "UNCERTIFIED",
                result.impedance,
                result,
                port_certificate,
                initial_residual,
                final_residual,
                tuple(
                    correction_iterations
                ),
                False,
                discretization_certified,
                False,
                "dense",
                relative_observable_correction,
                fast_domain_valid,
            )

        result = teacher.solve()
        port_certificate = certify_port_result(
            result,
            residual_tolerance=(
                algebraic_tolerance
            ),
            reciprocity_tolerance=1e-8,
            power_tolerance=1e-7,
            passivity_tolerance=1e-10,
        )
        algebraic_certified = bool(
            port_certificate.certified
        )
        final_residual = 0.0
        used_reference_fallback = True
        relative_observable_correction = float(
            np.linalg.norm(
                result.impedance
                - fast_impedance
            )
            / max(
                np.linalg.norm(
                    result.impedance
                ),
                1e-30,
            )
        )
        fast_domain_valid = bool(
            relative_observable_correction
            <= fast_domain_correction_limit
        )
        status = (
            "REFERENCE_FALLBACK"
        )
    elif discretization_certified:
        status = (
            "CERTIFIED"
            if fast_domain_valid
            else "CORRECTED_OUT_OF_FAST_DOMAIN"
        )
    else:
        status = (
            "DISCRETE_CERTIFIED"
        )

    return CertifiedPortResult(
        status,
        result.impedance,
        result,
        port_certificate,
        initial_residual,
        final_residual,
        tuple(
            correction_iterations
        ),
        algebraic_certified,
        discretization_certified,
        used_reference_fallback,
        "dense",
        relative_observable_correction,
        fast_domain_valid,
    )



def _current_constraint_port_map(
    teacher: DenseMQSTeacher,
):
    n_segments = len(
        teacher._segments
    )
    n_ports = len(
        teacher.scene.coils
    )
    constraint = np.zeros(
        (
            n_segments,
            teacher._n_modes,
        ),
        dtype=float,
    )
    port_map = np.zeros(
        (
            n_segments,
            n_ports,
        ),
        dtype=float,
    )
    for segment_index, segment in enumerate(
        teacher._segments
    ):
        constraint[
            segment_index,
            segment.mode_slice,
        ] = (
            segment.basis.moments
        )
        port_map[
            segment_index,
            segment.coil,
        ] = 1.0
    return (
        constraint,
        port_map,
    )


def _mixed_fast_lift(
    teacher: DenseMixedConductorTeacher,
    R,
    D,
    Phi,
    B,
    Q,
    fast_impedance,
    *,
    current_constraint=None,
    current_port_map=None,
):
    """Build a compatible current/potential/charge initial state."""
    if (
        current_constraint is None
        or current_port_map is None
    ):
        (
            current_constraint,
            current_port_map,
        ) = _current_constraint_port_map(
            teacher._mqs
        )
    current, _ = _uniform_fast_lift(
        teacher._mqs,
        current_constraint,
        current_port_map,
        fast_impedance,
    )

    Dr = Q.T @ D
    Br = Q.T @ B
    Phir = (
        Q.T
        @ Phi
        @ Q
    )
    nr = Dr.shape[0]
    n_ports = B.shape[1]

    potential_r = np.zeros(
        (
            nr,
            n_ports,
        ),
        dtype=complex,
    )
    charge_r = np.zeros_like(
        potential_r
    )
    port_observation = Br.T
    for port in range(
        n_ports
    ):
        potential_r[
            :,
            port,
        ] = np.linalg.lstsq(
            port_observation,
            fast_impedance[
                :,
                port,
            ],
            rcond=None,
        )[
            0
        ]
        charge_r[
            :,
            port,
        ] = np.linalg.lstsq(
            Phir,
            potential_r[
                :,
                port,
            ],
            rcond=None,
        )[
            0
        ]
    return (
        current,
        potential_r,
        charge_r,
    )

def _make_mixed_result(
    teacher: DenseMixedConductorTeacher,
    R,
    L,
    D,
    Phi,
    B,
    Q,
    corrected,
    residual,
) -> MixedResult:
    m = R.shape[0]
    nr = Q.shape[1]
    current = corrected[
        :m
    ]
    potential_r = corrected[
        m : m + nr
    ]
    charge_r = corrected[
        m + nr :
    ]
    potential = (
        Q
        @ potential_r
    )
    charge = (
        Q
        @ charge_r
    )
    impedance = (
        B.T
        @ potential
    )
    segment_coils = np.asarray(
        [
            segment.coil
            for segment
            in teacher._mqs._segments
        ],
        dtype=int,
    )
    mode_segments = np.empty(
        teacher._mqs._n_modes,
        dtype=int,
    )
    for index, segment in enumerate(
        teacher._mqs._segments
    ):
        mode_segments[
            segment.mode_slice
        ] = index
    return MixedResult(
        impedance,
        current,
        potential,
        charge,
        R,
        L,
        D,
        Phi,
        B,
        float(
            residual
        ),
        segment_coils,
        mode_segments,
    )


def certify_mixed_ports(
    scene: Scene,
    frequency_hz: float,
    artifact,
    *,
    config: MQSConfig | None = None,
    convergence_report=None,
    algebraic_tolerance: float = 1e-9,
    correction_rtol: float = 1e-10,
    correction_restart: int = 40,
    correction_maxiter: int = 100,
    allow_reference_fallback: bool = True,
    operator_backend: str = "dense",
    matrix_free_chunk_size: int = 512,
    fast_domain_correction_limit: float = 0.20,
) -> CertifiedPortResult:
    """CERTIFIED correction in the canonical current-potential-charge KKT.

    The dense and matrix-free backends represent the same mixed discretization.
    Matrix-free mode avoids formation of the global magnetic inductance matrix;
    the electrostatic potential block remains dense in this MVP.
    """
    if (
        algebraic_tolerance <= 0.0
        or correction_rtol <= 0.0
        or correction_restart < 1
        or correction_maxiter < 1
        or matrix_free_chunk_size < 1
        or fast_domain_correction_limit < 0.0
    ):
        raise ValueError(
            "invalid certification tolerances"
        )
    if operator_backend not in (
        "dense",
        "matrix_free",
    ):
        raise ValueError(
            "operator_backend must be 'dense' or 'matrix_free'"
        )
    if not hasattr(
        artifact,
        "predict_structured",
    ):
        raise TypeError(
            "artifact must expose predict_structured"
        )

    resolved_config = (
        config
        or MQSConfig()
    )
    teacher = DenseMixedConductorTeacher(
        scene,
        frequency_hz,
        resolved_config,
    )

    matrix_free = None
    if operator_backend == "dense":
        (
            R,
            L,
            D,
            Phi,
            B,
            Q,
        ) = teacher.assemble()
        A = (
            R.astype(
                complex
            )
            + 1j
            * teacher.omega
            * L
        )
        Dr = Q.T @ D
        Br = Q.T @ B
        Phir = (
            Q.T
            @ Phi
            @ Q
        )
        m = A.shape[0]
        nr = Dr.shape[0]
        K = np.block(
            [
                [
                    A,
                    -Dr.T.astype(
                        complex
                    ),
                    np.zeros(
                        (
                            m,
                            nr,
                        ),
                        dtype=complex,
                    ),
                ],
                [
                    Dr.astype(
                        complex
                    ),
                    np.zeros(
                        (
                            nr,
                            nr,
                        ),
                        dtype=complex,
                    ),
                    1j
                    * teacher.omega
                    * np.eye(
                        nr,
                        dtype=complex,
                    ),
                ],
                [
                    np.zeros(
                        (
                            nr,
                            m,
                        ),
                        dtype=complex,
                    ),
                    np.eye(
                        nr,
                        dtype=complex,
                    ),
                    -Phir.astype(
                        complex
                    ),
                ],
            ]
        )
        rhs_matrix = np.vstack(
            (
                np.zeros(
                    (
                        m,
                        B.shape[1],
                    ),
                    dtype=complex,
                ),
                Br.astype(
                    complex
                ),
                np.zeros(
                    (
                        nr,
                        B.shape[1],
                    ),
                    dtype=complex,
                ),
            )
        )
        row_weights = (
            _scaled_row_weights(
                K
            )
        )
        preconditioner = None

        def residual_measure(
            rhs,
            solution,
        ):
            return _scaled_residual(
                K,
                rhs,
                solution,
                row_weights,
            )

        (
            current_constraint,
            current_port_map,
        ) = _current_constraint_port_map(
            teacher._mqs
        )
    else:
        matrix_free = (
            MatrixFreeMixedOperator(
                scene,
                frequency_hz,
                resolved_config,
                chunk_size=(
                    matrix_free_chunk_size
                ),
            )
        )
        K = (
            matrix_free.linear_operator()
        )
        R = (
            matrix_free.resistance_diagonal
        )
        L = None
        D = (
            matrix_free.divergence_matrix
        )
        Phi = (
            matrix_free.potential_matrix
        )
        B = (
            matrix_free.port_injection
        )
        Q = (
            matrix_free.gauge_basis
        )
        Dr = (
            matrix_free.reduced_divergence
        )
        Br = (
            matrix_free.reduced_port_injection
        )
        Phir = (
            matrix_free.reduced_potential
        )
        m = (
            matrix_free.metadata.n_current_modes
        )
        nr = (
            matrix_free.metadata.n_reduced_potential
        )
        rhs_matrix = np.column_stack(
            [
                matrix_free.port_rhs(
                    port
                )
                for port in range(
                    matrix_free.metadata.n_ports
                )
            ]
        )

        def residual_measure(
            rhs,
            solution,
        ):
            residual = (
                rhs
                - K @ solution
            )
            return float(
                np.linalg.norm(
                    residual
                )
                / max(
                    np.linalg.norm(
                        rhs
                    ),
                    1.0,
                )
            )

        current_constraint = (
            matrix_free.mqs.constraint_matrix
        )
        current_port_map = (
            matrix_free.mqs.port_map
        )
        preconditioner = (
            matrix_free.resistive_preconditioner()
        )

    fast_prediction = (
        artifact.predict_structured(
            scene,
            frequency_hz,
        )
    )
    fast_impedance = np.asarray(
        fast_prediction.impedance,
        dtype=complex,
    )
    if fast_impedance.shape != (
        B.shape[1],
        B.shape[1],
    ):
        raise ValueError(
            "FAST artifact returned wrong impedance shape"
        )

    (
        current,
        potential_r,
        charge_r,
    ) = _mixed_fast_lift(
        teacher,
        R,
        D,
        Phi,
        B,
        Q,
        fast_impedance,
        current_constraint=(
            current_constraint
        ),
        current_port_map=(
            current_port_map
        ),
    )
    lift = np.vstack(
        (
            current,
            potential_r,
            charge_r,
        )
    )
    initial_residual = float(
        max(
            residual_measure(
                rhs_matrix[
                    :,
                    port,
                ],
                lift[
                    :,
                    port,
                ],
            )
            for port in range(
                B.shape[1]
            )
        )
    )

    corrected = (
        lift.copy()
    )
    correction_iterations = []
    correction_failed = False
    if (
        initial_residual
        > algebraic_tolerance
    ):
        for port in range(
            B.shape[1]
        ):
            counter = {
                "iterations": 0
            }

            def callback(_):
                counter[
                    "iterations"
                ] += 1

            solution, info = gmres(
                K,
                rhs_matrix[
                    :,
                    port,
                ],
                x0=lift[
                    :,
                    port,
                ],
                M=preconditioner,
                rtol=(
                    correction_rtol
                ),
                atol=0.0,
                restart=min(
                    correction_restart,
                    K.shape[0],
                ),
                maxiter=(
                    correction_maxiter
                ),
                callback=callback,
                callback_type="pr_norm",
            )
            corrected[
                :,
                port,
            ] = solution
            correction_iterations.append(
                int(
                    counter[
                        "iterations"
                    ]
                )
            )
            if info != 0:
                correction_failed = True
    else:
        correction_iterations = [
            0
            for _ in range(
                B.shape[1]
            )
        ]

    final_residual = float(
        max(
            residual_measure(
                rhs_matrix[
                    :,
                    port,
                ],
                corrected[
                    :,
                    port,
                ],
            )
            for port in range(
                B.shape[1]
            )
        )
    )
    result = _make_mixed_result(
        teacher,
        R,
        L,
        D,
        Phi,
        B,
        Q,
        corrected,
        final_residual,
    )
    port_certificate = (
        certify_port_result(
            result,
            residual_tolerance=(
                algebraic_tolerance
            ),
            reciprocity_tolerance=1e-8,
            power_tolerance=1e-7,
            passivity_tolerance=1e-10,
        )
    )
    algebraic_certified = bool(
        not correction_failed
        and final_residual
        <= algebraic_tolerance
        and port_certificate.certified
    )
    discretization_certified = bool(
        convergence_report
        is not None
        and getattr(
            convergence_report,
            "converged",
            False,
        )
    )

    relative_observable_correction = float(
        np.linalg.norm(
            result.impedance
            - fast_impedance
        )
        / max(
            np.linalg.norm(
                result.impedance
            ),
            1e-30,
        )
    )
    fast_domain_valid = bool(
        relative_observable_correction
        <= fast_domain_correction_limit
    )

    used_reference_fallback = False
    if not algebraic_certified:
        if not allow_reference_fallback:
            status = "UNCERTIFIED"
        else:
            result = teacher.solve()
            port_certificate = (
                certify_port_result(
                    result,
                    residual_tolerance=(
                        algebraic_tolerance
                    ),
                    reciprocity_tolerance=1e-8,
                    power_tolerance=1e-7,
                    passivity_tolerance=1e-10,
                )
            )
            algebraic_certified = bool(
                port_certificate.certified
            )
            final_residual = float(
                result.normalized_residual
            )
            used_reference_fallback = True
            relative_observable_correction = float(
                np.linalg.norm(
                    result.impedance
                    - fast_impedance
                )
                / max(
                    np.linalg.norm(
                        result.impedance
                    ),
                    1e-30,
                )
            )
            fast_domain_valid = bool(
                relative_observable_correction
                <= fast_domain_correction_limit
            )
            status = (
                "REFERENCE_FALLBACK"
            )
    elif discretization_certified:
        status = (
            "CERTIFIED"
            if fast_domain_valid
            else "CORRECTED_OUT_OF_FAST_DOMAIN"
        )
    else:
        status = (
            "DISCRETE_CERTIFIED"
        )

    return CertifiedPortResult(
        status,
        result.impedance,
        result,
        port_certificate,
        initial_residual,
        final_residual,
        tuple(
            correction_iterations
        ),
        algebraic_certified,
        discretization_certified,
        used_reference_fallback,
        operator_backend,
        relative_observable_correction,
        fast_domain_valid,
    )

