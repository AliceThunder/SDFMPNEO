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

    @property
    def certified(self) -> bool:
        return self.status == "CERTIFIED"


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
        status = (
            "REFERENCE_FALLBACK"
        )
    elif discretization_certified:
        status = "CERTIFIED"
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
    )



def _mixed_fast_lift(
    teacher: DenseMixedConductorTeacher,
    R,
    D,
    Phi,
    B,
    Q,
    fast_impedance,
):
    """Build a compatible current/potential/charge initial state."""
    _, _, current_constraint, current_port_map = (
        teacher._mqs.assemble()
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
) -> CertifiedPortResult:
    """CERTIFIED correction in the canonical current-potential-charge KKT."""
    if (
        algebraic_tolerance <= 0.0
        or correction_rtol <= 0.0
        or correction_restart < 1
        or correction_maxiter < 1
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

    teacher = DenseMixedConductorTeacher(
        scene,
        frequency_hz,
        config or MQSConfig(),
    )
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
    )
    lift = np.vstack(
        (
            current,
            potential_r,
            charge_r,
        )
    )
    row_weights = (
        _scaled_row_weights(
            K
        )
    )
    initial_residual = float(
        max(
            _scaled_residual(
                K,
                rhs_matrix[
                    :,
                    port,
                ],
                lift[
                    :,
                    port,
                ],
                row_weights,
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
            _scaled_residual(
                K,
                rhs_matrix[
                    :,
                    port,
                ],
                corrected[
                    :,
                    port,
                ],
                row_weights,
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
            status = (
                "REFERENCE_FALLBACK"
            )
    elif discretization_certified:
        status = "CERTIFIED"
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
    )
