from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


RawModeSpec = tuple[
    str,
    float,
    float,
    float,
]


@dataclass(frozen=True)
class SectionQuadrature:
    xy: np.ndarray
    weights: np.ndarray
    rho: np.ndarray
    theta: np.ndarray


def superellipse_section_quadrature(
    width: float,
    thickness: float,
    exponent: float,
    radial_order: int = 5,
    angular_order: int = 32,
) -> SectionQuadrature:
    if (
        width <= 0
        or thickness <= 0
        or exponent < 2
    ):
        raise ValueError(
            "invalid superellipse section"
        )
    if (
        radial_order < 2
        or angular_order < 8
    ):
        raise ValueError(
            "quadrature orders are too small"
        )
    xg, wg = leggauss(
        radial_order
    )
    rho = 0.5 * (
        xg + 1.0
    )
    wr = 0.5 * wg
    theta = (
        np.arange(
            angular_order
        )
        + 0.5
    ) * (
        2.0
        * np.pi
        / angular_order
    )
    wt = np.full(
        angular_order,
        2.0
        * np.pi
        / angular_order,
    )
    rr, tt = np.meshgrid(
        rho,
        theta,
        indexing="ij",
    )
    wrr, wtt = np.meshgrid(
        wr,
        wt,
        indexing="ij",
    )
    a = 0.5 * width
    b = 0.5 * thickness
    denom = (
        (
            np.abs(
                np.cos(
                    tt
                )
            )
            / a
        ) ** exponent
        + (
            np.abs(
                np.sin(
                    tt
                )
            )
            / b
        ) ** exponent
    )
    rb = denom ** (
        -1.0
        / exponent
    )
    x = (
        rr
        * rb
        * np.cos(
            tt
        )
    )
    y = (
        rr
        * rb
        * np.sin(
            tt
        )
    )
    jac = (
        rr
        * rb**2
    )
    weights = (
        wrr
        * wtt
        * jac
    )
    return SectionQuadrature(
        np.stack(
            (
                x.ravel(),
                y.ravel(),
            ),
            axis=1,
        ),
        weights.ravel(),
        rr.ravel(),
        tt.ravel(),
    )


def _monomial_powers(
    degree: int,
):
    out = []
    for total in range(
        degree + 1
    ):
        for px in range(
            total + 1
        ):
            out.append(
                (
                    px,
                    total - px,
                )
            )
    return tuple(
        out
    )


def _normalized_coordinates(
    xy,
    width: float,
    thickness: float,
    exponent: float,
):
    points = np.asarray(
        xy,
        dtype=float,
    )
    points = np.atleast_2d(
        points
    )
    if (
        points.ndim != 2
        or points.shape[1]
        != 2
    ):
        raise ValueError(
            "xy must have shape (2,) or (n,2)"
        )
    x = (
        points[:, 0]
        / (
            0.5
            * width
        )
    )
    y = (
        points[:, 1]
        / (
            0.5
            * thickness
        )
    )
    rho = (
        np.abs(
            x
        ) ** exponent
        + np.abs(
            y
        ) ** exponent
    ) ** (
        1.0
        / exponent
    )

    theta = np.zeros_like(
        rho
    )
    nonzero = (
        rho > 1e-15
    )
    if np.any(
        nonzero
    ):
        cos_theta = (
            np.sign(
                x[
                    nonzero
                ]
            )
            * np.abs(
                x[
                    nonzero
                ]
                / rho[
                    nonzero
                ]
            ) ** (
                0.5
                * exponent
            )
        )
        sin_theta = (
            np.sign(
                y[
                    nonzero
                ]
            )
            * np.abs(
                y[
                    nonzero
                ]
                / rho[
                    nonzero
                ]
            ) ** (
                0.5
                * exponent
            )
        )
        theta[
            nonzero
        ] = np.arctan2(
            sin_theta,
            cos_theta,
        )
    return (
        x,
        y,
        rho,
        theta,
    )


def _raw_mode_matrix(
    xy,
    width: float,
    thickness: float,
    exponent: float,
    specs,
):
    (
        x,
        y,
        rho,
        theta,
    ) = (
        _normalized_coordinates(
            xy,
            width,
            thickness,
            exponent,
        )
    )
    columns = []
    for spec in specs:
        kind = spec[
            0
        ]
        if kind == "monomial":
            px = int(
                spec[
                    1
                ]
            )
            py = int(
                spec[
                    2
                ]
            )
            columns.append(
                x**px
                * y**py
            )
        elif kind == "boundary_cos":
            lam = float(
                spec[
                    1
                ]
            )
            harmonic = int(
                spec[
                    2
                ]
            )
            columns.append(
                np.exp(
                    -lam
                    * (
                        1.0
                        - rho
                    )
                )
                * np.cos(
                    harmonic
                    * theta
                )
            )
        elif kind == "boundary_sin":
            lam = float(
                spec[
                    1
                ]
            )
            harmonic = int(
                spec[
                    2
                ]
            )
            columns.append(
                np.exp(
                    -lam
                    * (
                        1.0
                        - rho
                    )
                )
                * np.sin(
                    harmonic
                    * theta
                )
            )
        else:
            raise ValueError(
                f"unknown raw section mode: {kind}"
            )
    if not columns:
        raise ValueError(
            "section basis requires at least one raw mode"
        )
    return np.stack(
        columns,
        axis=1,
    )


def _weighted_modified_gram_schmidt(
    raw,
    weights,
    *,
    tolerance: float = 1e-12,
):
    raw = np.asarray(
        raw,
        dtype=float,
    )
    weights = np.asarray(
        weights,
        dtype=float,
    )
    if (
        raw.ndim != 2
        or weights.shape
        != (
            raw.shape[
                0
            ],
        )
    ):
        raise ValueError(
            "invalid weighted basis arrays"
        )
    if np.any(
        weights <= 0.0
    ):
        raise ValueError(
            "quadrature weights must be positive"
        )

    values = []
    transforms = []
    n_raw = raw.shape[
        1
    ]
    for column in range(
        n_raw
    ):
        vector = raw[
            :,
            column,
        ].copy()
        coefficients = np.zeros(
            n_raw,
            dtype=float,
        )
        coefficients[
            column
        ] = 1.0

        # Two modified Gram-Schmidt sweeps are cheap at the small intrinsic
        # ranks used here and materially improve loss of orthogonality.
        for _ in range(
            2
        ):
            for basis_value, basis_coefficients in zip(
                values,
                transforms,
            ):
                projection = float(
                    np.dot(
                        weights
                        * basis_value,
                        vector,
                    )
                )
                vector -= (
                    projection
                    * basis_value
                )
                coefficients -= (
                    projection
                    * basis_coefficients
                )

        norm = float(
            np.sqrt(
                max(
                    np.dot(
                        weights
                        * vector,
                        vector,
                    ),
                    0.0,
                )
            )
        )
        raw_norm = float(
            np.sqrt(
                max(
                    np.dot(
                        weights
                        * raw[
                            :,
                            column,
                        ],
                        raw[
                            :,
                            column,
                        ],
                    ),
                    0.0,
                )
            )
        )
        threshold = (
            tolerance
            * max(
                raw_norm,
                1.0,
            )
        )
        if norm <= threshold:
            continue
        values.append(
            vector
            / norm
        )
        transforms.append(
            coefficients
            / norm
        )

    if not values:
        raise ValueError(
            "cross-section basis is numerically empty"
        )
    return (
        np.stack(
            values,
            axis=1,
        ),
        np.stack(
            transforms,
            axis=1,
        ),
    )


@dataclass(frozen=True)
class SectionBasis:
    quadrature: SectionQuadrature
    values: np.ndarray
    moments: np.ndarray
    degree: int
    width: float
    thickness: float
    exponent: float
    powers: tuple[
        tuple[int, int],
        ...,
    ]
    transform: np.ndarray
    raw_specs: tuple[
        RawModeSpec,
        ...,
    ]
    skin_parameter: float = 0.0

    @property
    def n_modes(
        self,
    ) -> int:
        return int(
            self.values.shape[
                1
            ]
        )

    @property
    def area(
        self,
    ) -> float:
        return float(
            np.sum(
                self.quadrature.weights
            )
        )

    def evaluate_xy(
        self,
        xy,
    ) -> np.ndarray:
        points = np.asarray(
            xy,
            dtype=float,
        )
        scalar = (
            points.ndim
            == 1
        )
        raw = _raw_mode_matrix(
            points,
            self.width,
            self.thickness,
            self.exponent,
            self.raw_specs,
        )
        out = (
            raw
            @ self.transform
        )
        return (
            out[
                0
            ]
            if scalar
            else out
        )


def _build_section_basis(
    width: float,
    thickness: float,
    exponent: float,
    degree: int,
    radial_order: int,
    angular_order: int,
    raw_specs,
    *,
    skin_parameter: float = 0.0,
) -> SectionBasis:
    q = (
        superellipse_section_quadrature(
            width,
            thickness,
            exponent,
            radial_order,
            angular_order,
        )
    )
    raw_specs = tuple(
        raw_specs
    )
    raw = _raw_mode_matrix(
        q.xy,
        width,
        thickness,
        exponent,
        raw_specs,
    )
    values, transform = (
        _weighted_modified_gram_schmidt(
            raw,
            q.weights,
        )
    )
    moments = (
        values.T
        @ q.weights
    )

    # The raw constant is deliberately first, so the first orthonormal mode is
    # exactly the normalized constant. All retained later modes are zero-mean
    # to quadrature precision and therefore cannot alter the prescribed port
    # current by themselves.
    area = float(
        np.sum(
            q.weights
        )
    )
    expected_constant = (
        1.0
        / np.sqrt(
            area
        )
    )
    if not np.allclose(
        values[
            :,
            0,
        ],
        expected_constant,
        rtol=2e-12,
        atol=2e-12
        * max(
            abs(
                expected_constant
            ),
            1.0,
        ),
    ):
        raise RuntimeError(
            "section orthogonalization failed to preserve the constant mode"
        )

    powers = tuple(
        (
            int(
                spec[
                    1
                ]
            ),
            int(
                spec[
                    2
                ]
            ),
        )
        for spec in raw_specs
        if spec[
            0
        ]
        == "monomial"
    )
    return SectionBasis(
        q,
        values,
        moments,
        int(
            degree
        ),
        float(
            width
        ),
        float(
            thickness
        ),
        float(
            exponent
        ),
        powers,
        transform,
        raw_specs,
        float(
            skin_parameter
        ),
    )


def polynomial_section_basis(
    width: float,
    thickness: float,
    exponent: float,
    degree: int = 1,
    radial_order: int = 5,
    angular_order: int = 32,
) -> SectionBasis:
    if degree < 0:
        raise ValueError(
            "degree must be nonnegative"
        )
    specs = [
        (
            "monomial",
            float(
                px
            ),
            float(
                py
            ),
            0.0,
        )
        for px, py
        in _monomial_powers(
            degree
        )
    ]
    return _build_section_basis(
        width,
        thickness,
        exponent,
        degree,
        radial_order,
        angular_order,
        specs,
        skin_parameter=0.0,
    )


def adaptive_section_basis(
    width: float,
    thickness: float,
    exponent: float,
    *,
    degree: int = 1,
    radial_order: int = 5,
    angular_order: int = 32,
    skin_parameter: float = 0.0,
    skin_threshold: float = 2.0,
    boundary_layers: int = 2,
    boundary_angular_order: int = 1,
    lambda_cap: float = 24.0,
) -> SectionBasis:
    """Nested polynomial basis enriched by intrinsic skin boundary layers."""
    if (
        degree < 0
        or skin_parameter < 0.0
        or skin_threshold < 0.0
        or boundary_layers < 0
        or boundary_angular_order < 0
        or lambda_cap <= 0.0
    ):
        raise ValueError(
            "invalid adaptive section-basis configuration"
        )

    specs = [
        (
            "monomial",
            float(
                px
            ),
            float(
                py
            ),
            0.0,
        )
        for px, py
        in _monomial_powers(
            degree
        )
    ]

    if (
        skin_parameter
        > skin_threshold
        and boundary_layers
        > 0
    ):
        if boundary_layers == 1:
            factors = np.asarray(
                [
                    1.0,
                ],
                dtype=float,
            )
        else:
            factors = np.geomspace(
                0.5,
                1.5,
                boundary_layers,
            )
        for factor in factors:
            lam = min(
                float(
                    factor
                    * skin_parameter
                ),
                float(
                    lambda_cap
                ),
            )
            specs.append(
                (
                    "boundary_cos",
                    lam,
                    0.0,
                    0.0,
                )
            )
            for harmonic in range(
                1,
                boundary_angular_order
                + 1,
            ):
                specs.append(
                    (
                        "boundary_cos",
                        lam,
                        float(
                            harmonic
                        ),
                        0.0,
                    )
                )
                specs.append(
                    (
                        "boundary_sin",
                        lam,
                        float(
                            harmonic
                        ),
                        0.0,
                    )
                )

        maximum_lambda = max(
            spec[
                1
            ]
            for spec in specs
            if spec[
                0
            ].startswith(
                "boundary_"
            )
        )
        radial_order = max(
            int(
                radial_order
            ),
            int(
                6
                + np.ceil(
                    maximum_lambda
                    / 3.0
                )
            ),
        )

    return _build_section_basis(
        width,
        thickness,
        exponent,
        degree,
        radial_order,
        angular_order,
        specs,
        skin_parameter=(
            skin_parameter
        ),
    )
