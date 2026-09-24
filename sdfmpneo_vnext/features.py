from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import MU0, Scene


@dataclass(frozen=True)
class EncodedScene:
    node_features: np.ndarray
    pair_features: np.ndarray
    length_scale: float


def _skin_depth(
    frequency_hz: float,
    conductivity: float,
    permeability: float,
) -> float:
    if frequency_hz <= 0.0:
        return np.inf
    omega = 2.0 * np.pi * frequency_hz
    return float(
        np.sqrt(
            2.0
            / (
                omega
                * permeability
                * conductivity
            )
        )
    )


def encode_scene_invariant(
    scene: Scene,
    frequency_hz: float,
) -> EncodedScene:
    scene.require_mvp_electromagnetic_scope()
    if (
        not np.isfinite(frequency_hz)
        or frequency_hz < 0.0
    ):
        raise ValueError(
            "frequency_hz must be finite and nonnegative"
        )
    radii = np.array(
        [
            np.sqrt(
                coil.geometry.outer_a
                * coil.geometry.outer_b
            )
            for coil in scene.coils
        ],
        dtype=float,
    )
    length_scale = float(
        np.exp(
            np.mean(
                np.log(radii)
            )
        )
    )
    length_scale = max(
        length_scale,
        1e-12,
    )
    node = []
    for coil in scene.coils:
        g = coil.geometry
        mat = coil.material
        delta = _skin_depth(
            frequency_hz,
            mat.conductivity,
            MU0 * mat.relative_permeability,
        )
        skin_w = (
            0.0
            if not np.isfinite(delta)
            else g.conductor_width / delta
        )
        skin_h = (
            0.0
            if not np.isfinite(delta)
            else g.conductor_thickness / delta
        )
        node.append(
            [
                g.outer_a / length_scale,
                g.outer_b / length_scale,
                g.turns,
                g.pitch_a / length_scale,
                g.pitch_b / length_scale,
                g.exponent,
                g.conductor_width / length_scale,
                g.conductor_thickness / length_scale,
                g.cross_section_exponent,
                np.log(
                    mat.conductivity
                    / 1e7
                ),
                mat.relative_permeability,
                skin_w,
                skin_h,
                scene.medium.relative_permittivity,
                scene.medium.relative_permeability,
                np.log1p(
                    scene.medium.conductivity
                    / 1e-6
                ),
                np.log1p(
                    frequency_hz
                    / 1e3
                ),
            ]
        )
    node_features = np.asarray(
        node,
        dtype=float,
    )

    n = len(scene.coils)
    pair = np.zeros(
        (n, n, 15),
        dtype=float,
    )
    for i, ci in enumerate(
        scene.coils
    ):
        Ri = ci.geometry.pose.rotation
        ti = ci.geometry.pose.translation
        for j, cj in enumerate(
            scene.coils
        ):
            if i == j:
                pair[i, j, 3:12] = (
                    np.eye(3).ravel()
                )
                continue
            Rj = cj.geometry.pose.rotation
            tj = cj.geometry.pose.translation
            relative_translation = (
                Ri.T @ (tj - ti)
            ) / length_scale
            relative_rotation = (
                Ri.T @ Rj
            )
            size_i = np.sqrt(
                ci.geometry.outer_a
                * ci.geometry.outer_b
            )
            size_j = np.sqrt(
                cj.geometry.outer_a
                * cj.geometry.outer_b
            )
            pair[i, j, :3] = (
                relative_translation
            )
            pair[i, j, 3:12] = (
                relative_rotation.ravel()
            )
            pair[i, j, 12] = (
                np.linalg.norm(
                    relative_translation
                )
            )
            pair[i, j, 13] = (
                size_j
                / max(size_i, 1e-12)
            )
            pair[i, j, 14] = (
                (
                    ci.geometry.equivalent_radius
                    + cj.geometry.equivalent_radius
                )
                / max(
                    np.linalg.norm(tj - ti),
                    1e-12,
                )
            )
    return EncodedScene(
        node_features,
        pair,
        length_scale,
    )
