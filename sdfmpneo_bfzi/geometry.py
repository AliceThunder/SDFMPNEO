"""Exact BFZI round-square/circular *continuous* spiral centerline.

This is not the historical superellipse family.  It implements the source
geometry equations from 0UnifiedModelingUWPT/sections/02_geometry_model.tex.
All distances are in meters.  The wire is represented by its centerline;
finite conductor cross-section, terminal leads and return path are separate
physical objects that must be supplied by a later source/port layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _normal(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)


def _tangent(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.stack((-np.sin(theta), np.cos(theta)), axis=-1)


@dataclass(frozen=True)
class RigidPose:
    """Right-handed rigid map: world = local @ rotation.T + center."""

    rotation: ArrayLike
    center: ArrayLike

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        center = np.asarray(self.center, dtype=float)
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise ValueError("rotation must be 3x3 and center must be a 3-vector")
        if not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise ValueError("rigid pose must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=1e-9):
            raise ValueError("rotation must be orthonormal")
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-9:
            raise ValueError("rotation must be proper (det=+1)")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "center", center.copy())

    @classmethod
    def identity(cls) -> "RigidPose":
        return cls(np.eye(3), np.zeros(3))

    def position(self, xyz: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(xyz, dtype=float) @ self.rotation.T + self.center

    def vector(self, xyz: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(xyz, dtype=float) @ self.rotation.T


@dataclass(frozen=True)
class BFZICoil:
    """Continuous round-square spiral defined by the current BFZI equations.

    Parameters
    ----------
    turns : int
        Number of full turns of the single continuous conductor.
    outer_half_size : float
        A_0 (meters), initial outer half-size / circular radius.
    corner_radius : float
        varrho_0 (meters), initial rounded-corner scale.
    pitch : float
        s (meters), nominal normal turn-to-turn pitch.
    alpha : float
        Fraction of pitch allocated to corner shrinkage, in [0, 1].
    winding : int
        +1 counterclockwise, -1 clockwise.
    """

    turns: int
    outer_half_size: float
    corner_radius: float
    pitch: float
    alpha: float = 1.0
    winding: int = 1

    def __post_init__(self) -> None:
        values = (self.outer_half_size, self.corner_radius, self.pitch, self.alpha)
        if isinstance(self.turns, bool) or not isinstance(self.turns, (int, np.integer)):
            raise ValueError("turns must be an integer")
        if self.turns < 1 or any(not math.isfinite(float(v)) for v in values):
            raise ValueError("turns must be positive and BFZI parameters finite")
        if self.winding not in (-1, 1):
            raise ValueError("winding must be +1 or -1")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        if not (self.outer_half_size > 0 and self.corner_radius > 0 and self.pitch > 0):
            raise ValueError("length parameters must be positive")
        if self.corner_radius > self.outer_half_size:
            raise ValueError("corner_radius may not exceed outer_half_size")
        # Full final corner ends at eta=N.  A, varrho and chi must
        # be physically admissible throughout that *continuous* interval.
        if self.outer_half_size - self.pitch * self.turns <= 0:
            raise ValueError("outer size collapses before final turn ends")
        if self.corner_radius - self.alpha * self.pitch * self.turns <= 0:
            raise ValueError("corner radius collapses before final turn ends")
        # The true BFZI support-function corner derivative is
        # g_corner = winding * [rho(eta) -
        #   2*winding*(1-alpha)*beta*(e_m dot t)] * t.
        # On each corner winding*(e_m dot t) lies in [-1,1],
        # hence rho_min>2*(1-alpha)*beta is a simple sufficient
        # non-reversal condition for both clockwise and anticlockwise
        # continuous spirals. rho>0 alone does NOT guarantee this.
        min_rho = self.corner_radius - self.alpha * self.pitch * self.turns
        if min_rho <= 2.0 * (1.0 - self.alpha) * self.beta:
            raise ValueError(
                "BFZI rounded corner tangent degenerates or reverses; "
                "increase corner radius, reduce pitch or adjust alpha"
            )
        straight_end = (
            self.outer_half_size - self.corner_radius
            - (1.0 - self.alpha) * self.pitch * self.turns
        )
        if straight_end < -1e-12 * self.outer_half_size:
            raise ValueError("straight half-length becomes negative")
        # These are necessary geometric constraints, not a finite-wire
        # cross-section clearance/collision certificate.

    @property
    def beta(self) -> float:
        return self.pitch / (2.0 * math.pi)

    def _scales(self, eta: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
        a = self.outer_half_size - self.pitch * eta
        rho = self.corner_radius - self.alpha * self.pitch * eta
        chi = a - rho
        return a, rho, chi

    @staticmethod
    def _xyz(xy: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.concatenate((xy, np.zeros(xy.shape[:-1] + (1,))), axis=-1)

    @staticmethod
    def _turn_index(turn: int, index: int) -> None:
        if not isinstance(turn, (int, np.integer)) or not 0 <= turn:
            raise ValueError("turn must be a nonnegative integer")
        if not isinstance(index, (int, np.integer)) or not 0 <= index < 4:
            raise ValueError("side/corner index must be in {0,1,2,3}")

    def segment(
        self,
        turn: int,
        kind: str,
        index: int,
        parameter: ArrayLike,
        pose: RigidPose | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return position and oriented d(position)/d(parameter.

        A side takes xi in [-1,1], a corner takes u in [0,pi/2].
        Both outputs have trailing dimension 3, with arbitrary leading
        parameter array shape.  Integrate the oriented derivative with
        respect to the *provided local segment parameter*.
        """
        self._turn_index(turn, index)
        if turn >= self.turns:
            raise ValueError("turn index exceeds coil.turns")
        t = np.asarray(parameter, dtype=float)
        if not np.isfinite(t).all():
            raise ValueError("segment parameter must be finite")
        sigma = self.winding
        beta = self.beta

        if kind == "side":
            if np.any((t < -1.0) | (t > 1.0)):
                raise ValueError("side parameter xi must lie in [-1, 1]")
            eta = turn + index / 4.0
            a, _, chi = self._scales(np.asarray(eta, dtype=float))
            psi = sigma * index * (math.pi / 2.0)
            n, tangent = _normal(np.asarray(psi)), _tangent(np.asarray(psi))
            xy = a * n + sigma * (chi * t[..., None] - beta) * tangent
            dxy = np.broadcast_to(sigma * chi * tangent, xy.shape).copy()
        elif kind == "corner":
            if np.any((t < 0.0) | (t > math.pi / 2.0)):
                raise ValueError("corner parameter u must lie in [0, pi/2]")
            eta = turn + index / 4.0 + t / (2.0 * math.pi)
            _, rho, chi = self._scales(eta)
            psi = sigma * (index * (math.pi / 2.0) + t)
            n = _normal(psi)
            tangent = _tangent(psi)
            psi0 = sigma * index * (math.pi / 2.0)
            e = _normal(np.asarray(psi0)) + sigma * _tangent(np.asarray(psi0))
            en = np.sum(n * e, axis=-1)
            et = np.sum(tangent * e, axis=-1)
            h = chi * en + rho
            hprime = -(1.0 - self.alpha) * beta * en + sigma * chi * et - self.alpha * beta
            xy = h[..., None] * n + sigma * hprime[..., None] * tangent
            dxy = (
                sigma
                * (rho - 2.0 * sigma * (1.0 - self.alpha) * beta * et)[..., None]
                * tangent
            )
        else:
            raise ValueError("kind must be 'side' or 'corner'")

        position, directed = self._xyz(xy), self._xyz(dxy)
        if pose is not None:
            position = pose.position(position)
            directed = pose.vector(directed)
        return position, directed

    def endpoints(self, pose: RigidPose | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Unconnected physical outer/inner wire ends; lead routing is external."""
        start, _ = self.segment(0, "side", 0, -1.0, pose)
        stop, _ = self.segment(self.turns - 1, "corner", 3, math.pi / 2.0, pose)
        return start, stop

    def gauss_nodes(
        self,
        order: int,
        pose: RigidPose | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Ordered Gauss source points and weighted oriented line elements.

        The returned line elements already incorporate Gauss weights.
        This covers *only* the BFZI spiral; complete port sources must
        also include external terminal leads and the return path.
        """
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("Gauss order must be a positive integer")
        x, w = np.polynomial.legendre.leggauss(order)
        uc = (math.pi / 4.0) * (x + 1.0)
        wc = (math.pi / 4.0) * w
        pts: list[NDArray[np.float64]] = []
        directions: list[NDArray[np.float64]] = []
        for turn in range(self.turns):
            for side in range(4):
                for kind, zeta, weights in (("side", x, w), ("corner", uc, wc)):
                    p, d = self.segment(turn, kind, side, zeta, pose)
                    pts.append(p)
                    directions.append(d * weights[:, None])
        return np.concatenate(pts, axis=0), np.concatenate(directions, axis=0)
