"""Core geometry checks against the BFZI continuous-spiral equations.

Run explicitly with: python -m pytest tests/test_bfzi_geometry_v1.py -q
"""
import math

import numpy as np
import pytest

from sdfmpneo_bfzi.geometry import BFZICoil, RigidPose


@pytest.mark.parametrize("winding", [1, -1])
@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_side_corner_and_turn_connections(winding, alpha):
    coil = BFZICoil(
        turns=5, outer_half_size=0.16, corner_radius=0.08,
        pitch=0.006, alpha=alpha, winding=winding,
    )
    for k in range(coil.turns):
        for side in range(4):
            s_end, _ = coil.segment(k, "side", side, 1.0)
            c_start, _ = coil.segment(k, "corner", side, 0.0)
            np.testing.assert_allclose(s_end, c_start, atol=1e-12)
            c_end, _ = coil.segment(k, "corner", side, math.pi / 2.0)
            if side < 3:
                next_side, _ = coil.segment(k, "side", side + 1, -1.0)
                np.testing.assert_allclose(c_end, next_side, atol=1e-12)
            elif k + 1 < coil.turns:
                next_turn, _ = coil.segment(k + 1, "side", 0, -1.0)
                np.testing.assert_allclose(c_end, next_turn, atol=1e-12)


def test_corner_directed_line_element_is_derivative():
    coil = BFZICoil(6, 0.14, 0.085, 0.005, alpha=0.7, winding=-1)
    h = 1e-6
    for k, side, u in ((0, 0, 0.13), (2, 1, 0.71), (5, 3, 1.3)):
        right, _ = coil.segment(k, "corner", side, u + h)
        left, _ = coil.segment(k, "corner", side, u - h)
        _, tangent = coil.segment(k, "corner", side, u)
        np.testing.assert_allclose((right - left) / (2 * h), tangent, atol=3e-10)


def test_circular_limit_retains_tangential_pitch_offset():
    coil = BFZICoil(7, 0.16, 0.16, 0.007, alpha=1.0, winding=1)
    for k in (0, 4, 6):
        for m, u in ((0, 0.1), (1, 0.4), (3, 1.3)):
            r, dr = coil.segment(k, "corner", m, u)
            tau = m * math.pi / 2.0 + u
            n = np.array([math.cos(tau), math.sin(tau)])
            t = np.array([-math.sin(tau), math.cos(tau)])
            eta = k + tau / (2 * math.pi)
            target = (coil.outer_half_size - coil.pitch * eta) * n - coil.beta * t
            np.testing.assert_allclose(r[:2], target, atol=1e-12)
            assert r[2] == 0
            np.testing.assert_allclose(dr[:2], (coil.outer_half_size - coil.pitch * eta) * t, atol=1e-12)


def test_affine_turns_and_pose_equivariance():
    coil = BFZICoil(5, 0.18, 0.1, 0.005, alpha=0.4, winding=-1)
    angle = 0.41
    rot = np.array([[math.cos(angle), 0, math.sin(angle)],
                    [0, 1, 0],
                    [-math.sin(angle), 0, math.cos(angle)]])
    pose = RigidPose(rot, [0.02, -0.03, 0.11])
    for kind, parameter in (("side", 0.3), ("corner", 0.9)):
        p0, d0 = coil.segment(0, kind, 2, parameter, pose)
        p1, d1 = coil.segment(1, kind, 2, parameter, pose)
        for k in (2, 3, 4):
            pk, dk = coil.segment(k, kind, 2, parameter, pose)
            np.testing.assert_allclose(pk, p0 + k * (p1 - p0), atol=1e-12)
            np.testing.assert_allclose(dk, d0 + k * (d1 - d0), atol=1e-12)


def test_quadrature_and_invalid_shapes():
    coil = BFZICoil(3, 0.14, 0.08, 0.006, alpha=0.5)
    points, dl = coil.gauss_nodes(5)
    assert points.shape == dl.shape == (3 * 8 * 5, 3)
    outer, inner = coil.endpoints()
    np.testing.assert_allclose(dl.sum(axis=0), inner - outer, atol=2e-7)
    with pytest.raises(ValueError):
        BFZICoil(5, 0.1, 0.1, 0.01, alpha=0.1)
    with pytest.raises(ValueError):
        RigidPose(np.diag([1, 1, -1]), [0, 0, 0])


@pytest.mark.parametrize("winding", [+1, -1])
def test_corner_support_curve_rejects_reversing_tangent(winding):
    # Positive rho alone is not sufficient: for a rounded-square
    # corner with alpha=0, min corner speed is rho-2*beta.
    pitch=0.004
    bad_rho=0.0005
    with pytest.raises(ValueError, match="corner tangent"):
        BFZICoil(
            turns=1,outer_half_size=0.035,
            corner_radius=bad_rho,pitch=pitch,
            alpha=0.0,winding=winding,
        )
    good_rho=0.0025
    coil=BFZICoil(
        turns=1,outer_half_size=0.035,
        corner_radius=good_rho,pitch=pitch,
        alpha=0.0,winding=winding,
    )
    for j in range(4):
        _,dr=coil.segment(
            0,"corner",j,np.linspace(0,np.pi/2,21),
        )
        assert np.min(np.linalg.norm(dr,axis=1))>0.0
