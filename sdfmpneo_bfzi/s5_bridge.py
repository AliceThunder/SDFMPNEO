"""S5 V150 thermal-skin anchor adapted to GeomFracET's five material groups.

Only changes the layout of stored state and FV operators. No physical
solver approximation or spatial coarsening is introduced by this adapter.
"""
from __future__ import annotations

import numpy as np

from .geom_heat import GeomHeat, pack
from .pose import desc

THERM_SCALE = np.array([.02, .25, .001, .25, 20.], dtype=float)


def state_order(nwet, npkg, nskin):
    """BFZI: wet, pkg1, pkg2, copper1, copper2, skin.

    GeomFracET: wet, skin, pkg1, pkg2, copper1, copper2.
    """
    nwet, npkg, nskin = int(nwet), int(npkg), int(nskin)
    if min(nwet, npkg, nskin) <= 0:
        raise ValueError("S5 wet/package/skin partitions must be nonempty")
    return np.r_[np.arange(nwet),
                 np.arange(nwet + 2*npkg + 2, nwet + 2*npkg + 2 + nskin),
                 np.arange(nwet, nwet + 2*npkg + 2)]


def skin_field(mix, wet, shell_vol):
    """Volume-consistent weak source/current density on sea skin nodes."""
    data = np.asarray(wet)
    out = mix.T @ data.reshape(len(data), -1)
    shape = (-1,) + (1,)*(data.ndim-1)
    return np.asarray(out).reshape((len(shell_vol),) + data.shape[1:]) / np.asarray(shell_vol).reshape(shape)


def from_skin(em, et, ext, *, thermal):
    """Export a cold BFZI S5 model into one GeomHeat-compatible skin anchor.

    This does ONE original-BFZI cold solve. Existing S5 transient solver and
    its truth data remain unchanged. A conserved volume-weighted native weak
    heat shape is used for the neural spatial field head.
    """
    nwet = int(et.count)
    npkg = int(et.pkg_n)
    nskin = int(et.skin.stop-et.skin.start)
    order = state_order(nwet, npkg, nskin)
    if len(order) != et.n or len(et.skin_xyz) != nskin:
        raise ValueError("S5 thermal state or physical skin centers mismatch")
    cold = em.solve(np.full(nwet, et.amb), np.full(2, et.amb))
    families = em.base["families"]
    g = np.asarray([
        *[float(f.coil.outer_radius_m) for f in families],
        *[float(f.coil.rounded_square_core_half_m) for f in families],
        *[float(f.coil.rounded_square_alpha) for f in families],
        *[float(f.coil.turns) for f in families],
        *[float(f.coil.wire_radius_m) for f in families],
        float(np.linalg.norm(np.asarray(families[0].coil.center)-np.asarray(families[1].coil.center))),
        em.air[0, 0], em.air[0, 1], em.air[1, 1],
        *np.asarray(et.cap[et.cu], float)], float)
    vol = np.r_[et.bulk_vol, et.skin_vol]
    xyz = np.vstack((em.points, et.skin_xyz))
    unit = np.concatenate((cold.unit_j,
                           skin_field(et.skin_mix, cold.unit_j, et.skin_vol)), axis=0)
    weak = np.concatenate((cold.weak_q,
                           skin_field(et.skin_mix, cold.weak_q, et.skin_vol)), axis=0)
    cap = np.asarray(et.cap)[order]
    H = et.H.tocsr()[order, :][:, order]
    therm = np.asarray(thermal, float)
    if therm.shape != (5,) or not np.isfinite(therm).all() or np.any(therm <= 0):
        raise ValueError("S5 thermal parameters: alpha, insulation k/t, package k, water h")
    head = GeomHeat(
        g, pack(cold.total_z), ext, vol, xyz, np.asarray(et.pkg_xyz),
        cap, H, unit, weak, ambient=et.amb, alpha=em.alpha,
        pose=desc([f.coil for f in families], et.boxes),
        pkg_axes=np.asarray(et.pkg_axes),
        pkg_mid=np.asarray([box[0] for box in et.boxes]),
        pkg_half=np.asarray([box[1] for box in et.boxes]))
    head.state_order = order
    head.thermal = therm
    head.nwet = nwet
    head.nskin = nskin
    return head


def feature(rom, ng):
    """Use existing 40D BFZI feature; optionally append five physical inputs."""
    from .geom_memory import G_SCALE, geom_feat
    base = geom_feat(rom.geom, rom.z0, rom.pose if ng > len(G_SCALE) else None)
    if ng == len(base):
        return base
    if ng == len(base)+5 and hasattr(rom, "thermal"):
        return np.r_[base, np.asarray(rom.thermal) / THERM_SCALE]
    raise ValueError("GeomFrac geometry/thermal feature dimension mismatch")


def teacher_temp(temp, rom):
    """Return physical S5 T in GeomFrac ordering, without losing skin cells."""
    a = np.asarray(temp, float)
    if hasattr(rom, "state_order"):
        order = np.asarray(rom.state_order, int)
        if a.shape[-1] != len(order):
            raise ValueError("teacher thermal node count differs from S5 anchor")
        a = a[..., order]
    return a
