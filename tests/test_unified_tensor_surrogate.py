import numpy as np

from sdfmpneo.unified_tensor_surrogate import (
    decode_physical_tensors,
    encode_geometry,
    pack_tensors,
    unpack_tensors,
)


def geometry(shape="circle"):
    coil = {
        "shape": shape,
        "turns": 0.5,
        "outer_half_size": 0.012,
        "pitch": 0.002,
        "conductor_width": 0.001,
        "conductor_thickness": 0.001,
        "corner_radius": 0.006,
        "angles": [0.1, -0.2, 0.3],
    }
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.01]),
        "receiver": dict(coil, translation=[0.001, 0.0, 0.01]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def test_pack_unpack_preserves_complex_symmetric_and_hermitian_tensors():
    z = np.array([[2 + 1j, 0.3 - 0.4j], [0.3 - 0.4j, 1.5 + 0.2j]])
    d = np.array([[1.0, 0.2 + 0.1j], [0.2 - 0.1j, 0.7]])
    h = np.array([
        [[0.4, 0.1 - 0.05j], [0.1 + 0.05j, 0.2]],
        [[0.8, -0.03 + 0.02j], [-0.03 - 0.02j, 0.5]],
    ])
    packed = pack_tensors(z, d, h)
    z2, d2, h2 = unpack_tensors(packed, 2, 2)
    assert np.allclose(z2, z)
    assert np.allclose(d2, d)
    assert np.allclose(h2, h)


def test_physical_decoder_enforces_passivity_and_modal_loewner_bounds():
    z = np.array([[0.2 + 1.0j, 0.1 + 0.3j], [0.1 + 0.3j, 0.1 + 0.4j]])
    d = np.array([[1.0, 0.8 + 0.2j], [0.8 - 0.2j, -0.3]])
    h = np.array([
        [[4.0, 1.0 + 0.5j], [1.0 - 0.5j, -2.0]],
        [[-1.0, 0.5 - 0.2j], [0.5 + 0.2j, 3.0]],
    ])
    phi_min = np.array([-1.0, 0.2])
    phi_max = np.array([2.0, 1.5])
    decoded = decode_physical_tensors(pack_tensors(z, d, h), 2, phi_min, phi_max)

    assert np.allclose(decoded.z_field, decoded.z_field.T)
    assert np.allclose(decoded.d_vol, decoded.d_vol.conj().T)
    assert np.min(np.linalg.eigvalsh(decoded.d_vol)) >= -1e-10
    assert np.min(np.linalg.eigvalsh(decoded.implied_d_out)) >= -1e-10
    for j, modal in enumerate(decoded.modal_h):
        assert np.allclose(modal, modal.conj().T)
        assert np.min(np.linalg.eigvalsh(modal - phi_min[j] * decoded.d_vol)) >= -1e-10
        assert np.min(np.linalg.eigvalsh(phi_max[j] * decoded.d_vol - modal)) >= -1e-10
    assert decoded.projection_correction > 0.0


def test_geometry_encoding_is_fixed_width_across_supported_shapes():
    circle = encode_geometry(geometry("circle"))
    square = encode_geometry(geometry("rounded_square"))
    assert circle.ndim == square.ndim == 1
    assert circle.shape == square.shape
    assert np.all(np.isfinite(circle))
    assert np.all(np.isfinite(square))
    assert not np.allclose(circle, square)


def test_zero_rank_global_tensor_roundtrip_is_well_shaped():
    z = np.array(
        [[2.0 + 0.2j, 0.1 - 0.05j], [0.1 - 0.05j, 1.6 + 0.3j]],
        complex,
    )
    d = np.array(
        [[1.2, 0.08 + 0.02j], [0.08 - 0.02j, 0.9]],
        complex,
    )
    packed = pack_tensors(
        z,
        d,
        np.empty((0, 2, 2), complex),
    )
    z2, d2, modal = unpack_tensors(packed, 2, 0)
    assert modal.shape == (0, 2, 2)
    assert np.allclose(z2, z)
    assert np.allclose(d2, d)

    decoded = decode_physical_tensors(
        packed,
        2,
        np.empty(0),
        np.empty(0),
    )
    assert decoded.modal_h.shape == (0, 2, 2)
    assert np.min(
        np.linalg.eigvalsh(decoded.d_vol).real
    ) >= -1e-12
    assert np.min(
        np.linalg.eigvalsh(decoded.implied_d_out).real
    ) >= -1e-12
