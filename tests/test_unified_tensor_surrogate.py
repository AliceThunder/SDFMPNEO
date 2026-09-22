from types import SimpleNamespace
import numpy as np

from sdfmpneo.unified_tensor_surrogate import (
    decode_physical_tensors,
    decode_spatial_global_tensors,
    encode_geometry,
    encode_geometry_invariant,
    normalize_cell_joule_tensors,
    pack_spatial_global_tensors,
    pack_tensors,
    pack_whitened_field_factors,
    spatial_joule_density_prior,
    unpack_tensors,
    unpack_whitened_field_factors,
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


def test_spatial_global_representation_preserves_small_outward_matrix():
    d = np.array(
        [[5.0, 0.4 + 0.12j], [0.4 - 0.12j, 4.0]],
        complex,
    )
    d_out = np.array(
        [[2.0e-3, 3.0e-4 - 0.12j], [3.0e-4 + 0.12j, 1.5e-3]],
        complex,
    )
    # Make the example exactly passive while preserving the required
    # Im(D_out)=-Im(D) reciprocity identity.
    minimum = np.min(np.linalg.eigvalsh(d_out).real)
    if minimum < 0.0:
        d_out = d_out + (-minimum + 1e-3) * np.eye(2)
    z = (d + d_out).real + 1j * np.array(
        [[0.3, -0.07], [-0.07, 0.45]],
        float,
    )
    packed = pack_spatial_global_tensors(z, d)
    decoded = decode_spatial_global_tensors(packed, 2)
    assert np.allclose(decoded.d_vol, d, rtol=1e-12, atol=1e-12)
    assert np.allclose(decoded.z_field, z, rtol=1e-12, atol=1e-12)
    assert np.allclose(
        decoded.implied_d_out,
        d_out,
        rtol=1e-10,
        atol=1e-10,
    )
    assert decoded.zd_projection_correction <= 1e-10


def test_whitened_field_factor_roundtrip_is_psd_and_density_exact():
    rng = np.random.default_rng(91)
    count = 23
    n = 2
    factor = (
        rng.normal(size=(count, n, n))
        + 1j * rng.normal(size=(count, n, n))
    )
    cells = np.einsum(
        "kab,kcb->kac",
        factor,
        factor.conj(),
        optimize=True,
    )
    _, cells = normalize_cell_joule_tensors(
        cells,
        np.eye(n, dtype=complex),
    )
    packed = pack_whitened_field_factors(
        cells,
        total_cells=count,
    )
    restored = unpack_whitened_field_factors(
        packed,
        n_ports=n,
        total_cells=count,
    )
    assert np.min(np.linalg.eigvalsh(restored).real) >= -1e-12
    assert np.allclose(
        restored,
        cells,
        rtol=2e-10,
        atol=2e-12,
    )


def test_invariant_geometry_encoding_ignores_common_rigid_translation():
    first = geometry("circle")
    shifted = geometry("circle")
    shift = np.array([0.013, -0.021, 0.007])
    for side in ("transmitter", "receiver"):
        shifted[side]["translation"] = (
            np.asarray(shifted[side]["translation"], float) + shift
        ).tolist()
    assert np.allclose(
        encode_geometry_invariant(first),
        encode_geometry_invariant(shifted),
        rtol=0.0,
        atol=1e-12,
    )



def test_density_prior_is_positive_and_mean_one_on_full_grid():
    weights_a = np.array([0.6, 0.3, 0.1, 0.0])
    weights_b = np.array([0.0, 0.2, 0.3, 0.5])
    context = SimpleNamespace(
        line_heat_weights=(weights_a, weights_b),
    )
    background = SimpleNamespace(n_cells=4)
    prior = spatial_joule_density_prior(
        background,
        context,
        strength=0.9,
    )
    assert np.all(prior > 0.0)
    assert np.isclose(np.mean(prior), 1.0)


def test_whitened_field_factor_roundtrip_with_density_prior():
    rng = np.random.default_rng(123)
    count = 17
    n = 2
    factor = (
        rng.normal(size=(count, n, n))
        + 1j * rng.normal(size=(count, n, n))
    )
    cells = np.einsum(
        "kab,kcb->kac",
        factor,
        factor.conj(),
        optimize=True,
    )
    _, cells = normalize_cell_joule_tensors(
        cells,
        np.eye(n, dtype=complex),
    )
    prior = np.linspace(0.15, 1.85, count)
    prior /= np.mean(prior)
    packed = pack_whitened_field_factors(
        cells,
        total_cells=count,
        density_prior=prior,
    )
    restored = unpack_whitened_field_factors(
        packed,
        n_ports=n,
        total_cells=count,
        density_prior=prior,
        log_density_bounds=(-35.0, 35.0),
    )
    assert np.min(np.linalg.eigvalsh(restored).real) >= -1e-12
    assert np.allclose(
        restored,
        cells,
        rtol=2e-10,
        atol=2e-12,
    )
