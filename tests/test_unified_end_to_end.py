import numpy as np
import pytest

from sdfmpneo.unified_model import UnifiedNeuralElectroThermalModel
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo.unified_tensor_surrogate import (
    SpatialTensorDataset,
    encode_geometry,
    pack_spatial_tensors,
)
from sdfmpneo.unified_tensor_training import train_spatial_tensor_surrogate


MATERIALS = {
    "tx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "rx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "tx_package": {
        "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "rx_package": {
        "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "seawater": {
        "electrical_conductivity": 5.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 80.0,
        "thermal_conductivity": 0.6,
        "volumetric_heat_capacity": 4.1e6,
    },
}


def geometry(rx_x=0.0):
    coil = {
        "shape": "circle",
        "turns": 0.5,
        "outer_half_size": 0.012,
        "pitch": 0.002,
        "conductor_width": 0.001,
        "conductor_thickness": 0.001,
        "corner_radius": 0.006,
        "angles": [0.0, 0.0, 0.0],
    }
    return {
        "transmitter": dict(
            coil,
            translation=[0.0, 0.0, -0.010],
        ),
        "receiver": dict(
            coil,
            translation=[rx_x, 0.0, 0.010],
        ),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def small_background():
    axis = np.linspace(-0.05, 0.05, 5)
    return OpenBoundaryBackground(
        axis,
        axis,
        axis,
        frequency_hz=100000.0,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        ambient_temperature=293.15,
    )


def synthetic_spatial_dataset(background, geometries):
    inputs = []
    outputs = []
    centers = np.asarray(background.cell_centers, float)
    for mapping in geometries:
        rx_x = float(mapping["receiver"]["translation"][0])
        profile = np.exp(
            -(
                (centers[:, 0] - rx_x) ** 2
                + centers[:, 1] ** 2
                + centers[:, 2] ** 2
            )
            / (2.0 * 0.028**2)
        )
        profile += 0.15
        profile /= np.sum(profile)

        d = np.array(
            [
                [1.1 + 8.0 * rx_x, 0.16 + 0.04j],
                [0.16 - 0.04j, 0.85 - 5.0 * rx_x],
            ],
            complex,
        )
        assert np.min(np.linalg.eigvalsh(d)) > 0.0
        cells = profile[:, None, None] * d[None, :, :]
        d_out = np.eye(2) * 0.25
        reactance = np.array(
            [[0.30, -0.08], [-0.08, 0.22]],
            float,
        )
        z = d + d_out + 1j * reactance
        inputs.append(encode_geometry(mapping))
        outputs.append(pack_spatial_tensors(z, d, cells))

    split = np.asarray(
        ["train", "train", "train", "train", "train", "validation", "test", "audit"],
        dtype="U16",
    )
    return SpatialTensorDataset(
        np.asarray(inputs, float),
        np.asarray(outputs, float),
        split,
        {},
        2,
        background.n_cells,
    )


def test_spatial_tensor_training_save_load_and_predict_without_online_maxwell(tmp_path):
    pytest.importorskip("torch")
    background = small_background()
    geometries = [geometry(-0.003 + 0.0008 * i) for i in range(8)]
    dataset = synthetic_spatial_dataset(background, geometries)

    surrogate, report = train_spatial_tensor_surrogate(
        dataset,
        network_settings={
            "width": 8,
            "blocks": 1,
            "activation": "silu",
        },
        training_settings={
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "patience": 3,
            "validation_interval": 1,
            "gradient_clip_norm": 10.0,
            "physics_penalty_weight": 0.01,
            "z_weight": 1.0,
            "d_weight": 1.0,
            "spatial_weight": 1.0,
            "pod_relative_tail_tolerance": 0.5,
            "seed": 3,
            "dtype": "float64",
        },
        device="cpu",
    )
    assert report.epochs_completed >= 1
    assert report.pod_rank >= 1
    assert np.isfinite(report.best_validation_loss)

    model = UnifiedNeuralElectroThermalModel(
        background,
        surrogate,
        default_geometry=geometry(),
        thermal_time_scales=(0.1, 1.0),
        thermal_target_relative_error=0.99,
    )
    query_geometry = geometry(0.001)
    context = model.geometry_context(query_geometry)
    rank = model.thermal_rank_for(query_geometry)
    assert rank == context.thermal_basis.shape[1]
    assert 1 <= rank <= 1 + 3 * 7
    assert context.online_thermal_report.source_count == 6

    projected_uniform = model.project_initial_temperature(
        1.0,
        query_geometry,
    )
    assert np.allclose(
        context.thermal_basis @ projected_uniform,
        1.0,
        rtol=1e-9,
        atol=1e-9,
    )

    result = model.predict(
        0.0,
        initial_state=np.zeros(rank),
        geometry=query_geometry,
        operating=[1.0, 0.0],
        max_step=1.0,
    )
    assert result.steps == 0
    assert np.all(np.isfinite(result.state))
    assert np.all(np.isfinite(result.heat_source))
    assert np.isfinite(result.maximum_temperature)
    assert np.isfinite(result.volume_power)
    assert np.isfinite(result.wire_power)
    assert result.volume_power >= -1e-10
    assert result.outward_power >= -1e-10

    steady = model.steady_state(
        initial_guess=np.zeros(rank),
        geometry=query_geometry,
        operating=[0.0, 0.0],
        tolerance=1e-11,
        max_iterations=4,
    )
    assert steady.converged
    assert steady.stable
    assert steady.spectral_abscissa < 0.0
    assert np.allclose(steady.state, 0.0, atol=1e-12)

    model_path = tmp_path / "unified_model.npz"
    model.save(model_path)
    with np.load(model_path, allow_pickle=False) as data:
        assert "pod_basis" in data.files
        assert "thermal_background_modes" not in data.files
        assert "thermal_local_modes_0" not in data.files
        assert "thermal_basis" not in data.files
        assert "em_basis" not in data.files
        assert not any(
            name.startswith("maxwell")
            for name in data.files
        )

    loaded = UnifiedNeuralElectroThermalModel.load(
        model_path,
        device="cpu",
    )
    loaded_context = loaded.geometry_context(query_geometry)
    loaded_rank = loaded.thermal_rank_for(query_geometry)
    assert loaded_rank == rank
    assert np.allclose(
        loaded_context.thermal_basis,
        context.thermal_basis,
    )
    loaded_result = loaded.predict(
        0.0,
        initial_state=np.zeros(loaded_rank),
        geometry=query_geometry,
        operating=[1.0, 0.0],
        max_step=1.0,
    )
    assert loaded.surrogate.pod_rank == model.surrogate.pod_rank
    assert np.allclose(
        loaded_result.impedance,
        result.impedance,
    )
    assert np.allclose(
        loaded_result.heat_source,
        result.heat_source,
    )
