import numpy as np
import pytest

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_model import UnifiedNeuralElectroThermalModel
from sdfmpneo.unified_runtime import _generate_tensor_dataset
from sdfmpneo.unified_tensor_training import train_matrix_tensor_surrogate
from sdfmpneo.unified_thermal import build_thermal_basis


MATERIALS = {
    "tx_copper": {"electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 1.0,
                   "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "rx_copper": {"electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 1.0,
                   "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "tx_package": {"electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 3.0,
                   "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "rx_package": {"electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 3.0,
                   "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "seawater": {"electrical_conductivity": 5.0, "resistivity_temperature_coefficient": 0.0,
                 "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 80.0,
                 "thermal_conductivity": 0.6, "volumetric_heat_capacity": 4.1e6},
}


def geometry(rx_x=0.0):
    coil = {"shape": "circle", "turns": 0.5, "outer_half_size": 0.012,
            "pitch": 0.002, "conductor_width": 0.001, "conductor_thickness": 0.001,
            "corner_radius": 0.006, "angles": [0.0, 0.0, 0.0]}
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.010]),
        "receiver": dict(coil, translation=[rx_x, 0.0, 0.010]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def small_background():
    axis = np.linspace(-0.05, 0.05, 5)
    return FixedMultiscaleBackground(
        axis, axis, axis, frequency_hz=100000.0,
        materials=MATERIALS, coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"), seawater_material="seawater",
        ambient_temperature=293.15,
    )


def test_tensor_rom_training_save_load_and_predict_without_online_maxwell(tmp_path):
    pytest.importorskip("torch")
    background = small_background()
    geometries = [geometry(0.0015 * np.sin(i)) for i in range(8)]
    _, thermal_report = build_thermal_basis(
        background,
        geometries[:2],
        validation_geometries=geometries[2:3],
        target_relative_error=0.9,
        time_scales=(0.1, 1.0),
    )
    assert thermal_report.converged
    assert background.thermal_rank > 0

    dataset = _generate_tensor_dataset(background, geometries[2:], seed=9)
    phi = background.thermal_basis
    surrogate, report = train_matrix_tensor_surrogate(
        dataset,
        np.min(phi, axis=0),
        np.max(phi, axis=0),
        network_settings={"width": 8, "blocks": 1, "activation": "silu"},
        training_settings={
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "patience": 2,
            "validation_interval": 1,
            "gradient_clip_norm": 10.0,
            "physics_penalty_weight": 0.01,
            "z_weight": 1.0,
            "d_weight": 1.0,
            "h_weight": 1.0,
            "seed": 3,
            "dtype": "float64",
        },
        device="cpu",
    )
    assert report.epochs_completed >= 1
    assert np.isfinite(report.best_validation_loss)

    model = UnifiedNeuralElectroThermalModel(
        background,
        surrogate,
        default_geometry=geometry(),
    )
    result = model.predict(
        0.0,
        initial_state=np.zeros(model.thermal_rank),
        geometry=geometry(0.001),
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

    model_path = tmp_path / "unified_model.npz"
    model.save(model_path)
    with np.load(model_path, allow_pickle=False) as data:
        assert "em_basis" not in data.files
        assert not any(name.startswith("maxwell") for name in data.files)
    loaded = UnifiedNeuralElectroThermalModel.load(model_path, device="cpu")
    loaded_result = loaded.predict(
        0.0,
        initial_state=np.zeros(loaded.thermal_rank),
        geometry=geometry(0.001),
        operating=[1.0, 0.0],
        max_step=1.0,
    )
    assert loaded.thermal_rank == model.thermal_rank
    assert np.allclose(loaded_result.impedance, result.impedance)
    assert np.allclose(loaded_result.heat_source, result.heat_source)
