import numpy as np
import pytest

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_dataset import generate_residual_dataset
from sdfmpneo.unified_maxwell import NeuralMaxwellAccelerator
from sdfmpneo.unified_model import UnifiedNeuralElectroThermalModel
from sdfmpneo.unified_thermal import build_thermal_basis
from sdfmpneo.unified_trainer import train_maxwell_accelerator


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
    return FixedMultiscaleBackground(axis, axis, axis, frequency_hz=100000.0,
        materials=MATERIALS, coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"), seawater_material="seawater",
        ambient_temperature=293.15)


def test_unified_fullspace_training_save_load_and_predict(tmp_path):
    pytest.importorskip("torch")
    background = small_background()
    geometries = [geometry(0.002 * np.sin(i)) for i in range(12)]
    _, thermal_report = build_thermal_basis(background, geometries[:2], target_relative_residual=0.8)
    assert thermal_report.converged
    assert background.thermal_rank > 0

    states = [{"tx_copper": float(i), "rx_copper": float(11 - i)} for i in range(12)]
    dataset = generate_residual_dataset(background, geometries, states, seed=9, residual_steps=1)
    network, report = train_maxwell_accelerator(
        background, dataset,
        network_settings={"width": 8, "message_passing_steps": 1, "activation": "silu"},
        training_settings={"epochs": 2, "batch_size": 1, "learning_rate": 1e-3,
                           "weight_decay": 0.0, "patience": 2, "validation_interval": 1,
                           "min_relative_improvement": 1e-3, "unroll_steps": 1,
                           "benchmark_samples_per_split": 1, "seed": 3, "dtype": "float64"},
        device="cpu",
        benchmark_settings={"neural_steps": 1, "residual_tolerance": 1e-8,
                            "max_iterations": 80, "restart": 20},
    )
    assert report.epochs_completed >= 1
    assert np.isfinite(report.best_validation_residual_loss)

    accelerator = NeuralMaxwellAccelerator(network, residual_tolerance=1e-9,
                                            max_iterations=80, restart=20, neural_steps=1)
    model = UnifiedNeuralElectroThermalModel(background, accelerator, default_geometry=geometry())
    model_path = tmp_path / "unified_model.npz"
    model.save(model_path)
    with np.load(model_path, allow_pickle=False) as data:
        assert "em_basis" not in data.files
    loaded = UnifiedNeuralElectroThermalModel.load(model_path, device="cpu")

    assert loaded.thermal_rank == model.thermal_rank
    assert np.allclose(loaded.background.thermal_basis, model.background.thermal_basis)
    result = loaded.predict(0.0, initial_state=np.zeros(loaded.thermal_rank),
                            geometry=geometry(0.001), operating=[1.0, 0.0], max_step=1.0)
    assert result.steps == 0
    assert np.all(np.isfinite(result.state))
    assert np.all(np.isfinite(result.heat_source))
    assert np.isfinite(result.maximum_temperature)
    assert max(result.maxwell_final_residual) <= 1e-9
