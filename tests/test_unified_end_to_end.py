import numpy as np
import pytest

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_dataset import generate_operator_dataset
from sdfmpneo.unified_maxwell import NeuralMaxwellAccelerator
from sdfmpneo.unified_model import UnifiedNeuralElectroThermalModel
from sdfmpneo.unified_trainer import train_maxwell_accelerator


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
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.010]),
        "receiver": dict(coil, translation=[rx_x, 0.0, 0.010]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def small_background():
    axis = np.linspace(-0.05, 0.05, 5)
    return FixedMultiscaleBackground(
        axis,
        axis,
        axis,
        frequency_hz=100000.0,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        thermal_rank=3,
        ambient_temperature=293.15,
    )


def test_unified_operator_training_save_load_and_predict(tmp_path):
    pytest.importorskip("torch")
    background = small_background()
    basis = np.eye(background.n_edges, dtype=complex)
    geometries = [geometry(0.002 * np.sin(i)) for i in range(12)]
    rng = np.random.default_rng(5)
    states = rng.uniform(-1e-3, 1e-3, size=(12, 3))
    dataset = generate_operator_dataset(background, basis, geometries, states, seed=9)

    network, report = train_maxwell_accelerator(
        dataset,
        network_settings={"width": 16, "blocks": 1, "activation": "silu"},
        training_settings={
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "patience": 2,
            "validation_interval": 1,
            "seed": 3,
            "dtype": "float64",
        },
        device="cpu",
    )
    assert report.epochs_completed >= 1
    assert np.isfinite(report.best_validation_residual_loss)

    accelerator = NeuralMaxwellAccelerator(
        network,
        basis,
        residual_tolerance=1e-9,
        max_iterations=50,
    )
    model = UnifiedNeuralElectroThermalModel(
        background,
        accelerator,
        default_geometry=geometry(),
    )
    model_path = tmp_path / "unified_model.npz"
    model.save(model_path)
    loaded = UnifiedNeuralElectroThermalModel.load(model_path, device="cpu")

    assert np.array_equal(loaded.background.thermal_basis, model.background.thermal_basis)
    result = loaded.predict(
        0.0,
        initial_state=np.zeros(3),
        geometry=geometry(0.001),
        operating=[1.0, 0.0],
        max_step=1.0,
    )
    assert result.steps == 0
    assert np.all(np.isfinite(result.state))
    assert np.all(np.isfinite(result.heat_source))
    assert np.isfinite(result.maximum_temperature)
    assert max(result.maxwell_final_residual) <= 5e-9
