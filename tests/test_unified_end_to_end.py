import numpy as np
import pytest

from sdfmpneo.unified_model import UnifiedNeuralElectroThermalModel
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo.unified_tensor_surrogate import generate_tensor_dataset
from sdfmpneo.unified_tensor_training import train_matrix_tensor_surrogate
from sdfmpneo.unified_thermal import build_geometry_aware_thermal_library


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
    return OpenBoundaryBackground(
        axis, axis, axis, frequency_hz=100000.0,
        materials=MATERIALS, coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"), seawater_material="seawater",
        ambient_temperature=293.15,
    )


def test_tensor_rom_training_save_load_and_predict_without_online_maxwell(tmp_path):
    pytest.importorskip("torch")
    background = small_background()
    geometries = [geometry(0.0015 * np.sin(i)) for i in range(9)]
    library, thermal_report = build_geometry_aware_thermal_library(
        background,
        geometry(),
        geometries[:2],
        validation_geometries=geometries[2:3],
        target_relative_error=0.99,
        time_scales=(0.1, 1.0),
    )
    assert thermal_report.converged
    background.set_thermal_library(library)
    assert background.thermal_rank == library.rank > 0

    dataset = generate_tensor_dataset(background, geometries[3:], seed=9)
    assert len(dataset.indices("train")) >= 3
    assert len(dataset.indices("validation")) == 1
    assert len(dataset.indices("test")) == 1
    assert len(dataset.indices("audit")) == 1
    assert dataset.phi_min.shape == dataset.phi_max.shape == (6, library.rank)
    assert dataset.audit["maximum_linear_relative_residual"] <= 1e-8
    assert dataset.audit["maximum_reciprocity_relative_error"] <= 1e-8
    assert dataset.audit["maximum_open_boundary_power_balance_relative_error"] <= 1e-7
    assert dataset.audit["minimum_d_vol_eigenvalue"] >= -1e-9
    assert dataset.audit["minimum_physical_outward_eigenvalue"] >= -1e-9
    assert dataset.audit["independent_outward_power_available"] == 1.0
    assert dataset.audit["maximum_relative_loewner_violation"] <= 1e-8

    surrogate, report = train_matrix_tensor_surrogate(
        dataset,
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
            "pod_relative_tail_tolerance": 0.5,
            "seed": 3,
            "dtype": "float64",
        },
        device="cpu",
    )
    assert report.epochs_completed >= 1
    assert report.pod_rank >= 1
    assert np.isfinite(report.best_validation_loss)

    model = UnifiedNeuralElectroThermalModel(background, surrogate, default_geometry=geometry())
    query_geometry = geometry(0.001)
    context = model.geometry_context(query_geometry)
    reference_context = model.geometry_context(geometry())
    assert context.thermal_basis.shape == reference_context.thermal_basis.shape
    assert not np.allclose(context.thermal_basis, reference_context.thermal_basis)

    full_initial = np.linspace(0.0, 1.0, background.n_cells)
    projected = model.project_initial_temperature(full_initial, query_geometry)
    phi = context.thermal_basis
    projection_residual = phi.T @ (
        context.thermal_mass_full @ (full_initial - phi @ projected)
    )
    assert np.linalg.norm(projection_residual) <= 1e-10 * max(
        1.0, np.linalg.norm(context.thermal_mass_full @ full_initial)
    )

    result = model.predict(
        0.0,
        initial_state=np.zeros(model.thermal_rank),
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
        initial_guess=np.zeros(model.thermal_rank),
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
        assert "thermal_background_modes" in data.files
        assert "thermal_local_modes_0" in data.files
        assert "thermal_basis" not in data.files
        assert "em_basis" not in data.files
        assert not any(name.startswith("maxwell") for name in data.files)
    loaded = UnifiedNeuralElectroThermalModel.load(model_path, device="cpu")
    loaded_context = loaded.geometry_context(query_geometry)
    assert np.allclose(loaded_context.thermal_basis, context.thermal_basis)
    loaded_result = loaded.predict(
        0.0,
        initial_state=np.zeros(loaded.thermal_rank),
        geometry=query_geometry,
        operating=[1.0, 0.0],
        max_step=1.0,
    )
    assert loaded.thermal_rank == model.thermal_rank
    assert loaded.surrogate.pod_rank == model.surrogate.pod_rank
    assert np.allclose(loaded_result.impedance, result.impedance)
    assert np.allclose(loaded_result.heat_source, result.heat_source)
