import importlib.util
from pathlib import Path

from sdfmpneo_vnext.tensor_sampling import TensorHybridSceneSamplerConfig
from sdfmpneo_vnext.em import MQSConfig
from sdfmpneo_vnext.workflow import _runtime_config
from sdfmpneo_vnext.workflow_cache import build_sampler_config, build_teacher_config


def _load_run_module():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_project_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_config_constructs_current_sampler_and_teacher_contracts():
    module = _load_run_module()
    config = module.CONFIG
    sampler = build_sampler_config(config["SAMPLER"])
    teacher = build_teacher_config(config["TEACHER"])

    assert isinstance(sampler, TensorHybridSceneSamplerConfig)
    assert isinstance(teacher, MQSConfig)
    assert sampler.base.dc_probability == 0.0
    assert config["TRUTH"]["include_spatial"]


def test_run_config_uses_higher_order_port_energy_truth_than_spatial_sampling():
    truth = _load_run_module().CONFIG["TRUTH"]

    assert truth["energy_volume_axial_order"] >= truth["package_volume_axial_order"]
    assert truth["energy_volume_radial_order"] >= truth["package_volume_radial_order"]
    assert truth["energy_volume_azimuthal_order"] >= truth["package_volume_azimuthal_order"]
    assert truth["energy_background_radial_order"] >= truth["background_radial_order"]
    assert truth["energy_background_angular_order"] >= truth["background_angular_order"]


def test_run_config_inherited_device_is_resolved_without_mutating_source():
    module = _load_run_module()
    source = module.CONFIG
    resolved = _runtime_config(source)

    assert source["PORT_TRAINING"]["device"] == "inherit"
    assert source["SPATIAL_TRAINING"]["device"] == "inherit"
    assert resolved["PORT_TRAINING"]["device"] == resolved["RUNTIME"]["device"]
    assert resolved["SPATIAL_TRAINING"]["device"] == resolved["RUNTIME"]["device"]
