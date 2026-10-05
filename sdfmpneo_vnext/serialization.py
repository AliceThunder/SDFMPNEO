from __future__ import annotations

from hashlib import sha256
import json
import numpy as np

from .geometry import RigidPose, SuperellipseSpiral
from .package_geometry import SuperquadricPackageGeometry
from .scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    TensorElectricMaterial,
    DebyeMaterial,
    MultiDebyeMaterial,
    TabulatedMaterial,
    PackageObject,
    Scene,
)


def _thermal_payload(material):
    return {
        "thermal_conductivity": getattr(material, "thermal_conductivity", None),
        "thermal_conductivity_tensor": (
            None
            if getattr(material, "thermal_conductivity_tensor", None) is None
            else np.asarray(
                material.thermal_conductivity_tensor,
                dtype=float,
            ).tolist()
        ),
        "density": getattr(material, "density", None),
        "heat_capacity": getattr(material, "heat_capacity", None),
    }


def _thermal_from_dict(data, *, prefix: str):
    scalar = data.get("thermal_conductivity")
    tensor = data.get("thermal_conductivity_tensor")
    density = data.get("density")
    heat_capacity = data.get("heat_capacity")
    if scalar is not None and tensor is not None:
        raise ValueError(
            f"{prefix} thermal_conductivity and thermal_conductivity_tensor "
            "are mutually exclusive"
        )
    has_conductivity = scalar is not None or tensor is not None
    has_thermal = (
        has_conductivity
        or density is not None
        or heat_capacity is not None
    )
    if has_thermal and (
        not has_conductivity
        or density is None
        or heat_capacity is None
    ):
        raise ValueError(
            f"{prefix} thermal conductivity, density, and heat_capacity must "
            "be supplied together"
        )
    return {
        "thermal_conductivity": (
            None if scalar is None else float(scalar)
        ),
        "thermal_conductivity_tensor": (
            None
            if tensor is None
            else np.asarray(tensor, dtype=float)
        ),
        "density": None if density is None else float(density),
        "heat_capacity": (
            None if heat_capacity is None else float(heat_capacity)
        ),
    }


def _tensor_electric_to_dict(material):
    return {
        "model": "tensor_electric",
        "relative_permittivity_tensor": np.asarray(
            material.relative_permittivity_tensor,
            dtype=float,
        ).tolist(),
        "conductivity_tensor": np.asarray(
            material.conductivity_tensor,
            dtype=float,
        ).tolist(),
        "relative_permeability": float(material.relative_permeability),
        **_thermal_payload(material),
    }


def _frequency_material_to_dict(material):
    if isinstance(material, TabulatedMaterial):
        return {
            "model": "tabulated",
            "frequencies_hz": list(material.frequencies_hz),
            "relative_permittivity_real": list(
                material.relative_permittivity_real
            ),
            "loss_conductivity_values": list(
                material.loss_conductivity_values
            ),
            "relative_permeability": material.relative_permeability,
            "conductivity": material.conductivity,
            **_thermal_payload(material),
        }
    if isinstance(material, MultiDebyeMaterial):
        return {
            "model": "multi_debye",
            "relative_permittivity_infinite": (
                material.relative_permittivity_infinite
            ),
            "relaxation_strengths": list(material.relaxation_strengths),
            "relaxation_times": list(material.relaxation_times),
            "relative_permeability": material.relative_permeability,
            "conductivity": material.conductivity,
            **_thermal_payload(material),
        }
    if isinstance(material, DebyeMaterial):
        return {
            "model": "debye",
            "relative_permittivity_static": (
                material.relative_permittivity_static
            ),
            "relative_permittivity_infinite": (
                material.relative_permittivity_infinite
            ),
            "relaxation_time": material.relaxation_time,
            "relative_permeability": material.relative_permeability,
            "conductivity": material.conductivity,
            **_thermal_payload(material),
        }
    raise TypeError("unsupported frequency-response material type")


def _medium_to_dict(medium):
    if isinstance(medium, TensorElectricMaterial):
        return _tensor_electric_to_dict(medium)
    if isinstance(
        medium,
        (TabulatedMaterial, MultiDebyeMaterial, DebyeMaterial),
    ):
        return _frequency_material_to_dict(medium)
    if isinstance(medium, IsotropicMaterial):
        return {
            "model": "constant",
            "relative_permittivity": medium.relative_permittivity,
            "relative_permeability": medium.relative_permeability,
            "conductivity": medium.conductivity,
            **_thermal_payload(medium),
        }
    if isinstance(medium, HomogeneousMedium):
        return {
            "model": "constant",
            "relative_permittivity": medium.relative_permittivity,
            "relative_permeability": medium.relative_permeability,
            "conductivity": medium.conductivity,
        }
    raise TypeError("unsupported scene background medium type")


def _material_from_dict(data, *, package: bool):
    if not isinstance(data, dict):
        raise TypeError(
            "package material must be a dictionary"
            if package
            else "scene.medium must be a dictionary"
        )
    prefix = "package" if package else "scene medium"
    thermal = _thermal_from_dict(data, prefix=prefix)
    model = str(data.get("model", "constant")).lower()

    if model == "tensor_electric":
        if "relative_permittivity_tensor" not in data:
            raise ValueError(
                f"tensor-electric {prefix} is missing "
                "relative_permittivity_tensor"
            )
        return TensorElectricMaterial(
            relative_permittivity_tensor=np.asarray(
                data["relative_permittivity_tensor"],
                dtype=float,
            ),
            conductivity_tensor=np.asarray(
                data.get("conductivity_tensor", np.zeros((3, 3))),
                dtype=float,
            ),
            relative_permeability=float(
                data.get("relative_permeability", 1.0)
            ),
            **thermal,
        )

    if model == "constant":
        epsilon = float(data.get("relative_permittivity", 1.0))
        permeability = float(data.get("relative_permeability", 1.0))
        conductivity = float(data.get("conductivity", 0.0))
        has_thermal = any(value is not None for value in thermal.values())
        if package or has_thermal:
            return IsotropicMaterial(
                relative_permittivity=epsilon,
                relative_permeability=permeability,
                conductivity=conductivity,
                **thermal,
            )
        return HomogeneousMedium(
            epsilon,
            permeability,
            conductivity,
        )

    if model == "debye":
        required = (
            "relative_permittivity_static",
            "relative_permittivity_infinite",
            "relaxation_time",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"Debye {prefix} is missing: " + ", ".join(missing)
            )
        return DebyeMaterial(
            relative_permittivity_static=float(
                data["relative_permittivity_static"]
            ),
            relative_permittivity_infinite=float(
                data["relative_permittivity_infinite"]
            ),
            relaxation_time=float(data["relaxation_time"]),
            relative_permeability=float(
                data.get("relative_permeability", 1.0)
            ),
            conductivity=float(data.get("conductivity", 0.0)),
            **thermal,
        )

    if model == "multi_debye":
        required = (
            "relative_permittivity_infinite",
            "relaxation_strengths",
            "relaxation_times",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"multi-Debye {prefix} is missing: " + ", ".join(missing)
            )
        return MultiDebyeMaterial(
            relative_permittivity_infinite=float(
                data["relative_permittivity_infinite"]
            ),
            relaxation_strengths=tuple(
                float(value) for value in data["relaxation_strengths"]
            ),
            relaxation_times=tuple(
                float(value) for value in data["relaxation_times"]
            ),
            relative_permeability=float(
                data.get("relative_permeability", 1.0)
            ),
            conductivity=float(data.get("conductivity", 0.0)),
            **thermal,
        )

    if model == "tabulated":
        required = (
            "frequencies_hz",
            "relative_permittivity_real",
            "loss_conductivity_values",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                f"tabulated {prefix} is missing: " + ", ".join(missing)
            )
        return TabulatedMaterial(
            frequencies_hz=tuple(
                float(value) for value in data["frequencies_hz"]
            ),
            relative_permittivity_real=tuple(
                float(value) for value in data["relative_permittivity_real"]
            ),
            loss_conductivity_values=tuple(
                float(value) for value in data["loss_conductivity_values"]
            ),
            relative_permeability=float(
                data.get("relative_permeability", 1.0)
            ),
            conductivity=float(data.get("conductivity", 0.0)),
            **thermal,
        )
    raise ValueError(f"unsupported {prefix} material model: {model}")


def _medium_from_dict(data):
    return _material_from_dict(data, package=False)


def _package_material_to_dict(material):
    if isinstance(material, TensorElectricMaterial):
        return _tensor_electric_to_dict(material)
    if isinstance(
        material,
        (TabulatedMaterial, MultiDebyeMaterial, DebyeMaterial),
    ):
        return _frequency_material_to_dict(material)
    if isinstance(material, IsotropicMaterial):
        # Preserve the historical package payload: omitted model means constant.
        return {
            "relative_permittivity": material.relative_permittivity,
            "relative_permeability": material.relative_permeability,
            "conductivity": material.conductivity,
            **_thermal_payload(material),
        }
    raise TypeError("unsupported package material type")


def _package_material_from_dict(data):
    return _material_from_dict(data, package=True)


def scene_to_dict(scene: Scene):
    payload = {
        "coils": [
            {
                "name": coil.name,
                "geometry": {
                    "outer_a": coil.geometry.outer_a,
                    "outer_b": coil.geometry.outer_b,
                    "turns": coil.geometry.turns,
                    "pitch_a": coil.geometry.pitch_a,
                    "pitch_b": coil.geometry.pitch_b,
                    "exponent": coil.geometry.exponent,
                    "conductor_width": coil.geometry.conductor_width,
                    "conductor_thickness": coil.geometry.conductor_thickness,
                    "cross_section_exponent": (
                        coil.geometry.cross_section_exponent
                    ),
                    "rotation": coil.geometry.pose.rotation.tolist(),
                    "translation": coil.geometry.pose.translation.tolist(),
                },
                "material": {
                    "conductivity": coil.material.conductivity,
                    "relative_permeability": (
                        coil.material.relative_permeability
                    ),
                    "resistance_temperature_coefficient": (
                        coil.material.resistance_temperature_coefficient
                    ),
                    "reference_temperature": (
                        coil.material.reference_temperature
                    ),
                },
            }
            for coil in scene.coils
        ],
        "medium": _medium_to_dict(scene.medium),
    }
    if scene.packages:
        payload["packages"] = [
            {
                "name": package.name,
                "geometry": {
                    "half_extents": package.geometry.half_extents.tolist(),
                    "exponent_xy": package.geometry.exponent_xy,
                    "exponent_z": package.geometry.exponent_z,
                    "rotation": package.geometry.pose.rotation.tolist(),
                    "translation": package.geometry.pose.translation.tolist(),
                },
                "material": _package_material_to_dict(package.material),
            }
            for package in scene.packages
        ]
    return payload


def scene_from_dict(data) -> Scene:
    if not isinstance(data, dict):
        raise TypeError("scene must be a dictionary")
    raw_coils = data.get("coils")
    if not isinstance(raw_coils, (list, tuple)) or not raw_coils:
        raise ValueError("scene.coils must be a non-empty list")

    coils = []
    for index, item in enumerate(raw_coils):
        if not isinstance(item, dict):
            raise TypeError(f"scene.coils[{index}] must be a dictionary")
        g = item.get("geometry", {})
        m = item.get("material", {})
        if not isinstance(g, dict) or not isinstance(m, dict):
            raise TypeError("geometry and material must be dictionaries")
        pose = RigidPose(
            np.asarray(g.get("rotation", np.eye(3)), dtype=float),
            np.asarray(g.get("translation", np.zeros(3)), dtype=float),
        )
        pitch_default = float(g["pitch"]) if "pitch" in g else None
        if "pitch_a" not in g and pitch_default is None:
            raise ValueError(
                f"scene.coils[{index}].geometry requires pitch or pitch_a"
            )
        if "pitch_b" not in g and pitch_default is None:
            raise ValueError(
                f"scene.coils[{index}].geometry requires pitch or pitch_b"
            )
        required_geometry = (
            "outer_a",
            "outer_b",
            "turns",
            "conductor_width",
            "conductor_thickness",
        )
        missing_geometry = [
            key for key in required_geometry if key not in g
        ]
        if missing_geometry:
            raise ValueError(
                f"scene.coils[{index}].geometry is missing: "
                + ", ".join(missing_geometry)
            )
        if "conductivity" not in m:
            raise ValueError(
                f"scene.coils[{index}].material requires conductivity"
            )
        exponent = float(g.get("exponent", 2.0))
        geometry = SuperellipseSpiral(
            float(g["outer_a"]),
            float(g["outer_b"]),
            float(g["turns"]),
            float(g.get("pitch_a", pitch_default)),
            float(g.get("pitch_b", pitch_default)),
            exponent=exponent,
            conductor_width=float(g["conductor_width"]),
            conductor_thickness=float(g["conductor_thickness"]),
            cross_section_exponent=float(
                g.get("cross_section_exponent", exponent)
            ),
            pose=pose,
        )
        material = ConductorMaterial(
            conductivity=float(m["conductivity"]),
            relative_permeability=float(
                m.get("relative_permeability", 1.0)
            ),
            resistance_temperature_coefficient=float(
                m.get("resistance_temperature_coefficient", 0.0)
            ),
            reference_temperature=float(
                m.get("reference_temperature", 293.15)
            ),
        )
        coils.append(
            CoilObject(
                geometry,
                material,
                str(item.get("name", f"coil_{index}")),
            )
        )

    raw_packages = data.get("packages", ())
    if not isinstance(raw_packages, (list, tuple)):
        raise TypeError("scene.packages must be a list")
    packages = []
    for index, item in enumerate(raw_packages):
        if not isinstance(item, dict):
            raise TypeError(f"scene.packages[{index}] must be a dictionary")
        g = item.get("geometry", {})
        m = item.get("material", {})
        if not isinstance(g, dict) or not isinstance(m, dict):
            raise TypeError("package geometry and material must be dictionaries")
        if "half_extents" not in g:
            raise ValueError(
                f"scene.packages[{index}].geometry requires half_extents"
            )
        pose = RigidPose(
            np.asarray(g.get("rotation", np.eye(3)), dtype=float),
            np.asarray(g.get("translation", np.zeros(3)), dtype=float),
        )
        geometry = SuperquadricPackageGeometry(
            np.asarray(g["half_extents"], dtype=float),
            float(g.get("exponent_xy", 4.0)),
            float(g.get("exponent_z", 4.0)),
            pose,
        )
        packages.append(
            PackageObject(
                geometry,
                _package_material_from_dict(m),
                str(item.get("name", f"package_{index}")),
            )
        )

    medium = _medium_from_dict(data.get("medium", {}))
    return Scene(tuple(coils), medium, tuple(packages))


def canonical_json(data) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(data) -> str:
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()
