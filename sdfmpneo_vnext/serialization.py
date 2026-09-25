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
    DebyeMaterial,
    PackageObject,
    Scene,
)


def _package_material_to_dict(
    material,
):
    if isinstance(
        material,
        DebyeMaterial,
    ):
        return {
            "model": "debye",
            "relative_permittivity_static": (
                material.relative_permittivity_static
            ),
            "relative_permittivity_infinite": (
                material.relative_permittivity_infinite
            ),
            "relaxation_time": (
                material.relaxation_time
            ),
            "relative_permeability": (
                material.relative_permeability
            ),
            "conductivity": (
                material.conductivity
            ),
            "thermal_conductivity": (
                material.thermal_conductivity
            ),
            "density": material.density,
            "heat_capacity": (
                material.heat_capacity
            ),
        }
    if isinstance(
        material,
        IsotropicMaterial,
    ):
        return {
            "relative_permittivity": (
                material.relative_permittivity
            ),
            "relative_permeability": (
                material.relative_permeability
            ),
            "conductivity": (
                material.conductivity
            ),
            "thermal_conductivity": (
                material.thermal_conductivity
            ),
            "density": material.density,
            "heat_capacity": (
                material.heat_capacity
            ),
        }
    raise TypeError(
        "unsupported package material type"
    )


def _package_material_from_dict(
    data,
):
    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "package material must be a dictionary"
        )
    thermal_conductivity = (
        data.get(
            "thermal_conductivity"
        )
    )
    density = data.get(
        "density"
    )
    heat_capacity = data.get(
        "heat_capacity"
    )
    thermal = (
        None
        if thermal_conductivity
        is None
        else float(
            thermal_conductivity
        )
    )
    rho = (
        None
        if density is None
        else float(
            density
        )
    )
    capacity = (
        None
        if heat_capacity is None
        else float(
            heat_capacity
        )
    )
    model = str(
        data.get(
            "model",
            "constant",
        )
    ).lower()
    if model == "constant":
        return IsotropicMaterial(
            float(
                data.get(
                    "relative_permittivity",
                    1.0,
                )
            ),
            float(
                data.get(
                    "relative_permeability",
                    1.0,
                )
            ),
            float(
                data.get(
                    "conductivity",
                    0.0,
                )
            ),
            thermal,
            rho,
            capacity,
        )
    if model == "debye":
        required = (
            "relative_permittivity_static",
            "relative_permittivity_infinite",
            "relaxation_time",
        )
        missing = [
            key
            for key in required
            if key not in data
        ]
        if missing:
            raise ValueError(
                "Debye package material is missing: "
                + ", ".join(
                    missing
                )
            )
        return DebyeMaterial(
            float(
                data[
                    "relative_permittivity_static"
                ]
            ),
            float(
                data[
                    "relative_permittivity_infinite"
                ]
            ),
            float(
                data[
                    "relaxation_time"
                ]
            ),
            float(
                data.get(
                    "relative_permeability",
                    1.0,
                )
            ),
            float(
                data.get(
                    "conductivity",
                    0.0,
                )
            ),
            thermal,
            rho,
            capacity,
        )
    raise ValueError(
        f"unsupported package material model: {model}"
    )


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
                    "cross_section_exponent": coil.geometry.cross_section_exponent,
                    "rotation": coil.geometry.pose.rotation.tolist(),
                    "translation": coil.geometry.pose.translation.tolist(),
                },
                "material": {
                    "conductivity": coil.material.conductivity,
                    "relative_permeability": coil.material.relative_permeability,
                    "resistance_temperature_coefficient": (
                        coil.material.resistance_temperature_coefficient
                    ),
                    "reference_temperature": coil.material.reference_temperature,
                },
            }
            for coil in scene.coils
        ],
        "medium": {
            "relative_permittivity": scene.medium.relative_permittivity,
            "relative_permeability": scene.medium.relative_permeability,
            "conductivity": scene.medium.conductivity,
        },
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
                "material": _package_material_to_dict(
                    package.material
                ),
            }
            for package in scene.packages
        ]
    return payload


def scene_from_dict(data) -> Scene:
    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            "scene must be a dictionary"
        )
    raw_coils = data.get(
        "coils"
    )
    if not isinstance(
        raw_coils,
        (list, tuple),
    ) or not raw_coils:
        raise ValueError(
            "scene.coils must be a non-empty list"
        )

    coils = []
    for index, item in enumerate(
        raw_coils
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"scene.coils[{index}] must be a dictionary"
            )
        g = item.get(
            "geometry",
            {}
        )
        m = item.get(
            "material",
            {}
        )
        if not isinstance(
            g,
            dict,
        ) or not isinstance(
            m,
            dict,
        ):
            raise TypeError(
                "geometry and material must be dictionaries"
            )

        rotation = np.asarray(
            g.get(
                "rotation",
                np.eye(3),
            ),
            dtype=float,
        )
        translation = np.asarray(
            g.get(
                "translation",
                np.zeros(3),
            ),
            dtype=float,
        )
        pose = RigidPose(
            rotation,
            translation,
        )

        if "pitch" in g:
            pitch_default = float(
                g["pitch"]
            )
        else:
            pitch_default = None
        if (
            "pitch_a" not in g
            and pitch_default is None
        ):
            raise ValueError(
                f"scene.coils[{index}].geometry requires pitch or pitch_a"
            )
        if (
            "pitch_b" not in g
            and pitch_default is None
        ):
            raise ValueError(
                f"scene.coils[{index}].geometry requires pitch or pitch_b"
            )
        pitch_a = float(
            g.get(
                "pitch_a",
                pitch_default,
            )
        )
        pitch_b = float(
            g.get(
                "pitch_b",
                pitch_default,
            )
        )
        exponent = float(
            g.get(
                "exponent",
                2.0,
            )
        )
        cross_section_exponent = float(
            g.get(
                "cross_section_exponent",
                exponent,
            )
        )

        required_geometry = (
            "outer_a",
            "outer_b",
            "turns",
            "conductor_width",
            "conductor_thickness",
        )
        missing_geometry = [
            key
            for key in required_geometry
            if key not in g
        ]
        if missing_geometry:
            raise ValueError(
                "scene.coils["
                + str(index)
                + "].geometry is missing: "
                + ", ".join(
                    missing_geometry
                )
            )
        if "conductivity" not in m:
            raise ValueError(
                f"scene.coils[{index}].material requires conductivity"
            )

        geometry = SuperellipseSpiral(
            float(
                g["outer_a"]
            ),
            float(
                g["outer_b"]
            ),
            float(
                g["turns"]
            ),
            pitch_a,
            pitch_b,
            exponent=exponent,
            conductor_width=float(
                g[
                    "conductor_width"
                ]
            ),
            conductor_thickness=float(
                g[
                    "conductor_thickness"
                ]
            ),
            cross_section_exponent=(
                cross_section_exponent
            ),
            pose=pose,
        )
        material = ConductorMaterial(
            float(
                m["conductivity"]
            ),
            float(
                m.get(
                    "relative_permeability",
                    1.0,
                )
            ),
            float(
                m.get(
                    "resistance_temperature_coefficient",
                    0.0,
                )
            ),
            float(
                m.get(
                    "reference_temperature",
                    293.15,
                )
            ),
        )
        coils.append(
            CoilObject(
                geometry,
                material,
                str(
                    item.get(
                        "name",
                        f"coil_{index}",
                    )
                ),
            )
        )

    raw_packages = data.get(
        "packages",
        ()
    )
    if not isinstance(
        raw_packages,
        (list, tuple),
    ):
        raise TypeError(
            "scene.packages must be a list"
        )
    packages = []
    for index, item in enumerate(
        raw_packages
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise TypeError(
                f"scene.packages[{index}] must be a dictionary"
            )
        g = item.get(
            "geometry",
            {}
        )
        m = item.get(
            "material",
            {}
        )
        if not isinstance(
            g,
            dict,
        ) or not isinstance(
            m,
            dict,
        ):
            raise TypeError(
                "package geometry and material must be dictionaries"
            )
        if "half_extents" not in g:
            raise ValueError(
                f"scene.packages[{index}].geometry requires half_extents"
            )
        pose = RigidPose(
            np.asarray(
                g.get(
                    "rotation",
                    np.eye(3),
                ),
                dtype=float,
            ),
            np.asarray(
                g.get(
                    "translation",
                    np.zeros(3),
                ),
                dtype=float,
            ),
        )
        geometry = SuperquadricPackageGeometry(
            np.asarray(
                g["half_extents"],
                dtype=float,
            ),
            float(
                g.get(
                    "exponent_xy",
                    4.0,
                )
            ),
            float(
                g.get(
                    "exponent_z",
                    4.0,
                )
            ),
            pose,
        )
        material = (
            _package_material_from_dict(
                m
            )
        )
        packages.append(
            PackageObject(
                geometry,
                material,
                str(
                    item.get(
                        "name",
                        f"package_{index}",
                    )
                ),
            )
        )

    md = data.get(
        "medium",
        {}
    )
    if not isinstance(
        md,
        dict,
    ):
        raise TypeError(
            "scene.medium must be a dictionary"
        )
    return Scene(
        tuple(
            coils
        ),
        HomogeneousMedium(
            float(
                md.get(
                    "relative_permittivity",
                    1.0,
                )
            ),
            float(
                md.get(
                    "relative_permeability",
                    1.0,
                )
            ),
            float(
                md.get(
                    "conductivity",
                    0.0,
                )
            ),
        ),
        tuple(
            packages
        ),
    )


def canonical_json(data) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(data) -> str:
    return sha256(
        canonical_json(data).encode("utf-8")
    ).hexdigest()
