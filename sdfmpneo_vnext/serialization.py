from __future__ import annotations

from hashlib import sha256
import json
import numpy as np

from .geometry import RigidPose, SuperellipseSpiral
from .scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    Scene,
)


def scene_to_dict(scene: Scene):
    return {
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


def scene_from_dict(data) -> Scene:
    coils = []
    for item in data["coils"]:
        g = item["geometry"]
        m = item["material"]
        pose = RigidPose(
            np.asarray(g["rotation"], dtype=float),
            np.asarray(g["translation"], dtype=float),
        )
        geometry = SuperellipseSpiral(
            g["outer_a"],
            g["outer_b"],
            g["turns"],
            g["pitch_a"],
            g["pitch_b"],
            exponent=g["exponent"],
            conductor_width=g["conductor_width"],
            conductor_thickness=g["conductor_thickness"],
            cross_section_exponent=g["cross_section_exponent"],
            pose=pose,
        )
        material = ConductorMaterial(
            m["conductivity"],
            m["relative_permeability"],
            m["resistance_temperature_coefficient"],
            m["reference_temperature"],
        )
        coils.append(
            CoilObject(
                geometry,
                material,
                item.get("name", "coil"),
            )
        )
    md = data["medium"]
    return Scene(
        tuple(coils),
        HomogeneousMedium(
            md["relative_permittivity"],
            md["relative_permeability"],
            md["conductivity"],
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
