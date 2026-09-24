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
