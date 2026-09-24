import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    Scene,
    SuperellipseSpiral,
)


class _UnsupportedMedium:
    pass


def test_scene_rejects_unsupported_electromagnetic_medium_kind():
    coil = CoilObject(
        SuperellipseSpiral(
            0.02,
            0.018,
            0.6,
            0.001,
            0.001,
        ),
        ConductorMaterial(
            5.8e7
        ),
    )
    with pytest.raises(
        TypeError,
        match="HomogeneousMedium",
    ):
        Scene(
            (coil,),
            _UnsupportedMedium(),
        )
