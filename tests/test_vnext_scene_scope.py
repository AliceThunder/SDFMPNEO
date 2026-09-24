import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMixedConductorTeacher,
    HomogeneousMedium,
    MQSConfig,
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



def test_scene_represents_lossy_background_but_mvp_solver_rejects_it():
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
    scene = Scene(
        (coil,),
        HomogeneousMedium(
            relative_permittivity=2.5,
            relative_permeability=1.0,
            conductivity=0.01,
        ),
    )
    assert scene.medium.conductivity == 0.01
    with pytest.raises(
        ValueError,
        match="lossless homogeneous background",
    ):
        DenseMixedConductorTeacher(
            scene,
            20_000.0,
            MQSConfig(
                segments_per_turn=4,
                min_segments=4,
                section_degree=0,
                radial_order=2,
                angular_order=8,
                line_order=1,
            ),
        )


def test_scene_accepts_lossless_nonvacuum_homogeneous_background():
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
    scene = Scene(
        (coil,),
        HomogeneousMedium(
            relative_permittivity=3.2,
            relative_permeability=1.4,
            conductivity=0.0,
        ),
    )
    assert scene.medium.relative_permittivity == 3.2
    assert scene.medium.relative_permeability == 1.4
