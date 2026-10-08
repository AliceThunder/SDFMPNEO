import pytest

pytest.importorskip("torch")

from sdfmpneo_vnext.tensor_training_cli import _parser, _teacher_options


def test_tensor_training_cli_keeps_magnetic_spatial_and_energy_quadrature_independent():
    args = _parser().parse_args(
        [
            "bundle-out",
            "--with-spatial",
            "--magnetic-volume-axial-order",
            "9",
            "--magnetic-volume-radial-order",
            "7",
            "--magnetic-volume-azimuthal-order",
            "26",
            "--package-volume-axial-order",
            "5",
            "--package-volume-radial-order",
            "3",
            "--package-volume-azimuthal-order",
            "14",
            "--background-radial-order",
            "8",
            "--background-angular-order",
            "28",
            "--energy-volume-axial-order",
            "11",
            "--energy-volume-radial-order",
            "8",
            "--energy-volume-azimuthal-order",
            "30",
            "--energy-background-radial-order",
            "15",
            "--energy-background-angular-order",
            "52",
        ]
    )
    options = _teacher_options(args)

    assert options["magnetic_volume_axial_order"] == 9
    assert options["magnetic_volume_radial_order"] == 7
    assert options["magnetic_volume_azimuthal_order"] == 26
    assert options["package_volume_axial_order"] == 5
    assert options["package_volume_radial_order"] == 3
    assert options["package_volume_azimuthal_order"] == 14
    assert options["background_radial_order"] == 8
    assert options["background_angular_order"] == 28
    assert options["energy_volume_axial_order"] == 11
    assert options["energy_volume_radial_order"] == 8
    assert options["energy_volume_azimuthal_order"] == 30
    assert options["energy_background_radial_order"] == 15
    assert options["energy_background_angular_order"] == 52
