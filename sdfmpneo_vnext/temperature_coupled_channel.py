from __future__ import annotations

from .channel_thermal import (
    ChannelResolvedCurrentEnvelope,
    ChannelResolvedVoltageEnvelope,
)
from .scene import PackageObject, Scene


class _TemperatureResolvedSceneMixin:
    """Map thermal observations back into conductor and dielectric materials."""

    def _configure_material_temperature_laws(
        self,
        *,
        background_temperature_law=None,
        background_temperature_index=None,
        package_temperature_laws=None,
        package_temperature_indices=None,
    ):
        n_observations = int(self.thermal_model.decoder.shape[0])
        n_packages = len(self.scene.packages)

        if package_temperature_laws is None:
            package_laws = (None,) * n_packages
        else:
            package_laws = tuple(package_temperature_laws)
        if package_temperature_indices is None:
            package_indices = (None,) * n_packages
        else:
            package_indices = tuple(package_temperature_indices)
        if len(package_laws) != n_packages or len(package_indices) != n_packages:
            raise ValueError(
                "package temperature laws/indices must contain one entry per package"
            )

        def validate_pair(law, index, name):
            if law is None:
                if index is not None:
                    raise ValueError(f"{name} temperature index requires a law")
                return None
            if not hasattr(law, "at_temperature"):
                raise TypeError(f"{name} temperature law must expose at_temperature")
            if index is None:
                raise ValueError(f"{name} temperature law requires an observation index")
            try:
                value = int(index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} temperature index must be an integer") from exc
            if value != index or value < 0 or value >= n_observations:
                raise ValueError(
                    f"{name} temperature index is outside the thermal observation range"
                )
            return value

        background_index = validate_pair(
            background_temperature_law,
            background_temperature_index,
            "background",
        )
        resolved_package_indices = tuple(
            validate_pair(law, index, f"package {package_index}")
            for package_index, (law, index) in enumerate(
                zip(package_laws, package_indices)
            )
        )

        self.background_temperature_law = background_temperature_law
        self.background_temperature_index = background_index
        self.package_temperature_laws = package_laws
        self.package_temperature_indices = resolved_package_indices

    def _scene_at_state(self, state):
        warm_scene, temperatures = super()._scene_at_state(state)

        medium = warm_scene.medium
        if self.background_temperature_law is not None:
            medium = self.background_temperature_law.at_temperature(
                float(temperatures[self.background_temperature_index])
            )

        packages = []
        for package_index, package in enumerate(self.scene.packages):
            law = self.package_temperature_laws[package_index]
            if law is None:
                material = package.material
            else:
                material = law.at_temperature(
                    float(
                        temperatures[
                            self.package_temperature_indices[package_index]
                        ]
                    )
                )
            packages.append(
                PackageObject(
                    package.geometry,
                    material,
                    package.name,
                )
            )

        return (
            Scene(
                warm_scene.coils,
                medium,
                tuple(packages),
            ),
            temperatures,
        )


class TemperatureResolvedChannelCurrentEnvelope(
    _TemperatureResolvedSceneMixin,
    ChannelResolvedCurrentEnvelope,
):
    """Channel-resolved current envelope with explicit dielectric temperature laws."""

    def __init__(
        self,
        *args,
        background_temperature_law=None,
        background_temperature_index=None,
        package_temperature_laws=None,
        package_temperature_indices=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._configure_material_temperature_laws(
            background_temperature_law=background_temperature_law,
            background_temperature_index=background_temperature_index,
            package_temperature_laws=package_temperature_laws,
            package_temperature_indices=package_temperature_indices,
        )


class TemperatureResolvedChannelVoltageEnvelope(
    _TemperatureResolvedSceneMixin,
    ChannelResolvedVoltageEnvelope,
):
    """Channel-resolved voltage envelope with explicit dielectric temperature laws."""

    def __init__(
        self,
        *args,
        background_temperature_law=None,
        background_temperature_index=None,
        package_temperature_laws=None,
        package_temperature_indices=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._configure_material_temperature_laws(
            background_temperature_law=background_temperature_law,
            background_temperature_index=background_temperature_index,
            package_temperature_laws=package_temperature_laws,
            package_temperature_indices=package_temperature_indices,
        )
