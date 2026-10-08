from __future__ import annotations

"""Public vNext system API with optional temperature-resolved dielectric coupling."""

from . import _system_core as _core
from ._system_core import *  # noqa: F401,F403
from .temperature_coupled_channel import (
    TemperatureResolvedChannelCurrentEnvelope,
    TemperatureResolvedChannelVoltageEnvelope,
)


def _temperature_coupling_requested(options) -> bool:
    if options.get("background_temperature_law") is not None:
        return True
    if options.get("background_temperature_index") is not None:
        return True
    laws = options.get("package_temperature_laws")
    if laws is not None and any(law is not None for law in tuple(laws)):
        return True
    indices = options.get("package_temperature_indices")
    if indices is not None and any(index is not None for index in tuple(indices)):
        return True
    return False


class MeshfreeVNextSystem(_core.MeshfreeVNextSystem):
    """vNext runtime with opt-in dielectric material/temperature feedback."""

    def reference_channel_current_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().reference_channel_current_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        return TemperatureResolvedChannelCurrentEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self._reference_artifact_for_scene(scene),
            **options,
        )

    def reference_channel_voltage_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().reference_channel_voltage_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        return TemperatureResolvedChannelVoltageEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self._reference_artifact_for_scene(scene),
            **options,
        )

    def fast_current_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().fast_current_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        self._require_fast_port_artifact(scene, frequency_hz)
        return TemperatureResolvedChannelCurrentEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self.port_artifact,
            **options,
        )

    def fast_voltage_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().fast_voltage_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        self._require_fast_port_artifact(scene, frequency_hz)
        return TemperatureResolvedChannelVoltageEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self.port_artifact,
            **options,
        )

    def reference_current_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().reference_current_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        return TemperatureResolvedChannelCurrentEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self._reference_artifact_for_scene(scene),
            **options,
        )

    def reference_voltage_envelope(
        self,
        scene,
        frequency_hz,
        thermal_model,
        **options,
    ):
        if not _temperature_coupling_requested(options):
            return super().reference_voltage_envelope(
                scene,
                frequency_hz,
                thermal_model,
                **options,
            )
        return TemperatureResolvedChannelVoltageEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            self._reference_artifact_for_scene(scene),
            **options,
        )
