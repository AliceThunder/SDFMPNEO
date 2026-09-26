from __future__ import annotations

from dataclasses import dataclass

from .certified import certify_mixed_ports
from .hybrid_certified import certify_dielectric_ports
from .em import MQSConfig
from .fast import (
    FastCurrentControlledEnvelope,
    FastVoltageControlledEnvelope,
)
from .channel_thermal import (
    ChannelResolvedCurrentEnvelope,
    ChannelResolvedVoltageEnvelope,
)
from .field import UniformLossFieldDecoder
from .reference import MixedReferenceArtifact
from .hybrid_dielectric import DielectricCoupledReferenceArtifact
from .scene import Scene
from .thermal_field import ContinuousThermalGreenArtifact


@dataclass(frozen=True)
class SystemCapabilities:
    fast_ports: bool
    fast_spatial: bool
    reference: bool
    certified: bool
    electrothermal: bool
    electromagnetic_formulation: str
    background_medium: str
    lossy_background_media: bool
    heterogeneous_media: bool
    retardation: bool
    arbitrary_se3_pose: bool
    superelliptic_conductors: bool
    package_geometry: bool
    package_dielectric_sie: bool
    package_em_coupling: bool
    continuous_spatial_loss: bool


def mvp_system_capabilities(
) -> SystemCapabilities:
    return SystemCapabilities(
        fast_ports=True,
        fast_spatial=True,
        reference=True,
        certified=True,
        electrothermal=True,
        electromagnetic_formulation=(
            "magnetoquasistatic_current_potential_charge"
        ),
        background_medium=(
            "homogeneous_isotropic_unbounded_lossless_or_lossy_ac"
        ),
        lossy_background_media=True,
        heterogeneous_media=False,
        retardation=False,
        arbitrary_se3_pose=True,
        superelliptic_conductors=True,
        package_geometry=True,
        package_dielectric_sie=True,
        package_em_coupling=True,
        continuous_spatial_loss=True,
    )


class MeshfreeVNextSystem:
    """Unified vNext runtime over one immutable Scene contract."""

    def __init__(
        self,
        port_artifact,
        *,
        spatial_artifact=None,
        reference_config: MQSConfig | None = None,
        dielectric_surface_vertical_order: int = 16,
        dielectric_surface_azimuthal_order: int = 32,
    ):
        if not hasattr(
            port_artifact,
            "predict_structured",
        ):
            raise TypeError(
                "port_artifact must expose predict_structured"
            )
        self.port_artifact = (
            port_artifact
        )
        self.spatial_artifact = (
            spatial_artifact
        )
        self.reference_config = (
            reference_config
            or MQSConfig()
        )
        self._reference = (
            MixedReferenceArtifact(
                config=(
                    self.reference_config
                )
            )
        )
        self.dielectric_surface_vertical_order = int(
            dielectric_surface_vertical_order
        )
        self.dielectric_surface_azimuthal_order = int(
            dielectric_surface_azimuthal_order
        )
        self._dielectric_reference = (
            DielectricCoupledReferenceArtifact(
                config=(
                    self.reference_config
                ),
                surface_vertical_order=(
                    self.dielectric_surface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.dielectric_surface_azimuthal_order
                ),
            )
        )

    @property
    def capabilities(
        self,
    ) -> SystemCapabilities:
        return mvp_system_capabilities()

    def _require_package_fast_port_artifact(
        self,
        scene: Scene,
    ):
        if (
            scene.packages
            and not bool(
                getattr(
                    self.port_artifact,
                    "supports_packages",
                    False,
                )
            )
        ):
            raise NotImplementedError(
                "package scenes require a package-aware FAST port artifact; "
                "conductor-only FAST predictions are not used for dielectric scenes"
            )


    def _require_fast_port_artifact(
        self,
        scene: Scene,
    ):
        self._require_package_fast_port_artifact(
            scene
        )
        if (
            scene.medium.conductivity
            > 0.0
            and not bool(
                getattr(
                    self.port_artifact,
                    "supports_lossy_background",
                    False,
                )
            )
        ):
            raise NotImplementedError(
                "lossy homogeneous backgrounds require a FAST artifact with "
                "an explicit background-dissipation channel; conductor-only "
                "FAST predictions must not silently ignore medium loss"
            )

    def _require_package_fast_spatial_artifact(
        self,
        scene: Scene,
    ):
        if not scene.packages:
            return
        if (
            self.spatial_artifact is None
            or not bool(
                getattr(
                    self.spatial_artifact,
                    "supports_packages",
                    False,
                )
            )
        ):
            raise NotImplementedError(
                "package scenes require a package-aware FAST spatial artifact; "
                "the conductor-only spatial decoder is not used as a dielectric field"
            )

    def fast_ports(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        self._require_fast_port_artifact(
            scene
        )
        return (
            self.port_artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )

    def reference_ports(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        artifact = (
            self._dielectric_reference
            if scene.packages
            else self._reference
        )
        return (
            artifact.predict_structured(
                scene,
                frequency_hz,
            )
        )

    def reference_result(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        if scene.packages:
            return (
                self._dielectric_reference.solve(
                    scene,
                    frequency_hz,
                )
            )
        return self._reference.solve(
            scene,
            frequency_hz,
        )[
            1
        ]

    def certified_ports(
        self,
        scene: Scene,
        frequency_hz: float,
        *,
        convergence_report=None,
        config: MQSConfig | None = None,
        **certification_options,
    ):
        if scene.packages:
            return certify_dielectric_ports(
                scene,
                frequency_hz,
                self.port_artifact,
                config=(
                    config
                    or self.reference_config
                ),
                convergence_report=(
                    convergence_report
                ),
                surface_vertical_order=(
                    self.dielectric_surface_vertical_order
                ),
                surface_azimuthal_order=(
                    self.dielectric_surface_azimuthal_order
                ),
                **certification_options,
            )
        return certify_mixed_ports(
            scene,
            frequency_hz,
            self.port_artifact,
            config=(
                config
                or self.reference_config
            ),
            convergence_report=(
                convergence_report
            ),
            **certification_options,
        )

    def fast_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        self._require_fast_port_artifact(
            scene
        )
        self._require_package_fast_spatial_artifact(
            scene
        )
        if (
            self.spatial_artifact
            is not None
        ):
            return (
                self.spatial_artifact.prepare(
                    scene,
                    frequency_hz,
                )
            )
        return (
            UniformLossFieldDecoder(
                self.port_artifact
            ).prepare(
                scene,
                frequency_hz,
            )
        )

    def reference_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        if scene.packages:
            return (
                self._dielectric_reference.prepare_spatial(
                    scene,
                    frequency_hz,
                )
            )
        return (
            self._reference.prepare_spatial(
                scene,
                frequency_hz,
            )
        )

    def reference_channel_current_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        artifact = (
            self._dielectric_reference
            if scene.packages
            else self._reference
        )
        return ChannelResolvedCurrentEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            artifact,
            **options,
        )

    def reference_channel_voltage_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        artifact = (
            self._dielectric_reference
            if scene.packages
            else self._reference
        )
        return ChannelResolvedVoltageEnvelope(
            scene,
            frequency_hz,
            thermal_model,
            artifact,
            **options,
        )

    def fast_current_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        self._require_fast_port_artifact(
            scene
        )
        if (
            scene.packages
            or scene.medium.conductivity > 0.0
        ):
            return ChannelResolvedCurrentEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self.port_artifact,
                **options,
            )
        return (
            FastCurrentControlledEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self.port_artifact,
                **options,
            )
        )

    def fast_voltage_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        self._require_fast_port_artifact(
            scene
        )
        if (
            scene.packages
            or scene.medium.conductivity > 0.0
        ):
            return ChannelResolvedVoltageEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self.port_artifact,
                **options,
            )
        return (
            FastVoltageControlledEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self.port_artifact,
                **options,
            )
        )

    def reference_current_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        if (
            scene.packages
            or scene.medium.conductivity > 0.0
        ):
            artifact = (
                self._dielectric_reference
                if scene.packages
                else self._reference
            )
            return ChannelResolvedCurrentEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                artifact,
                **options,
            )
        return (
            FastCurrentControlledEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self._reference,
                **options,
            )
        )

    def reference_voltage_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
        if (
            scene.packages
            or scene.medium.conductivity > 0.0
        ):
            artifact = (
                self._dielectric_reference
                if scene.packages
                else self._reference
            )
            return ChannelResolvedVoltageEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                artifact,
                **options,
            )
        return (
            FastVoltageControlledEnvelope(
                scene,
                frequency_hz,
                thermal_model,
                self._reference,
                **options,
            )
        )


    def fast_continuous_thermal_field(
        self,
        scene: Scene,
        frequency_hz: float,
        medium,
        **options,
    ):
        self._require_fast_port_artifact(
            scene
        )
        self._require_package_fast_spatial_artifact(
            scene
        )
        spatial = (
            self.spatial_artifact
            if self.spatial_artifact is not None
            else UniformLossFieldDecoder(
                self.port_artifact
            )
        )
        return ContinuousThermalGreenArtifact(
            spatial,
            medium,
            **options,
        ).prepare(
            scene,
            frequency_hz,
        )

    def reference_continuous_thermal_field(
        self,
        scene: Scene,
        frequency_hz: float,
        medium,
        **options,
    ):
        artifact = (
            self._dielectric_reference
            if scene.packages
            else self._reference
        )
        return ContinuousThermalGreenArtifact(
            artifact,
            medium,
            **options,
        ).prepare(
            scene,
            frequency_hz,
        )
