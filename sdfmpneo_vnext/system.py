from __future__ import annotations

from dataclasses import dataclass

from .certified import certify_mixed_ports
from .em import MQSConfig
from .fast import (
    FastCurrentControlledEnvelope,
    FastVoltageControlledEnvelope,
)
from .field import UniformLossFieldDecoder
from .reference import MixedReferenceArtifact
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
            "homogeneous_isotropic_unbounded_lossless"
        ),
        lossy_background_media=False,
        heterogeneous_media=False,
        retardation=False,
        arbitrary_se3_pose=True,
        superelliptic_conductors=True,
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

    @property
    def capabilities(
        self,
    ) -> SystemCapabilities:
        return mvp_system_capabilities()

    def fast_ports(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
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
        return (
            self._reference.predict_structured(
                scene,
                frequency_hz,
            )
        )

    def reference_result(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
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
        return (
            self._reference.prepare_spatial(
                scene,
                frequency_hz,
            )
        )

    def fast_current_envelope(
        self,
        scene: Scene,
        frequency_hz: float,
        thermal_model,
        **options,
    ):
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
        return ContinuousThermalGreenArtifact(
            self._reference,
            medium,
            **options,
        ).prepare(
            scene,
            frequency_hz,
        )
