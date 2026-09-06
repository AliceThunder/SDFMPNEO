from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .em import (
    ImpressedCurrentPortSet,
    ResidualGreedyEMReducer,
    TetrahedralApsiDiscretization,
    build_tetrahedral_apsi_from_thermal_modes,
    build_tetrahedral_region_loss_projector,
)
from .spatial import TetrahedralComplex3D, TetrahedralThermalAssembly
from .thermal import ThermalSpectralModel, ThermalTailCertificate


@dataclass(frozen=True)
class TetrahedralElectroThermalCore:
    """One-call construction of the current unstructured electrothermal core.

    The retained thermal rank is never supplied as an arbitrary integer. With no
    truncation request the full discrete thermal spectrum is retained. If a
    physically meaningful initial state, source dual bound, and state tolerance
    are supplied, the smallest rank satisfying the spectral-tail certificate is
    selected.
    """

    mesh: TetrahedralComplex3D
    thermal_assembly: TetrahedralThermalAssembly
    full_thermal_spectrum: ThermalSpectralModel
    thermal_model: ThermalSpectralModel
    thermal_tail_certificate: ThermalTailCertificate | None
    thermal_mode_local_values: np.ndarray
    conductivity_reference_tetra: np.ndarray
    conductivity_temperature_slope_tetra: np.ndarray
    electromagnetic_discretization: TetrahedralApsiDiscretization
    electromagnetic_problem: object

    @classmethod
    def build(
        cls,
        mesh: TetrahedralComplex3D,
        *,
        omega: float,
        reluctivity_tetra: np.ndarray,
        conductivity_reference_tetra: np.ndarray,
        conductivity_temperature_slope_tetra: np.ndarray,
        rho_cp_tetra: np.ndarray,
        thermal_conductivity_tetra: np.ndarray,
        source_current: np.ndarray,
        initial_temperature_deviation_free: np.ndarray | None = None,
        source_dual_bound: float | None = None,
        requested_state_tolerance: float | None = None,
    ) -> "TetrahedralElectroThermalCore":
        thermal_assembly = mesh.assemble_p1_thermal(
            rho_cp_tetra=rho_cp_tetra,
            conductivity_tetra=thermal_conductivity_tetra,
            homogeneous_dirichlet_boundary=True,
        )
        full_spectrum = ThermalSpectralModel.build(
            thermal_assembly.M,
            thermal_assembly.K,
        )

        certificate_inputs = (
            initial_temperature_deviation_free,
            source_dual_bound,
            requested_state_tolerance,
        )
        supplied = tuple(value is not None for value in certificate_inputs)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "certified thermal truncation requires initial_temperature_deviation_free, "
                "source_dual_bound, and requested_state_tolerance together"
            )

        if all(supplied):
            selection = full_spectrum.select_certified_rank(
                initial_field=np.asarray(initial_temperature_deviation_free, dtype=float),
                source_dual_bound=float(source_dual_bound),
                requested_state_tolerance=float(requested_state_tolerance),
            )
            thermal_model = selection.model
            certificate = selection.certificate
        else:
            thermal_model = full_spectrum
            certificate = None

        local_modes = []
        for k in range(thermal_model.rank):
            full_mode = thermal_assembly.expand_free(thermal_model.Phi[:, k])
            local_modes.append(full_mode[mesh.tetrahedra])
        local_modes_array = np.asarray(local_modes, dtype=float)

        sigma0 = np.asarray(conductivity_reference_tetra, dtype=float).copy()
        slope = np.asarray(conductivity_temperature_slope_tetra, dtype=float).copy()
        em_discretization = build_tetrahedral_apsi_from_thermal_modes(
            mesh,
            omega=omega,
            reluctivity_tetra=reluctivity_tetra,
            conductivity_reference_tetra=sigma0,
            conductivity_temperature_slope_tetra=slope,
            thermal_mode_local_values=local_modes_array,
            source_current=source_current,
        )
        em_problem = em_discretization.to_parametric_problem()

        if em_problem.n_thermal != thermal_model.rank:
            raise RuntimeError("internal tetrahedral electrothermal rank mismatch")

        return cls(
            mesh=mesh,
            thermal_assembly=thermal_assembly,
            full_thermal_spectrum=full_spectrum,
            thermal_model=thermal_model,
            thermal_tail_certificate=certificate,
            thermal_mode_local_values=local_modes_array,
            conductivity_reference_tetra=sigma0,
            conductivity_temperature_slope_tetra=slope,
            electromagnetic_discretization=em_discretization,
            electromagnetic_problem=em_problem,
        )

    def build_reduced_electromagnetics(
        self,
        candidate_thermal_states: Iterable[np.ndarray],
        *,
        residual_tolerance: float,
    ):
        """Construct the snapshot-free electromagnetic reduced space.

        This method intentionally requires an explicit residual tolerance and a
        supplied verification set. For an actual certified parameter domain the
        resulting model must additionally pass the continuous-domain certificate;
        this convenience method does not reinterpret a finite set as proof.
        """

        return ResidualGreedyEMReducer(self.electromagnetic_problem).build(
            candidate_thermal_states,
            tolerance=residual_tolerance,
        )

    def build_ports(
        self,
        edge_currents: np.ndarray,
        *,
        names: Sequence[str] | None = None,
    ) -> ImpressedCurrentPortSet:
        """Build work-conjugate closed impressed-current ports on this mesh."""

        return ImpressedCurrentPortSet.build(
            self.mesh,
            a_basis=self.electromagnetic_discretization.a_basis.toarray(),
            n_scalar=self.electromagnetic_discretization.n_scalar,
            omega=self.electromagnetic_discretization.omega,
            edge_currents=edge_currents,
            names=names,
        )

    def build_region_loss_projector(
        self,
        regions: Mapping[str, np.ndarray],
    ):
        """Build total Joule-power diagnostics for named tetrahedral regions."""

        return build_tetrahedral_region_loss_projector(
            self.mesh,
            self.electromagnetic_discretization,
            conductivity_reference_tetra=self.conductivity_reference_tetra,
            conductivity_temperature_slope_tetra=self.conductivity_temperature_slope_tetra,
            thermal_mode_local_values=self.thermal_mode_local_values,
            regions=regions,
        )
