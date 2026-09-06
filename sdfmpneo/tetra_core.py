from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .em import (
    BlockActionFactory,
    ConductivityRegion,
    ImpressedCurrentPortSet,
    NonlinearTetrahedralApsiProblem,
    NonlinearTetrahedralRegionLossEvaluator,
    ResidualGreedyEMReducer,
    SparseEnergyResidualGreedyEMReducer,
    TetrahedralApsiDiscretization,
    build_tetrahedral_apsi_from_thermal_modes,
    build_tetrahedral_region_loss_projector,
    make_physical_block_pcg_riesz_factory,
)
from .spatial import TetrahedralComplex3D, TetrahedralThermalAssembly
from .thermal import ThermalSpectralModel, ThermalTailCertificate


def _build_thermal_components(
    mesh: TetrahedralComplex3D,
    *,
    rho_cp_tetra: np.ndarray,
    thermal_conductivity_tetra: np.ndarray,
    initial_temperature_deviation_free: np.ndarray | None,
    source_dual_bound: float | None,
    requested_state_tolerance: float | None,
):
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
    return (
        thermal_assembly,
        full_spectrum,
        thermal_model,
        certificate,
        np.asarray(local_modes, dtype=float),
    )


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
    material_backend: str
    conductivity_reference_tetra: np.ndarray | None
    conductivity_temperature_slope_tetra: np.ndarray | None
    conductivity_regions: tuple[ConductivityRegion, ...] | None
    electromagnetic_discretization: object
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
        (
            thermal_assembly,
            full_spectrum,
            thermal_model,
            certificate,
            local_modes_array,
        ) = _build_thermal_components(
            mesh,
            rho_cp_tetra=rho_cp_tetra,
            thermal_conductivity_tetra=thermal_conductivity_tetra,
            initial_temperature_deviation_free=initial_temperature_deviation_free,
            source_dual_bound=source_dual_bound,
            requested_state_tolerance=requested_state_tolerance,
        )

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
            material_backend="affine_verification",
            conductivity_reference_tetra=sigma0,
            conductivity_temperature_slope_tetra=slope,
            conductivity_regions=None,
            electromagnetic_discretization=em_discretization,
            electromagnetic_problem=em_problem,
        )

    @classmethod
    def build_nonlinear(
        cls,
        mesh: TetrahedralComplex3D,
        *,
        omega: float,
        reluctivity_tetra: np.ndarray,
        conductivity_regions: Sequence[ConductivityRegion],
        temperature_reference_nodal: np.ndarray,
        constitutive_relative_error_budget: float,
        rho_cp_tetra: np.ndarray,
        thermal_conductivity_tetra: np.ndarray,
        source_current: np.ndarray,
        initial_temperature_deviation_free: np.ndarray | None = None,
        source_dual_bound: float | None = None,
        requested_state_tolerance: float | None = None,
    ) -> "TetrahedralElectroThermalCore":
        """Build the real nonlinear material path without conductivity linearization."""

        (
            thermal_assembly,
            full_spectrum,
            thermal_model,
            certificate,
            local_modes_array,
        ) = _build_thermal_components(
            mesh,
            rho_cp_tetra=rho_cp_tetra,
            thermal_conductivity_tetra=thermal_conductivity_tetra,
            initial_temperature_deviation_free=initial_temperature_deviation_free,
            source_dual_bound=source_dual_bound,
            requested_state_tolerance=requested_state_tolerance,
        )

        reference_nodal = np.asarray(temperature_reference_nodal, dtype=float)
        if reference_nodal.shape != (mesh.n_nodes,):
            raise ValueError("temperature_reference_nodal must have shape (n_nodes,)")
        reference_local = reference_nodal[mesh.tetrahedra]
        regions = tuple(conductivity_regions)
        em_problem = NonlinearTetrahedralApsiProblem(
            mesh,
            omega=omega,
            reluctivity_tetra=reluctivity_tetra,
            source_current=source_current,
            temperature_reference_local=reference_local,
            thermal_modes_local=local_modes_array,
            conductivity_regions=regions,
            constitutive_relative_error_budget=constitutive_relative_error_budget,
        )

        if em_problem.n_thermal != thermal_model.rank:
            raise RuntimeError("internal nonlinear tetrahedral electrothermal rank mismatch")

        return cls(
            mesh=mesh,
            thermal_assembly=thermal_assembly,
            full_thermal_spectrum=full_spectrum,
            thermal_model=thermal_model,
            thermal_tail_certificate=certificate,
            thermal_mode_local_values=local_modes_array,
            material_backend="certified_nonlinear",
            conductivity_reference_tetra=None,
            conductivity_temperature_slope_tetra=None,
            conductivity_regions=regions,
            electromagnetic_discretization=em_problem,
            electromagnetic_problem=em_problem,
        )

    def build_reduced_electromagnetics(
        self,
        candidate_thermal_states: Iterable[np.ndarray],
        *,
        requested_energy_state_error: float,
        block_action_factory: BlockActionFactory | None = None,
    ):
        """Build the certified nonlinear sparse physical-energy EM reduced space.

        The production path always uses the same physical-block PCG Riesz
        architecture.  For H=K+D, the outer action is certified by

            H >= m(gamma) min(m_K,m_E) P,

        where m_K and m_E come from the replaceable magnetic/scalar block
        actions.  The default block action is complete sparse LU strictly as a
        correctness backend; a certified multilevel/auxiliary-space action can
        replace it without changing the scientific method or error theorem.

            ||e||_H <= sqrt(2) ||r||_(H^-1).

        The finite candidate set is an offline construction/certification set; it
        is not silently reinterpreted as a continuous-domain proof.
        """

        if self.material_backend != "certified_nonlinear":
            raise RuntimeError(
                "build_reduced_electromagnetics is the certified nonlinear sparse-energy path; "
                "use build_affine_verification_reduced_electromagnetics for the affine verification backend"
            )
        riesz_factory = make_physical_block_pcg_riesz_factory(
            self.electromagnetic_problem,
            block_action_factory=block_action_factory,
        )
        return SparseEnergyResidualGreedyEMReducer(
            self.electromagnetic_problem,
            riesz_action_factory=riesz_factory,
        ).build(
            candidate_thermal_states,
            requested_energy_state_error=requested_energy_state_error,
        )

    def build_affine_verification_reduced_electromagnetics(
        self,
        candidate_thermal_states: Iterable[np.ndarray],
        *,
        residual_tolerance: float,
    ):
        """Retain the legacy dense affine reducer strictly as a verification path."""

        if self.material_backend != "affine_verification":
            raise RuntimeError("affine verification reduction requires the affine verification backend")
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
        """Build work-conjugate closed impressed-current ports without densifying the gauge basis."""

        return ImpressedCurrentPortSet.build(
            self.mesh,
            a_basis=self.electromagnetic_discretization.a_basis,
            n_scalar=self.electromagnetic_discretization.n_scalar,
            omega=self.electromagnetic_discretization.omega,
            edge_currents=edge_currents,
            names=names,
        )

    def build_region_loss_projector(
        self,
        regions: Mapping[str, np.ndarray] | None = None,
    ):
        """Build region Joule-power diagnostics from the active material backend.

        The affine verification path requires explicit tetrahedral masks. The
        certified nonlinear path uses the conductivity regions declared when the
        core was constructed; supplying alternate masks there is rejected so the
        diagnostic cannot bypass the certified constitutive definition.
        """

        if self.material_backend == "certified_nonlinear":
            if regions is not None:
                raise ValueError(
                    "certified nonlinear diagnostics use the declared conductivity regions; "
                    "custom masks are not accepted"
                )
            return NonlinearTetrahedralRegionLossEvaluator(self.electromagnetic_problem)

        if regions is None:
            raise ValueError("affine tetrahedral region diagnostics require explicit region masks")
        if self.conductivity_reference_tetra is None or self.conductivity_temperature_slope_tetra is None:
            raise RuntimeError("affine material metadata is unavailable")
        return build_tetrahedral_region_loss_projector(
            self.mesh,
            self.electromagnetic_discretization,
            conductivity_reference_tetra=self.conductivity_reference_tetra,
            conductivity_temperature_slope_tetra=self.conductivity_temperature_slope_tetra,
            thermal_mode_local_values=self.thermal_mode_local_values,
            regions=regions,
        )
