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
    make_morse_auxiliary_physical_pcg_riesz_factory,
    make_physical_block_pcg_riesz_factory,
)
from .spatial import TetrahedralComplex3D, TetrahedralThermalAssembly
from .thermal import (
    ThermalSpectralModel,
    ThermalTailCertificate,
    build_partial_thermal_spectrum,
)


def _select_thermal_spectrum(
    mesh: TetrahedralComplex3D,
    thermal_assembly: TetrahedralThermalAssembly,
    *,
    rho_cp_tetra: np.ndarray,
    thermal_conductivity_tetra: np.ndarray,
    initial_temperature_deviation_free: np.ndarray | None,
    source_dual_bound: float | None,
    requested_state_tolerance: float | None,
    prefer_partial_spectrum: bool,
    thermal_rank: int | None = None,
):
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

    n_free = int(thermal_assembly.M.shape[0])
    if n_free <= 0:
        raise ValueError("thermal boundary treatment produced no free degrees of freedom")

    if thermal_rank is not None:
        if any(supplied):
            raise ValueError("choose an explicit research rank or certificate-selected truncation, not both")
        if int(thermal_rank) != thermal_rank or not 1 <= thermal_rank <= n_free:
            raise ValueError("thermal_rank must be an integer between one and the free thermal dimension")
        if thermal_rank < n_free:
            partial = build_partial_thermal_spectrum(
                thermal_assembly.M, thermal_assembly.K, rank=int(thermal_rank),
                domain_volume=float(np.sum(mesh.volumes)),
                thermal_conductivity_min=float(np.min(thermal_conductivity_tetra)),
                volumetric_heat_capacity_max=float(np.max(rho_cp_tetra)),
            )
            # Explicit rank is a numerical approximation for a convergence
            # study, not a claim that the thermal tail meets any error target.
            return None, partial.model, None, "partial_numerical_rank"

    if all(supplied) and prefer_partial_spectrum and n_free > 1:
        initial = np.asarray(initial_temperature_deviation_free, dtype=float)
        if initial.shape != (n_free,):
            raise ValueError("initial_temperature_deviation_free dimension mismatch")
        volume = float(np.sum(np.asarray(mesh.volumes, dtype=float)))
        kappa_min = float(np.min(np.asarray(thermal_conductivity_tetra, dtype=float)))
        rho_cp_max = float(np.max(np.asarray(rho_cp_tetra, dtype=float)))
        if volume <= 0.0 or kappa_min <= 0.0 or rho_cp_max <= 0.0:
            raise ValueError("partial thermal certification requires positive volume/material bounds")

        # The retained rank is not supplied as a free hyperparameter.  Starting
        # from the smallest admissible rank, compute only low modes and use the
        # Li-Yau omitted-eigenvalue lower bound.  The first rank whose rigorous
        # tail certificate meets the requested state tolerance is accepted.
        for rank in range(1, n_free):
            partial = build_partial_thermal_spectrum(
                thermal_assembly.M,
                thermal_assembly.K,
                rank=rank,
                domain_volume=volume,
                thermal_conductivity_min=kappa_min,
                volumetric_heat_capacity_max=rho_cp_max,
            )
            certificate = partial.tail_certificate(
                initial_field=initial,
                source_dual_bound=float(source_dual_bound),
                requested_state_tolerance=float(requested_state_tolerance),
            )
            if certificate.certified:
                return None, partial.model, certificate, "partial_certified"

    # Correctness fallback: either no truncation certificate was requested, the
    # discrete thermal dimension is one, partial selection was explicitly
    # disabled, or the conservative omitted-mode bound could not prove the
    # requested tolerance.  The full discrete spectrum then gives an exact
    # verification fallback rather than weakening the certificate.
    full_spectrum = ThermalSpectralModel.build(
        thermal_assembly.M,
        thermal_assembly.K,
    )
    if all(supplied):
        selection = full_spectrum.select_certified_rank(
            initial_field=np.asarray(initial_temperature_deviation_free, dtype=float),
            source_dual_bound=float(source_dual_bound),
            requested_state_tolerance=float(requested_state_tolerance),
        )
        return full_spectrum, selection.model, selection.certificate, "full_certified_fallback"
    return full_spectrum, full_spectrum, None, "full_discrete"


def _build_thermal_components(
    mesh: TetrahedralComplex3D,
    *,
    rho_cp_tetra: np.ndarray,
    thermal_conductivity_tetra: np.ndarray,
    initial_temperature_deviation_free: np.ndarray | None,
    source_dual_bound: float | None,
    requested_state_tolerance: float | None,
    prefer_partial_spectrum: bool = True,
    thermal_rank: int | None = None,
):
    thermal_assembly = mesh.assemble_p1_thermal(
        rho_cp_tetra=rho_cp_tetra,
        conductivity_tetra=thermal_conductivity_tetra,
        homogeneous_dirichlet_boundary=True,
    )
    full_spectrum, thermal_model, certificate, spectrum_backend = _select_thermal_spectrum(
        mesh,
        thermal_assembly,
        rho_cp_tetra=rho_cp_tetra,
        thermal_conductivity_tetra=thermal_conductivity_tetra,
        initial_temperature_deviation_free=initial_temperature_deviation_free,
        source_dual_bound=source_dual_bound,
        requested_state_tolerance=requested_state_tolerance,
        prefer_partial_spectrum=bool(prefer_partial_spectrum),
        thermal_rank=thermal_rank,
    )

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
        spectrum_backend,
    )


@dataclass(frozen=True)
class TetrahedralElectroThermalCore:
    """One-call construction of the unstructured electrothermal production core.

    Thermal rank is certificate-selected.  When a thermal truncation tolerance is
    supplied, the production default first tries a scalable partial eigensolve
    with an independently proved omitted-eigenvalue lower bound.  Full-spectrum
    construction is retained only as a deterministic correctness fallback.
    """

    mesh: TetrahedralComplex3D
    thermal_assembly: TetrahedralThermalAssembly
    full_thermal_spectrum: ThermalSpectralModel | None
    thermal_model: ThermalSpectralModel
    thermal_tail_certificate: ThermalTailCertificate | None
    thermal_mode_local_values: np.ndarray
    thermal_spectrum_backend: str
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
        prefer_partial_thermal_spectrum: bool = True,
    ) -> "TetrahedralElectroThermalCore":
        (
            thermal_assembly,
            full_spectrum,
            thermal_model,
            certificate,
            local_modes_array,
            spectrum_backend,
        ) = _build_thermal_components(
            mesh,
            rho_cp_tetra=rho_cp_tetra,
            thermal_conductivity_tetra=thermal_conductivity_tetra,
            initial_temperature_deviation_free=initial_temperature_deviation_free,
            source_dual_bound=source_dual_bound,
            requested_state_tolerance=requested_state_tolerance,
            prefer_partial_spectrum=prefer_partial_thermal_spectrum,
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
            thermal_spectrum_backend=spectrum_backend,
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
        prefer_partial_thermal_spectrum: bool = True,
        thermal_rank: int | None = None,
    ) -> "TetrahedralElectroThermalCore":
        """Build the real nonlinear material path without conductivity linearization."""

        (
            thermal_assembly,
            full_spectrum,
            thermal_model,
            certificate,
            local_modes_array,
            spectrum_backend,
        ) = _build_thermal_components(
            mesh,
            rho_cp_tetra=rho_cp_tetra,
            thermal_conductivity_tetra=thermal_conductivity_tetra,
            initial_temperature_deviation_free=initial_temperature_deviation_free,
            source_dual_bound=source_dual_bound,
            requested_state_tolerance=requested_state_tolerance,
            prefer_partial_spectrum=prefer_partial_thermal_spectrum,
            thermal_rank=thermal_rank,
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
            thermal_spectrum_backend=spectrum_backend,
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
        port_set: ImpressedCurrentPortSet | None = None,
        block_action_factory: BlockActionFactory | None = None,
    ):
        """Build the certified nonlinear sparse physical-energy EM reduced space.

        The production default is the factorization-free Morse magnetic auxiliary
        plus state-aware conductive scalar-tree Riesz action.  Complete sparse LU
        is no longer selected implicitly.  ``block_action_factory`` is retained
        only as an explicit compatibility hook for a caller that deliberately
        supplies the older physical-block theorem implementation.

        If ``port_set`` is supplied, every unit port excitation is included in
        one joint multi-RHS residual-greedy construction.  The finite candidate
        set is an offline construction set; continuous-domain proof remains a
        separate certificate.
        """

        if self.material_backend != "certified_nonlinear":
            raise RuntimeError(
                "build_reduced_electromagnetics is the certified nonlinear sparse-energy path; "
                "use build_affine_verification_reduced_electromagnetics for the affine verification backend"
            )
        if block_action_factory is None:
            riesz_factory = make_morse_auxiliary_physical_pcg_riesz_factory(
                self.electromagnetic_problem
            )
        else:
            riesz_factory = make_physical_block_pcg_riesz_factory(
                self.electromagnetic_problem,
                block_action_factory=block_action_factory,
            )
        reducer = SparseEnergyResidualGreedyEMReducer(
            self.electromagnetic_problem,
            riesz_action_factory=riesz_factory,
        )

        if port_set is None:
            rhs_matrix = np.asarray(self.electromagnetic_problem.b, dtype=complex)[:, None]
        else:
            rhs_matrix = np.asarray(port_set.coordinate_rhs, dtype=complex)
            if rhs_matrix.ndim != 2 or rhs_matrix.shape[0] != self.electromagnetic_problem.n_em:
                raise ValueError("port_set coordinates do not match electromagnetic problem")

        return reducer.build_multi_rhs(
            candidate_thermal_states,
            rhs_matrix,
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
