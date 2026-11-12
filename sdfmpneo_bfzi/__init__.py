"""BFZI continuous spiral + structure-preserving seawater MQS integration.

The new BFZI path keeps the useful vNext port/channel design but does not
alias the historical SuperellipseSpiral geometry. Seawater results are
finite-domain Galerkin approximations, not certified infinite-ocean solutions.
"""
from .geometry import BFZICoil, RigidPose
from .scene import BfziScene, BFZIAssembly, InsulatingBox, Seawater
from .source import BFZISources, LineSource, build_bfzi_source
from .sea_basis import SeaCurrentBasis, build_sea_basis
from .sea_reference import SeaMQSConfig, SeawaterResponse, solve_seawater_reference
from .green_coulomb import assemble_coulomb_split, assemble_basis_coulomb_split
from .exterior_dipole import ExteriorDipole
from .sea_rom import compress_seawater_response, restrict_seawater_response
from .sea_snapshot import BFZISnapshotROM
from .sea_dataset import (
    BFZIResponseLabel, geometry_features,
    scene_from_geometry_features,
    save_response_labels, load_response_labels,
)
from .m1_direct import M1Direct, solve_m1
from .m1_direct import M1Direct, solve_m1
from .dynamic_et import BFZIDynamicET, ETStep
from .electrothermal import (
    BFZICurrentControlledEnvelope, BFZIVoltageControlledEnvelope,
)

__all__ = [
    "BFZICoil", "RigidPose", "BfziScene", "BFZIAssembly",
    "InsulatingBox", "Seawater", "BFZISources", "LineSource",
    "build_bfzi_source", "SeaCurrentBasis", "build_sea_basis",
    "SeaMQSConfig", "SeawaterResponse", "solve_seawater_reference",
    "assemble_coulomb_split", "assemble_basis_coulomb_split",
    "ExteriorDipole",
    "compress_seawater_response", "restrict_seawater_response",
    "BFZISnapshotROM",
    "BFZIResponseLabel", "geometry_features",
    "scene_from_geometry_features",
    "save_response_labels", "load_response_labels",
    "M1Direct", "solve_m1",
    "M1Direct", "solve_m1",
    "BFZIDynamicET", "ETStep",
    "BFZICurrentControlledEnvelope",
    "BFZIVoltageControlledEnvelope",
]
