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
from .sea_rom import compress_seawater_response, restrict_seawater_response
from .sea_snapshot import BFZISnapshotROM
from .sea_dataset import (
    BFZIResponseLabel, geometry_features,
    scene_from_geometry_features,
    save_response_labels, load_response_labels,
)
from .electrothermal import (
    BFZICurrentControlledEnvelope, BFZIVoltageControlledEnvelope,
)

__all__ = [
    "BFZICoil", "RigidPose", "BfziScene", "BFZIAssembly",
    "InsulatingBox", "Seawater", "BFZISources", "LineSource",
    "build_bfzi_source", "SeaCurrentBasis", "build_sea_basis",
    "SeaMQSConfig", "SeawaterResponse", "solve_seawater_reference",
    "compress_seawater_response", "restrict_seawater_response",
    "BFZISnapshotROM",
    "BFZIResponseLabel", "geometry_features",
    "scene_from_geometry_features",
    "save_response_labels", "load_response_labels",
    "BFZICurrentControlledEnvelope",
    "BFZIVoltageControlledEnvelope",
]
