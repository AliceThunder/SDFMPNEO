"""BFZI-first electromagnetic/thermal research implementation.

Do not alias the historical SuperellipseSpiral to BFZICoil.
The v1 geometry package describes an ideal continuous conductor centerline
only.  A full electrical port requires explicit leads/return and cross-section.
"""
from .geometry import BFZICoil, RigidPose

__all__ = ["BFZICoil", "RigidPose"]
