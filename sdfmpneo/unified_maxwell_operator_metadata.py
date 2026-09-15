"""Attach non-numerical assembly metadata to production Maxwell matrices.

SciPy sparse matrices are Python objects, so the assembled matrix can carry the
background/context that produced it.  This lets the certified Krylov policy
construct the compatible scalar-gradient block without changing the matrix
entries or duplicating the local-self solver implementation.
"""
from __future__ import annotations


def install(open_boundary_class):
    if bool(getattr(open_boundary_class, "_maxwell_operator_metadata_installed", False)):
        return open_boundary_class
    original = open_boundary_class.em_operator

    def tagged_em_operator(self, context, state=None):
        matrix = original(self, context, state)
        matrix._sdfmpneo_background = self
        matrix._sdfmpneo_context = context
        matrix._sdfmpneo_mqs = False
        return matrix

    open_boundary_class.em_operator = tagged_em_operator
    open_boundary_class._maxwell_operator_metadata_installed = True
    return open_boundary_class


__all__ = ["install"]
