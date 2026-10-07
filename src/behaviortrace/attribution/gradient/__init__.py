"""Gradient-based attribution family.

LESS, Choe et al. LoGra, IF survey arXiv:2508.07297). Compared head-to-
head against the activation-difference family in an earlier experiment.

Strong baselines for C1: `tracin.tracin_attribute` (checkpoint-summed
first-order influence) and `trak.trak_attribute` (whitened influence).
`influence_functions` is a deferred stub (R6).
"""

from .tracin import tracin_attribute
from .trak import trak_attribute

__all__ = ["tracin_attribute", "trak_attribute"]
