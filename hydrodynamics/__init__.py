"""Scoped hydrodynamic theory and experimental references.

Start at HYDRODYNAMICS_CORE.md. The legacy scene adapter and candidate wake,
tail and wave-response fits are exploratory, not a validated CAD rowing model.
"""
from .core import UnboundedPotentialInertia, shift_wrench, cummins_kernel
from .reference_models import (OutsideReferenceDomain, DayReferenceResistance,
    grift_steady_normal_force, caplan_static_force, reijtenbagh_start_force,
    validate_components)

__all__=['UnboundedPotentialInertia','shift_wrench','cummins_kernel',
    'OutsideReferenceDomain','DayReferenceResistance','grift_steady_normal_force',
    'caplan_static_force','reijtenbagh_start_force','validate_components']
