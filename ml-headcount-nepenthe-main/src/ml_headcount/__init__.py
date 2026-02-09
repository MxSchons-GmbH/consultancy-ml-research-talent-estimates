"""
ML Headcount Pipeline

A comprehensive pipeline for analyzing ML headcount estimates across organizations.
"""

# Suppress upstream warnings on import
from .utils.warning_suppression import suppress_upstream_warnings
suppress_upstream_warnings()

# Import modal functions to ensure they're registered
from . import modal_functions

__version__ = "0.1.0"
