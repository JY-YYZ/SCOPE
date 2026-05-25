"""Small public API for the SCOPE core implementation."""

from .covariance import ActivationCovariance
from .directions import extract_refusal_direction, resolve_module
from .editing import project_weight_gradient, refusal_alignment_loss

# Keep the exported surface explicit so downstream scripts import stable names.
__all__ = [
    "ActivationCovariance",
    "extract_refusal_direction",
    "project_weight_gradient",
    "refusal_alignment_loss",
    "resolve_module",
]
