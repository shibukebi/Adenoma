"""Expert-confirmed 5x architecture benchmark preparation and freezing.

This package deliberately contains no model inference or training entry
points.  The benchmark must be frozen before any downstream model arm is
allowed to consume it.
"""

from .pipeline import (
    AnnotationConflictError,
    FreezeValidationError,
    ProvenanceError,
    ResumeRefusedError,
    freeze_benchmark,
    prepare_annotation,
    status_report,
)

__all__ = [
    "AnnotationConflictError",
    "FreezeValidationError",
    "ProvenanceError",
    "ResumeRefusedError",
    "freeze_benchmark",
    "prepare_annotation",
    "status_report",
]
