"""Independent Frozen-CONCH 5x architecture baseline research package.

The top-level import remains lightweight: model/CONCH modules are loaded only
by their explicit runtime commands, and no AgentFlow behavioral code is
imported or modified here.
"""

from .data import (
    AnnotationAudit,
    ArchitectureDataAudit,
    CasePath,
    CanonicalLabel,
    SplitFold,
    annotation_guard,
    audit_architecture_data,
    audit_label_rows,
    audit_patch_annotations,
    build_case_paths,
    build_canonical_labels,
    build_five_fold_splits,
    load_xlsx_rows,
    stratified_group_folds,
    write_five_fold_split_artifacts,
    yx_family_heuristic,
)
from .manifest import (
    eligible_case_aliases,
    manifest_rows_by_case,
    merge_sanitized_manifests,
    validate_manifest,
    validate_patch_row,
)

__all__ = [
    "AnnotationAudit",
    "ArchitectureDataAudit",
    "CasePath",
    "CanonicalLabel",
    "SplitFold",
    "annotation_guard",
    "audit_architecture_data",
    "audit_label_rows",
    "audit_patch_annotations",
    "build_case_paths",
    "build_canonical_labels",
    "build_five_fold_splits",
    "load_xlsx_rows",
    "stratified_group_folds",
    "write_five_fold_split_artifacts",
    "yx_family_heuristic",
    "eligible_case_aliases",
    "manifest_rows_by_case",
    "merge_sanitized_manifests",
    "validate_manifest",
    "validate_patch_row",
]
