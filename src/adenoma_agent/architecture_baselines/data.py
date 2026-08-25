"""Read-only data audit and deterministic split contracts for 5x baselines.

This module intentionally uses only the Python standard library.  In
particular, the workbook reader is a small XLSX (OOXML) reader so that the
data audit does not acquire a pandas/openpyxl dependency.  It never writes to
the WSI roots.

The two identities in the output are kept conceptually separate:

* ``case_alias`` is a stable, source-safe identifier used by model artifacts;
* ``source_path`` is local provenance and is never used as model input.

Patch-level architecture targets are blocked unless a reviewed annotation
manifest is supplied.  A slide label is never broadcast to a patch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from zipfile import ZipFile


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WSI_SUFFIXES = (".svs", ".isyntax")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return token.strip("_") or "unnamed"


def _normalise_slide_stem(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    raw = raw.rsplit("/", 1)[-1]
    for suffix in WSI_SUFFIXES:
        if raw.lower().endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    return raw.strip()


def _normalise_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalise_grade(value: Any) -> str:
    return _normalise_label(value).lower()


def _column_index(cell_ref: str) -> int:
    letters = "".join(character for character in str(cell_ref or "") if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return max(0, value - 1)


def _shared_strings(archive: ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    ns = "{" + SPREADSHEET_NS + "}"
    return [
        "".join(node.text or "" for node in item.findall(".//" + ns + "t"))
        for item in root.findall(ns + "si")
    ]


def _first_sheet_path(archive: ZipFile) -> str:
    ns = "{" + SPREADSHEET_NS + "}"
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(ns + "sheets/" + ns + "sheet")
    if sheet is None:
        raise ValueError("XLSX workbook has no worksheet")
    rel_id = sheet.attrib.get("{" + OFFICE_REL_NS + "}id", "")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = ""
    for relationship in relationships.findall("{" + PACKAGE_REL_NS + "}Relationship"):
        if relationship.attrib.get("Id") == rel_id:
            target = relationship.attrib.get("Target", "")
            break
    if not target:
        raise ValueError("XLSX first worksheet relationship is missing")
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return "xl/" + target.lstrip("/")


def load_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    """Load the first XLSX sheet as rows keyed by its first-row headers.

    The implementation supports shared strings, inline strings, ordinary
    numeric/text cells, and sparse cell coordinates.  Formula evaluation is
    intentionally not attempted; this audit consumes the stored cell value.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("XLSX label workbook is missing: {0}".format(path))
    ns = "{" + SPREADSHEET_NS + "}"
    with ZipFile(str(path)) as archive:
        names = set(archive.namelist())
        required = {"xl/workbook.xml", "xl/_rels/workbook.xml.rels"}
        missing = sorted(required - names)
        if missing:
            raise ValueError("Invalid XLSX, missing: {0}".format(", ".join(missing)))
        shared = _shared_strings(archive)
        sheet_root = ET.fromstring(archive.read(_first_sheet_path(archive)))
        raw_rows: List[List[str]] = []
        for row in sheet_root.findall(ns + "sheetData/" + ns + "row"):
            values: List[str] = []
            for cell in row.findall(ns + "c"):
                index = _column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                cell_type = cell.attrib.get("t", "")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//" + ns + "t"))
                else:
                    value_node = cell.find(ns + "v")
                    value = value_node.text if value_node is not None and value_node.text is not None else ""
                    if cell_type == "s" and value:
                        try:
                            value = shared[int(value)]
                        except (IndexError, ValueError):
                            raise ValueError("Invalid shared-string index in {0}".format(path))
                    elif cell_type == "b":
                        value = "true" if value == "1" else "false"
                values[index] = str(value).strip()
            raw_rows.append(values)
    if not raw_rows:
        return []
    headers = [_normalise_label(value) for value in raw_rows[0]]
    if not any(headers):
        return []
    output: List[Dict[str, str]] = []
    for raw in raw_rows[1:]:
        values = list(raw) + [""] * max(0, len(headers) - len(raw))
        output.append({headers[index]: str(values[index]).strip() for index in range(len(headers)) if headers[index]})
    return output


@dataclass(frozen=True)
class CasePath:
    case_alias: str
    source_code: str
    source_path: str
    slide_stem: str
    source_format: str
    family_id: str
    label_status: str = "missing"

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    def inference_identity(self) -> Dict[str, str]:
        """Return the path/label-free identity allowed into model artifacts."""

        return {
            "case_alias": self.case_alias,
            "source_code": self.source_code,
            "source_format": self.source_format,
        }

    def local_provenance(self) -> Dict[str, str]:
        return {
            "case_alias": self.case_alias,
            "source_path": self.source_path,
            "slide_stem": self.slide_stem,
        }


@dataclass(frozen=True)
class CanonicalLabel:
    case_alias: str
    source_code: str
    label: str
    grade: str
    family_id: str
    label_source_sha256: str

    def to_json(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationAudit:
    status: str
    reason: str
    n_rows: int = 0
    n_reviewed_rows: int = 0
    label_vocabulary: Tuple[str, ...] = ()
    slide_label_broadcast: bool = False

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["label_vocabulary"] = list(self.label_vocabulary)
        return payload


@dataclass(frozen=True)
class SplitFold:
    fold_index: int
    train_case_aliases: Tuple[str, ...]
    val_case_aliases: Tuple[str, ...]
    test_case_aliases: Tuple[str, ...]
    group_by_case: Mapping[str, str]
    label_support: Mapping[str, Mapping[str, int]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_case_aliases": list(self.train_case_aliases),
            "val_case_aliases": list(self.val_case_aliases),
            "test_case_aliases": list(self.test_case_aliases),
            "group_by_case": dict(self.group_by_case),
            "label_support": {
                split: dict(support) for split, support in self.label_support.items()
            },
        }


@dataclass(frozen=True)
class ArchitectureDataAudit:
    summary: Mapping[str, Any]
    case_paths: Tuple[CasePath, ...]
    canonical_labels: Tuple[CanonicalLabel, ...]
    splits: Tuple[SplitFold, ...]
    annotation: AnnotationAudit
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "summary": dict(self.summary),
            "artifact_paths": dict(self.artifact_paths),
            "annotation": self.annotation.to_json(),
            "case_paths": [row.to_json() for row in self.case_paths],
            "canonical_labels": [row.to_json() for row in self.canonical_labels],
            "splits": [row.to_json() for row in self.splits],
        }


def yx_family_heuristic(slide_stem: str) -> str:
    """Conservatively group YX serial sections by its accession-like token.

    YX names commonly look like ``138265_746091001``.  The trailing three
    digits identify the section/scan variant in the available naming scheme;
    retaining ``746091`` groups those variants while avoiding the source path.
    Names that do not match this shape remain singleton families.
    """

    stem = _normalise_slide_stem(slide_stem)
    parts = stem.split("_", 1)
    if len(parts) == 2:
        match = re.match(r"^(.*?)(\d{3})$", parts[1])
        if match and match.group(1):
            return "YX_FAMILY_" + _safe_token(match.group(1))
    return "YX_FAMILY_" + _safe_token(stem)


def _family_id(source_code: str, slide_stem: str) -> str:
    source = _safe_token(source_code).upper()
    if source == "YX":
        return yx_family_heuristic(slide_stem)
    return source + "_FAMILY_" + _safe_token(slide_stem)


def _case_alias(source_code: str, index: int, path: Path) -> str:
    source = _safe_token(source_code).upper()
    stable_key = (source + "\0" + path.name).encode("utf-8")
    return "CASE_{0}_{1:06d}_{2}".format(source, index, hashlib.sha256(stable_key).hexdigest()[:10])


def build_case_paths(
    source_roots: Mapping[str, Path],
    label_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[CasePath, ...]:
    """Inventory WSI paths and attach only a label *status*.

    Diagnosis values are intentionally not placed on ``CasePath``.  They are
    emitted separately as ``CanonicalLabel`` rows after conflict filtering.
    """

    labels_by_stem: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for row in label_rows or ():
        stem = _normalise_slide_stem(row.get("slide_name") or row.get("slide_id") or row.get("case_id"))
        label = _normalise_label(row.get("type") or row.get("label"))
        grade = _normalise_grade(row.get("grade"))
        if stem and label and grade:
            labels_by_stem[stem].add((label, grade))
    output: List[CasePath] = []
    for source_code, root_value in sorted(source_roots.items(), key=lambda item: str(item[0])):
        root = Path(root_value).resolve()
        if not root.is_dir():
            raise FileNotFoundError("WSI source directory is missing: {0}".format(root))
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in WSI_SUFFIXES),
            key=lambda path: str(path.relative_to(root)),
        )
        for index, path in enumerate(paths, 1):
            stem = path.stem
            labels = labels_by_stem.get(stem, set())
            status = "matched" if len(labels) == 1 else "conflicting" if len(labels) > 1 else "missing"
            output.append(
                CasePath(
                    case_alias=_case_alias(str(source_code), index, path),
                    source_code=_safe_token(source_code).upper(),
                    source_path=str(path),
                    slide_stem=stem,
                    source_format=path.suffix.lower().lstrip("."),
                    family_id=_family_id(str(source_code), stem),
                    label_status=status,
                )
            )
    return tuple(output)


def _label_columns(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    stem = _normalise_slide_stem(row.get("slide_name") or row.get("slide_id") or row.get("case_id"))
    label = _normalise_label(row.get("type") or row.get("label"))
    grade = _normalise_grade(row.get("grade"))
    return stem, label, grade


def audit_label_rows(label_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_stem: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    rows_by_stem: Counter = Counter()
    for row in label_rows:
        stem, label, grade = _label_columns(row)
        if stem:
            rows_by_stem[stem] += 1
            if label and grade:
                by_stem[stem].add((label, grade))
    conflicts = sorted(stem for stem, values in by_stem.items() if len(values) > 1)
    class_counts = Counter()
    grade_counts = Counter()
    for stem, values in by_stem.items():
        if len(values) == 1:
            label, grade = next(iter(values))
            class_counts[label] += 1
            grade_counts[grade] += 1
    return {
        "total_rows": len(label_rows),
        "unique_slide_keys": len(rows_by_stem),
        "duplicate_slide_keys": sorted(stem for stem, count in rows_by_stem.items() if count > 1),
        "duplicate_slide_key_count": sum(1 for count in rows_by_stem.values() if count > 1),
        "conflicting_slide_keys": conflicts,
        "conflicting_slide_key_count": len(conflicts),
        "class_vocabulary": sorted(class_counts),
        "class_counts_clean_keys": dict(sorted(class_counts.items())),
        "grade_vocabulary": sorted(grade_counts),
        "grade_counts_clean_keys": dict(sorted(grade_counts.items())),
    }


def build_canonical_labels(
    case_paths: Sequence[CasePath],
    label_rows: Sequence[Mapping[str, Any]],
    label_source_sha256: str,
) -> Tuple[CanonicalLabel, ...]:
    """Join labels to paths, excluding every conflicting workbook key."""

    values_by_stem: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for row in label_rows:
        stem, label, grade = _label_columns(row)
        if stem and label and grade:
            values_by_stem[stem].add((label, grade))
    output: List[CanonicalLabel] = []
    for case in case_paths:
        values = values_by_stem.get(case.slide_stem, set())
        if len(values) != 1:
            continue
        label, grade = next(iter(values))
        output.append(
            CanonicalLabel(
                case_alias=case.case_alias,
                source_code=case.source_code,
                label=label,
                grade=grade,
                family_id=case.family_id,
                label_source_sha256=label_source_sha256,
            )
        )
    return tuple(sorted(output, key=lambda row: row.case_alias))


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _deterministic_rank(seed: int, value: str) -> str:
    return hashlib.sha256((str(seed) + "\0" + str(value)).encode("utf-8")).hexdigest()


def stratified_group_folds(
    rows: Sequence[Any],
    n_splits: int = 5,
    seed: int = 17,
    group_key: str = "family_id",
    label_key: str = "label",
    expected_labels: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Assign groups with sklearn StratifiedGroupKFold and strict validation."""

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    group_labels: Dict[str, Set[str]] = defaultdict(set)
    row_groups = []
    row_labels = []
    for row in rows:
        group = str(_row_value(row, group_key, "")).strip()
        label = str(_row_value(row, label_key, "")).strip()
        if not group or not label:
            raise ValueError("Every split row requires non-empty group and label")
        group_labels[group].add(label)
        row_groups.append(group)
        row_labels.append(label)
    labels = sorted(set(expected_labels or ()) or {label for values in group_labels.values() for label in values})
    if expected_labels is not None and set(labels) != set(expected_labels):
        raise ValueError("Rows do not contain exactly the requested labels")
    support_groups = {label: sum(label in values for values in group_labels.values()) for label in labels}
    insufficient = {label: count for label, count in support_groups.items() if count < n_splits}
    if insufficient:
        raise ValueError(
            "Strict class support is impossible for {0}: fewer than {1} groups".format(
                sorted(insufficient), n_splits
            )
        )

    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception as exc:
        raise RuntimeError("Formal grouped splitting requires scikit-learn: {0}".format(exc))
    dummy = np.zeros(len(rows), dtype=np.uint8)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    assignments: Dict[str, int] = {}
    for fold, (_development, test_indices) in enumerate(
        splitter.split(dummy, np.asarray(row_labels), groups=np.asarray(row_groups))
    ):
        for index in test_indices.tolist():
            group = row_groups[int(index)]
            previous = assignments.get(group)
            if previous is not None and previous != fold:
                raise AssertionError("StratifiedGroupKFold split one group across folds")
            assignments[group] = fold
    group_counts = defaultdict(Counter)
    for label, group in zip(row_labels, row_groups):
        group_counts[group][label] += 1

    def fold_counts():
        counts = [Counter() for _ in range(n_splits)]
        sizes = [0] * n_splits
        for group, fold in assignments.items():
            counts[fold].update(group_counts[group])
            sizes[fold] += sum(group_counts[group].values())
        return counts, sizes

    # SGKF optimizes approximate balance and may omit a class from a fold.
    # Deterministically repair such omissions by moving the least disruptive
    # donor group while preserving every class already present in the donor.
    for _iteration in range(len(group_labels) * len(labels)):
        counts, sizes = fold_counts()
        missing_pairs = [
            (fold, label)
            for fold in range(n_splits)
            for label in labels
            if counts[fold][label] == 0
        ]
        if not missing_pairs:
            return assignments
        target_fold, missing_label = missing_pairs[0]
        candidates = []
        for group, donor_fold in assignments.items():
            if donor_fold == target_fold or group_counts[group][missing_label] == 0:
                continue
            if any(
                counts[donor_fold][label] - count <= 0
                for label, count in group_counts[group].items()
            ):
                continue
            size_penalty = abs((sizes[target_fold] + sum(group_counts[group].values())) - sizes[donor_fold])
            label_penalty = sum(
                abs((counts[target_fold][label] + group_counts[group][label]) - counts[donor_fold][label])
                for label in labels
            )
            candidates.append((size_penalty, label_penalty, _deterministic_rank(seed, group), group))
        if not candidates:
            raise ValueError(
                "StratifiedGroupKFold repair cannot provide {0} in fold {1}".format(
                    missing_label, target_fold
                )
            )
        group = min(candidates)[-1]
        assignments[group] = target_fold
    raise RuntimeError("StratifiedGroupKFold strict-support repair did not converge")


def _split_support(rows: Sequence[CanonicalLabel], aliases: Set[str]) -> Dict[str, int]:
    return dict(sorted(Counter(row.label for row in rows if row.case_alias in aliases).items()))


def build_five_fold_splits(
    labels: Sequence[CanonicalLabel],
    seed: int = 17,
    expected_labels: Optional[Sequence[str]] = None,
) -> Tuple[SplitFold, ...]:
    """Build folds and materialize test=k, val=(k+1)%5, train=others."""

    assignments = stratified_group_folds(
        labels,
        n_splits=5,
        seed=seed,
        expected_labels=expected_labels,
    )
    group_by_case = {row.case_alias: row.family_id for row in labels}
    folds: List[SplitFold] = []
    all_aliases = {row.case_alias for row in labels}
    for fold in range(5):
        test_aliases = {row.case_alias for row in labels if assignments[row.family_id] == fold}
        val_fold = (fold + 1) % 5
        val_aliases = {row.case_alias for row in labels if assignments[row.family_id] == val_fold}
        train_aliases = all_aliases - test_aliases - val_aliases
        if (test_aliases & val_aliases) or (test_aliases & train_aliases) or (val_aliases & train_aliases):
            raise AssertionError("Split alias leakage detected")
        if {group_by_case[alias] for alias in test_aliases} & {group_by_case[alias] for alias in val_aliases}:
            raise AssertionError("Split group leakage detected")
        support = {
            "train": _split_support(labels, train_aliases),
            "val": _split_support(labels, val_aliases),
            "test": _split_support(labels, test_aliases),
        }
        required = set(expected_labels or sorted(set(row.label for row in labels)))
        missing = {split: sorted(required - set(values)) for split, values in support.items() if required - set(values)}
        if missing:
            raise ValueError("Strict seven-class support failed: {0}".format(missing))
        folds.append(
            SplitFold(
                fold_index=fold,
                train_case_aliases=tuple(sorted(train_aliases)),
                val_case_aliases=tuple(sorted(val_aliases)),
                test_case_aliases=tuple(sorted(test_aliases)),
                group_by_case=group_by_case,
                label_support=support,
            )
        )
    return tuple(folds)


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_five_fold_split_artifacts(
    labels: Sequence[CanonicalLabel],
    output_dir: Path,
    eligible_aliases: Optional[Iterable[str]] = None,
    seed: int = 17,
) -> Tuple[SplitFold, ...]:
    """Write the formal YX-only five-fold split after Mucosa eligibility.

    ``eligible_aliases`` is intentionally explicit: the caller must derive it
    from the post-Mucosa 5x manifest (for this baseline, coverage >= 0.60).
    HP cases are inventory-only and are rejected from this writer.
    """

    eligible = None if eligible_aliases is None else {str(value) for value in eligible_aliases}
    selected = [
        row
        for row in labels
        if row.source_code == "YX" and (eligible is None or row.case_alias in eligible)
    ]
    if not selected:
        raise ValueError("No eligible YX canonical labels remain for formal split")
    selected_labels = sorted({row.label for row in selected})
    if len(selected_labels) != 7:
        raise ValueError(
            "Formal YX split requires all seven original workbook type classes; got {0}".format(
                selected_labels
            )
        )
    if eligible is not None:
        by_alias = {row.case_alias: row for row in labels}
        non_yx = sorted(alias for alias in eligible if alias in by_alias and by_alias[alias].source_code != "YX")
        if non_yx:
            raise ValueError("Formal split received non-YX aliases: {0}".format(non_yx[:5]))
    if eligible is not None and len(selected) != len(eligible & {row.case_alias for row in labels}):
        # Unknown aliases are not silently ignored; this catches a mismatch
        # between Mucosa manifest identity and label inventory.
        unknown = sorted(eligible - {row.case_alias for row in labels})
        if unknown:
            raise ValueError("Eligible aliases are absent from canonical labels: {0}".format(unknown[:5]))
    folds = build_five_fold_splits(selected, seed=seed, expected_labels=selected_labels)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    for fold in folds:
        fold_dir = out / "fold_{0}".format(fold.fold_index)
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows_by_split = {
            "train": fold.train_case_aliases,
            "val": fold.val_case_aliases,
            "test": fold.test_case_aliases,
        }
        for split_name, aliases in rows_by_split.items():
            payload = {
                "schema_version": "architecture_baseline_cases_v1",
                "fold_index": fold.fold_index,
                "split": split_name,
                "case_aliases": list(aliases),
                "group_by_case": {alias: fold.group_by_case[alias] for alias in aliases},
                "label_support": fold.label_support[split_name],
            }
            payload["sha256"] = _json_hash(payload)
            _write_json(fold_dir / "{0}_cases.json".format(split_name), payload)
        fold_summary = {
            "fold_index": fold.fold_index,
            "seed": int(seed),
            "case_count": sum(len(values) for values in rows_by_split.values()),
            "label_support": {key: dict(value) for key, value in fold.label_support.items()},
            "case_alias_hash": _json_hash({key: list(value) for key, value in rows_by_split.items()}),
        }
        fold_summary["sha256"] = _json_hash(fold_summary)
        _write_json(fold_dir / "split_summary.json", fold_summary)
        fold_summaries.append(fold_summary)
    summary = {
        "schema_version": "architecture_baseline_five_fold_split_v1",
        "seed": int(seed),
        "source_code": "YX",
        "eligible_alias_count": len(selected),
        "folds": fold_summaries,
    }
    summary["sha256"] = _json_hash(summary)
    class_labels = selected_labels
    class_map = {label: index for index, label in enumerate(class_labels)}
    _write_jsonl(
        out / "training_labels.jsonl",
        (
            {
                **row.to_json(),
                "label_index": class_map[row.label],
            }
            for row in sorted(selected, key=lambda item: item.case_alias)
        ),
    )
    _write_json(out / "class_map.json", class_map)
    summary["training_labels"] = str((out / "training_labels.jsonl").resolve())
    summary["class_map"] = str((out / "class_map.json").resolve())
    summary["sha256"] = _json_hash({key: value for key, value in summary.items() if key != "sha256"})
    _write_json(out / "split_summary.json", summary)
    return folds


def audit_patch_annotations(annotation_rows: Optional[Sequence[Mapping[str, Any]]]) -> AnnotationAudit:
    """Audit reviewed patch/ROI rows without accepting slide-label fallback."""

    rows = list(annotation_rows or ())
    reviewed: List[str] = []
    for row in rows:
        if bool(row.get("synthetic_smoke_only", False)):
            continue
        if str(row.get("label_source", "")).lower() in {"synthetic", "slide_label_broadcast"}:
            continue
        roi_id = str(row.get("roi_id", "")).strip()
        label = _normalise_label(row.get("label") or row.get("architecture_label"))
        architecture = row.get("architecture")
        has_architecture = isinstance(architecture, Mapping) and any(
            str(value).strip().lower() in {"present", "absent"} for value in architecture.values()
        )
        if roi_id and (label or has_architecture):
            reviewed.append(label or "multihead_architecture")
    if not reviewed:
        return AnnotationAudit(
            status="annotation_blocked",
            reason="no_reviewed_patch_or_roi_annotation",
            n_rows=len(rows),
            n_reviewed_rows=0,
            slide_label_broadcast=False,
        )
    return AnnotationAudit(
        status="available",
        reason="reviewed_patch_or_roi_annotation_present",
        n_rows=len(rows),
        n_reviewed_rows=len(reviewed),
        label_vocabulary=tuple(sorted(set(reviewed))),
        slide_label_broadcast=False,
    )


def annotation_guard(
    manifest_rows: Sequence[Mapping[str, Any]],
    annotation_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    slide_labels: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return a safe Baseline-1 target status; never broadcast slide labels."""

    manifest_count = len(manifest_rows)
    del slide_labels  # Explicitly not a fallback source.
    audit = audit_patch_annotations(annotation_rows)
    return {
        "status": audit.status,
        "reason": audit.reason,
        "n_manifest_rows": manifest_count,
        "n_annotation_rows": audit.n_rows,
        "n_reviewed_rows": audit.n_reviewed_rows,
        "slide_label_broadcast": False,
        "targets": [],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def audit_architecture_data(
    label_workbook: Path,
    yx_root: Optional[Path] = None,
    hp_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    annotation_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    five_x_manifest_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    seed: int = 17,
) -> ArchitectureDataAudit:
    """Audit labels/WSI sources and optionally write all baseline artifacts."""

    workbook = Path(label_workbook)
    label_rows = load_xlsx_rows(workbook)
    label_audit = audit_label_rows(label_rows)
    shareable_label_audit = {
        key: value
        for key, value in label_audit.items()
        if key not in ("duplicate_slide_keys", "conflicting_slide_keys")
    }
    source_roots: Dict[str, Path] = {}
    if yx_root is not None:
        source_roots["YX"] = Path(yx_root)
    if hp_root is not None:
        source_roots["HP"] = Path(hp_root)
    case_paths = build_case_paths(source_roots, label_rows=label_rows)
    workbook_sha = _sha256_file(workbook)
    canonical_labels = build_canonical_labels(case_paths, label_rows, workbook_sha)
    class_vocabulary = sorted(set(row.label for row in canonical_labels))
    annotation = audit_patch_annotations(annotation_rows)
    source_summary: Dict[str, Any] = {}
    canonical_by_alias = {row.case_alias: row for row in canonical_labels}
    for source_code in sorted(source_roots):
        source_cases = [case for case in case_paths if case.source_code == source_code]
        source_labels = [canonical_by_alias[case.case_alias] for case in source_cases if case.case_alias in canonical_by_alias]
        source_summary[source_code] = {
            "total_wsi": len(source_cases),
            "matched_clean_labels": sum(case.label_status == "matched" for case in source_cases),
            "conflicting_labels": sum(case.label_status == "conflicting" for case in source_cases),
            "missing_labels": sum(case.label_status == "missing" for case in source_cases),
            "source_formats": dict(Counter(case.source_format for case in source_cases)),
            "family_count": len({case.family_id for case in source_cases}),
            "class_counts": dict(sorted(Counter(row.label for row in source_labels).items())),
            "grade_counts": dict(sorted(Counter(row.grade for row in source_labels).items())),
        }
    summary: Dict[str, Any] = {
        "schema_version": "architecture_baseline_data_audit_v1",
        "label_source": str(workbook.resolve()),
        "label_source_sha256": workbook_sha,
        "labels": shareable_label_audit,
        "sources": source_summary,
        "canonical_label_count": len(canonical_labels),
        "class_vocabulary": class_vocabulary,
        "annotation": annotation.to_json(),
        "five_x_manifest": {
            "provided": five_x_manifest_rows is not None,
            "rows": len(five_x_manifest_rows or ()),
            "slides": len({str(row.get("slide_id", "")) for row in (five_x_manifest_rows or ()) if row.get("slide_id")}),
        },
        "split": {"method": "stratified_group_five_fold", "seed": int(seed), "grouping": "family_id"},
    }
    # Inventory is deliberately not a formal split.  The formal YX split is
    # written only by ``write_five_fold_split_artifacts`` after the caller has
    # validated post-Mucosa >=0.60 eligibility.
    splits: Tuple[SplitFold, ...] = ()
    summary["split"]["status"] = "requires_post_mucosa_eligible_aliases"
    artifact_paths: Dict[str, str] = {}
    if output_dir is not None:
        out = Path(output_dir)
        artifact_paths["case_paths"] = _write_jsonl(out / "local_provenance" / "case_paths.jsonl", (case.to_json() for case in case_paths))
        artifact_paths["label_issues"] = _write_json(
            out / "local_provenance" / "label_issues.json",
            {
                "duplicate_slide_keys": label_audit["duplicate_slide_keys"],
                "conflicting_slide_keys": label_audit["conflicting_slide_keys"],
            },
        )
        artifact_paths["source_roots"] = _write_json(
            out / "local_provenance" / "source_roots.json",
            {key: str(Path(value).resolve()) for key, value in source_roots.items()},
        )
        artifact_paths["canonical_labels"] = _write_jsonl(out / "labels" / "canonical_labels.jsonl", (row.to_json() for row in canonical_labels))
        artifact_paths["audit_summary"] = _write_json(out / "audit_summary.json", summary)
    return ArchitectureDataAudit(
        summary=summary,
        case_paths=tuple(case_paths),
        canonical_labels=tuple(canonical_labels),
        splits=splits,
        annotation=annotation,
        artifact_paths=artifact_paths,
    )
