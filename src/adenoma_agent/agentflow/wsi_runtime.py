"""Real-slide adapters for AgentFlow integration and deterministic smoke cohorts."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from adenoma_agent.agentflow.orchestrator import CropArtifact, ROICropper
from adenoma_agent.wsi import WSIReader, WSI_SUFFIXES


def _safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return token.strip("_") or "unnamed"


def _xlsx_column_index(cell_ref: str) -> int:
    letters = "".join(character for character in str(cell_ref or "") if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return max(0, value - 1)


def load_label_workbook_eligibility(path: Path) -> frozenset:
    """Return eligible slide stems only; no diagnosis labels leave this function."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Label workbook is missing: {0}".format(path))
    spreadsheet_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    relationship_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    office_relationship_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.findall(".//{0}t".format(spreadsheet_ns)))
                for item in root.findall("{0}si".format(spreadsheet_ns))
            ]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("{0}sheets/{0}sheet".format(spreadsheet_ns))
        if sheet is None:
            return frozenset()
        relationship_id = sheet.attrib.get("{0}id".format(office_relationship_ns), "")
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = ""
        for relationship in relationships.findall("{0}Relationship".format(relationship_ns)):
            if relationship.attrib.get("Id") == relationship_id:
                target = relationship.attrib.get("Target", "")
                break
        sheet_path = target if target.startswith("xl/") else "xl/{0}".format(target.lstrip("/"))
        if not target or sheet_path not in names:
            return frozenset()
        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows = []
        for row in sheet_root.findall("{0}sheetData/{0}row".format(spreadsheet_ns)):
            values = []
            for cell in row.findall("{0}c".format(spreadsheet_ns)):
                index = _xlsx_column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                cell_type = cell.attrib.get("t", "")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//{0}t".format(spreadsheet_ns))
                    )
                else:
                    value_node = cell.find("{0}v".format(spreadsheet_ns))
                    value = str(value_node.text or "") if value_node is not None else ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                values[index] = str(value).strip()
            rows.append(values)
    if not rows:
        return frozenset()
    header = [str(value).strip() for value in rows[0]]
    try:
        slide_index = header.index("slide_name")
    except ValueError:
        for candidate in ("case_id", "slide_id"):
            if candidate in header:
                slide_index = header.index(candidate)
                break
        else:
            raise ValueError("Label workbook has no slide_name/case_id/slide_id column")
    return frozenset(
        str(row[slide_index]).strip()
        for row in rows[1:]
        if slide_index < len(row) and str(row[slide_index]).strip()
    )


@dataclass(frozen=True)
class CohortCase:
    case_alias: str
    source_name: str
    source_code: str
    slide_path: Path
    source_format: str

    def inference_payload(self) -> Mapping[str, Any]:
        """The inference-facing identity deliberately excludes path and labels."""

        return {
            "case_alias": self.case_alias,
            "source_code": self.source_code,
            "source_format": self.source_format,
        }

    def local_provenance(self) -> Mapping[str, Any]:
        return {
            "source_name": self.source_name,
            "source_path": str(self.slide_path.resolve()),
        }


def select_deterministic_cohort(
    source_roots: Mapping[str, Path],
    per_source: int = 1,
    eligibility_predicate: Optional[Callable[[str, Path], bool]] = None,
    validation_predicate: Optional[Callable[[str, Path], bool]] = None,
    suffixes_by_source: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[CohortCase, ...]:
    """Select sorted first-N eligible slides per source without retaining labels.

    ``eligibility_predicate`` may consult a private label table, but only its
    boolean result is used.  Labels never become part of ``CohortCase``.
    """

    count = int(per_source)
    if count <= 0:
        raise ValueError("per_source must be positive")
    selected = []
    for source_index, source_name in enumerate(sorted(source_roots), 1):
        source_code = "SOURCE_{0:03d}".format(source_index)
        root = Path(source_roots[source_name]).resolve()
        if not root.is_dir():
            raise FileNotFoundError("Cohort source directory is missing: {0}".format(root))
        configured = (suffixes_by_source or {}).get(source_name)
        allowed = {
            str(value).lower() if str(value).startswith(".") else ".{0}".format(str(value).lower())
            for value in (configured or WSI_SUFFIXES)
        }
        eligible = []
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            if eligibility_predicate is not None and not bool(eligibility_predicate(source_name, path)):
                continue
            if validation_predicate is not None and not bool(validation_predicate(source_name, path)):
                continue
            eligible.append(path)
            if len(eligible) >= count:
                break
        for index, path in enumerate(eligible, 1):
            digest = hashlib.sha256(
                "{0}\0{1}".format(source_name, path.name).encode("utf-8")
            ).hexdigest()[:10]
            alias = "CASE_{0:03d}_{1:03d}_{2}".format(source_index, index, digest)
            selected.append(
                CohortCase(
                    case_alias=alias,
                    source_name=str(source_name),
                    source_code=source_code,
                    slide_path=path.resolve(),
                    source_format=path.suffix.lower().lstrip("."),
                )
            )
    return tuple(selected)


def integration_boundary_payload(
    cohort: Sequence[CohortCase],
    case_results: Mapping[str, Mapping[str, Any]],
    local_provenance_refs: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    rows = []
    for case in cohort:
        result = dict(case_results.get(case.case_alias, {}))
        dependencies = list(result.get("dependency_blocked", []))
        rows.append(
            {
                "case_alias": case.case_alias,
                "source_code": case.source_code,
                "source_format": case.source_format,
                "inference_identity": dict(case.inference_payload()),
                "status": result.get("status", "not_started"),
                "stages_reached": list(result.get("stages_reached", [])),
                "integration_boundary": result.get("integration_boundary", "source_discovery"),
                "reached_stage": (
                    list(result.get("stages_reached", []))[-1]
                    if result.get("stages_reached")
                    else "not_started"
                ),
                "blocking_dependency": [
                    item.get("code", "unknown_dependency")
                    for item in dependencies
                ],
                "recoverable_action": list(
                    result.get(
                        "recoverable_action",
                        [item.get("message", "") for item in dependencies if item.get("message")],
                    )
                ),
                "exit_status": result.get("status", "not_started"),
                "dependency_blocked": dependencies,
                "artifacts": dict(result.get("artifacts", {})),
                "local_provenance_ref": str(
                    (local_provenance_refs or {}).get(
                        case.case_alias,
                        "local_provenance/{0}.json".format(case.case_alias),
                    )
                ),
            }
        )
    return {
        "schema_version": "agentflow_real_smoke_boundary_v1",
        "selection": {
            "method": "sorted_first_n_eligible_per_source",
            "randomized": False,
            "labels_available_to_inference": False,
        },
        "cases": rows,
    }


def write_integration_boundary(
    path: Path,
    cohort: Sequence[CohortCase],
    case_results: Mapping[str, Mapping[str, Any]],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance_refs = {}
    provenance_dir = path.parent / "local_provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    for case in cohort:
        provenance_path = provenance_dir / "{0}.json".format(case.case_alias)
        provenance_path.write_text(
            json.dumps(
                {
                    "schema_version": "agentflow_local_source_provenance_v1",
                    "case_alias": case.case_alias,
                    **dict(case.local_provenance()),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        provenance_refs[case.case_alias] = str(provenance_path.relative_to(path.parent))
    path.write_text(
        json.dumps(
            integration_boundary_payload(
                cohort,
                case_results,
                local_provenance_refs=provenance_refs,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class WSIROICropper(ROICropper):
    """Materialize Planner-selected ROIs directly from one real WSI."""

    synthetic_stub = False

    def __init__(
        self,
        slide_path: Path,
        case_alias: str,
        base_magnification: Optional[float] = None,
        mpp: Any = None,
        output_pixels: Optional[Any] = None,
    ):
        self.case_alias = _safe_token(case_alias)
        self.reader = WSIReader(
            slide_path,
            base_magnification=base_magnification,
            mpp=mpp,
            allow_raster_fixture=True,
        )
        self.reader.require_physical_metadata()
        self.output_pixels = output_pixels
        self.crop_events = []

    @property
    def slide_dimensions(self) -> Tuple[int, int]:
        return tuple(self.reader.dimensions)

    def crop(self, roi, output_dir):
        if output_dir is None:
            raise ValueError("WSIROICropper requires an output directory for provenance")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        crop = self.reader.crop_level0_bbox(
            roi.level0_bbox,
            requested_magnification=float(roi.scale),
            output_pixels=self.output_pixels,
        )
        stem = "{0}__{1}".format(self.case_alias, _safe_token(roi.roi_id))
        image_path = crop.save(output_dir / "{0}.png".format(stem))
        public_provenance = dict(crop.provenance)
        local_source_path = public_provenance.pop("local_source_path", None)
        local_sidecar_path = output_dir / "{0}.local.provenance.json".format(stem)
        local_sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": "agentflow_wsi_roi_local_provenance_v1",
                    "case_alias": self.case_alias,
                    "roi_id": roi.roi_id,
                    "source_path": local_source_path,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        sidecar = {
            "schema_version": "agentflow_wsi_roi_crop_v1",
            "case_alias": self.case_alias,
            "roi_id": roi.roi_id,
            "slide_id": roi.slide_id,
            "crop": public_provenance,
            "local_provenance_ref": local_sidecar_path.name,
        }
        sidecar_path = output_dir / "{0}.provenance.json".format(stem)
        sidecar_path.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        event = {
            "case_alias": self.case_alias,
            "roi_id": roi.roi_id,
            "image_ref": str(image_path.resolve()),
            "provenance_ref": str(sidecar_path.resolve()),
            "image_sha256": crop.image_sha256,
        }
        self.crop_events.append(event)
        output_mpp = public_provenance["output_mpp"]
        return CropArtifact(
            image_ref=str(image_path.resolve()),
            image_sha256=crop.image_sha256,
            pixel_dimensions=crop.pixel_dimensions,
            mpp=(float(output_mpp[0]) + float(output_mpp[1])) / 2.0,
            hash_scope="png_file_content",
        )

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "WSIROICropper":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
