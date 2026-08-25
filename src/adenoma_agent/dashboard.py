import json
import shutil
import re
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from zipfile import ZipFile

from adenoma_agent.trace_supervision import (
    GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN,
    GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN,
    GLOBAL_SCREENING_SINGLE_MODEL_STATUS,
    TRACE_LABEL_COLORS,
    TRACE_LABEL_RUBRIC,
    _looks_like_dual_model_global_screening_patch,
    _normalize_patch_id,
    derive_global_screening_fusion,
    expected_global_screening_priority,
    expected_global_screening_require_high_magnification,
)
from adenoma_agent.utils import ensure_dir, read_json, read_jsonl, write_json, write_text


LABEL_ORDER = [
    "ssl_suspicious_mucosa",
    "conventional_adenoma_like",
    "inflammatory_polyp_like",
    "normal_mucosa",
    "background_artifact_stroma",
]

LABEL_SHORT = {
    "ssl_suspicious_mucosa": "SSL 可疑",
    "conventional_adenoma_like": "传统腺瘤样",
    "inflammatory_polyp_like": "炎性样",
    "normal_mucosa": "正常黏膜",
    "background_artifact_stroma": "背景/伪影",
    "unknown": "未知",
}

ERROR_LABELS = {
    "missing_patch": "缺失 Patch",
    "duplicate_patch": "重复 Patch",
    "orphan_patch": "孤儿 Patch",
    "illegal_patch_id": "非法 Patch ID",
    "out_of_bounds_patch": "越界 Patch",
}

AGREEMENT_DISPLAY = {
    "strong_agreement": "强一致",
    "risk_disagreement": "高风险分歧",
    "low_value_disagreement": "低价值分歧",
    "single_model_trace": "单模型 Trace",
}

SCORE_ORIGIN_DISPLAY = {
    "native_score": "原生分数",
    "normalized_diagnostic_priority": "优先级归一化分数",
    "consensus_fusion": "共识融合分数",
}

VALIDATION_STATUS_DISPLAY = {
    "ok": "正常",
    "warning": "警告",
}

BRANCH_STATE_DISPLAY = {
    "supported": {"label": "支持", "icon": "✅", "class_name": "supported"},
    "opposed": {"label": "排除", "icon": "❌", "class_name": "opposed"},
    "unresolved": {"label": "未决", "icon": "●", "class_name": "unresolved"},
}

NAVIGATION_ACTION_DISPLAY = {
    "inspect": "Inspect",
    "stop": "Stop",
}

NAVIGATION_MAG_DISPLAY = {
    "5x": "5x",
    "10x": "10x",
    "20x": "20x",
    "unknown": "未知倍率",
}

NAVIGATION_MAG_COLORS = {
    "5x": {"stroke": "#2563eb", "fill": "rgba(37, 99, 235, 0.22)"},
    "10x": {"stroke": "#e11d48", "fill": "rgba(225, 29, 72, 0.22)"},
    "20x": {"stroke": "#dc2626", "fill": "rgba(220, 38, 38, 0.22)"},
    "unknown": {"stroke": "#64748b", "fill": "rgba(100, 116, 139, 0.2)"},
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABEL_XLSX_PATH = REPO_ROOT / "data" / "label" / "Adenoma_filtered.xlsx"

LABEL_FAMILY_DISPLAY = {
    "serrated": "锯齿状病变家族",
    "conventional": "传统腺瘤家族",
    "inflammatory": "炎性息肉",
    "normal": "正常 / 低价值背景",
    "unknown": "未知",
}

LABEL_GRADE_DISPLAY = {
    "low": "低级别 / low",
    "high": "高级别 / high",
    "unknown": "未知",
}


def _xlsx_column_index(cell_ref):
    letters = "".join(ch for ch in str(cell_ref or "") if ch.isalpha())
    if not letters:
        return 0
    value = 0
    for char in letters.upper():
        value = value * 26 + ord(char) - ord("A") + 1
    return max(0, value - 1)


def _xlsx_text_from_shared_string(shared_string_node, namespace):
    return "".join(node.text or "" for node in shared_string_node.findall(".//{0}t".format(namespace)))


def _xlsx_cell_value(cell, shared_strings, namespace):
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//{0}t".format(namespace))).strip()
    value_node = cell.find("{0}v".format(namespace))
    if value_node is None:
        return ""
    raw_value = str(value_node.text or "").strip()
    if cell_type == "s":
        try:
            return str(shared_strings[int(raw_value)]).strip()
        except Exception:
            return raw_value
    return raw_value


def _xlsx_first_sheet_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_namespace = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    office_rel_namespace = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        shared_strings = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [_xlsx_text_from_shared_string(item, namespace) for item in shared_root.findall("{0}si".format(namespace))]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_node = workbook.find("{0}sheets/{0}sheet".format(namespace))
        if sheet_node is None:
            return []
        relation_id = sheet_node.attrib.get("{0}id".format(office_rel_namespace), "")
        relation_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_target = ""
        for relation in relation_root.findall("{0}Relationship".format(rel_namespace)):
            if relation.attrib.get("Id") == relation_id:
                relation_target = relation.attrib.get("Target", "")
                break
        if not relation_target:
            return []
        sheet_path = relation_target if relation_target.startswith("xl/") else "xl/{0}".format(relation_target.lstrip("/"))
        if sheet_path not in names:
            return []

        sheet_root = ET.fromstring(archive.read(sheet_path))
        rows = []
        for row in sheet_root.findall("{0}sheetData/{0}row".format(namespace)):
            values = []
            for cell in row.findall("{0}c".format(namespace)):
                index = _xlsx_column_index(cell.attrib.get("r", "A1"))
                while len(values) <= index:
                    values.append("")
                values[index] = _xlsx_cell_value(cell, shared_strings, namespace)
            rows.append(values)
    return rows


def _read_ground_truth_label_map(label_path=None):
    label_path = Path(label_path or DEFAULT_LABEL_XLSX_PATH)
    cache_key = str(label_path.resolve()) if label_path.exists() else str(label_path)
    cache = getattr(_read_ground_truth_label_map, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    result = {
        "source_path": str(label_path),
        "rows": {},
        "error": "",
    }
    try:
        rows = _xlsx_first_sheet_rows(label_path)
        if not rows:
            result["error"] = "label 文件为空或无法解析"
        else:
            header = [str(item or "").strip() for item in rows[0]]
            for row in rows[1:]:
                record = {}
                for index, name in enumerate(header):
                    if name:
                        record[name] = str(row[index] if index < len(row) else "").strip()
                slide_name = str(record.get("slide_name") or record.get("case_id") or record.get("slide_id") or "").strip()
                if slide_name:
                    result["rows"][slide_name] = record
    except Exception as exc:
        result["error"] = str(exc)

    cache[cache_key] = result
    _read_ground_truth_label_map._cache = cache
    return result


def _normalize_truth_family(label_type):
    text = str(label_type or "").strip().lower()
    if not text:
        return "unknown"
    if "inflammatory" in text:
        return "inflammatory"
    if "tubular" in text or "tubulovillous" in text or "villous" in text:
        return "conventional"
    if "serrated" in text or "hyperplastic" in text:
        return "serrated"
    if "normal" in text or "background" in text:
        return "normal"
    return "unknown"


def _normalize_truth_grade(grade):
    text = str(grade or "").strip().lower()
    if not text:
        return "unknown"
    if "high" in text:
        return "high"
    if "low" in text:
        return "low"
    return "unknown"


def _ground_truth_label_for_case(case_id, label_path=None):
    label_map = _read_ground_truth_label_map(label_path)
    record = dict(label_map.get("rows", {}).get(str(case_id), {}))
    if not record:
        return {
            "available": False,
            "case_id": str(case_id),
            "source_path": label_map.get("source_path", ""),
            "error": label_map.get("error", ""),
            "raw": {},
            "type": "",
            "grade": "",
            "family": "unknown",
            "family_display": LABEL_FAMILY_DISPLAY["unknown"],
            "grade_display": LABEL_GRADE_DISPLAY["unknown"],
            "summary": "未在 label 文件中找到该 case",
        }
    family = _normalize_truth_family(record.get("type"))
    grade = _normalize_truth_grade(record.get("grade"))
    summary = "{0}；grade={1}".format(record.get("type", "-"), record.get("grade", "-"))
    return {
        "available": True,
        "case_id": str(case_id),
        "source_path": label_map.get("source_path", ""),
        "error": label_map.get("error", ""),
        "raw": record,
        "type": str(record.get("type", "")).strip(),
        "grade": str(record.get("grade", "")).strip(),
        "family": family,
        "family_display": LABEL_FAMILY_DISPLAY.get(family, family),
        "grade_normalized": grade,
        "grade_display": LABEL_GRADE_DISPLAY.get(grade, str(record.get("grade", ""))),
        "summary": summary,
    }


def _prediction_text_from_sources(observe_report, case_result):
    observe_report = observe_report if isinstance(observe_report, dict) else {}
    case_result = case_result if isinstance(case_result, dict) else {}
    chunks = []
    for source in (observe_report, case_result):
        prediction = source.get("hierarchical_prediction") if isinstance(source, dict) else {}
        if isinstance(prediction, dict):
            chunks.append(json.dumps(prediction, ensure_ascii=False))
        integrated = source.get("integrated_report") if isinstance(source, dict) else ""
        if isinstance(integrated, dict):
            chunks.append(str(integrated.get("summary", "")))
            chunks.append(json.dumps(integrated.get("recommendations", []), ensure_ascii=False))
        else:
            chunks.append(str(integrated or ""))
    return "\n".join(chunk for chunk in chunks if str(chunk or "").strip())


def _prediction_families_from_text(text):
    lowered = str(text or "").lower()
    families = []
    if any(token in lowered for token in ["serrated", "ssl", "sessile serrated", "hyperplastic", "锯齿"]):
        families.append("serrated")
    if any(token in lowered for token in ["conventional", "tubular", "tubulovillous", "villous", "adenoma", "腺瘤"]):
        families.append("conventional")
    if any(token in lowered for token in ["inflammatory", "炎性"]):
        families.append("inflammatory")
    if any(token in lowered for token in ["normal mucosa", "normal", "benign", "正常"]):
        families.append("normal")
    return list(dict.fromkeys(families))


def _prediction_grade_from_text(text):
    lowered = str(text or "").lower()
    if re.search(r"high[- ]?grade|high grade|高级别", lowered):
        return "high"
    if re.search(r"low[- ]?grade|low grade|低级别", lowered):
        return "low"
    return "unknown"


def _compare_final_report_with_ground_truth(ground_truth, observe_report, case_result):
    ground_truth = ground_truth if isinstance(ground_truth, dict) else {}
    if not ground_truth.get("available"):
        return {
            "status": "missing_label",
            "status_display": "缺少真实标签",
            "truth_summary": ground_truth.get("summary", "未找到真实标签"),
            "prediction_summary": "",
            "predicted_families": [],
            "predicted_family_display": "报告未明确",
            "predicted_grade": "unknown",
            "items": [],
        }

    prediction_text = _prediction_text_from_sources(observe_report, case_result)
    predicted_families = _prediction_families_from_text(prediction_text)
    predicted_grade = _prediction_grade_from_text(prediction_text)
    truth_family = ground_truth.get("family", "unknown")
    truth_grade = ground_truth.get("grade_normalized", "unknown")
    family_match = truth_family != "unknown" and truth_family in predicted_families
    grade_match = None
    if truth_grade != "unknown" and predicted_grade != "unknown":
        grade_match = truth_grade == predicted_grade

    if not predicted_families:
        status = "no_prediction"
        status_display = "报告未明确病变家族"
    elif family_match and grade_match is not False:
        status = "match" if grade_match is True else "family_match"
        status_display = "家族匹配" if grade_match is None else "家族与 grade 匹配"
    elif family_match:
        status = "partial"
        status_display = "家族匹配，grade 不一致"
    else:
        status = "mismatch"
        status_display = "家族不匹配"

    predicted_family_display = "、".join(LABEL_FAMILY_DISPLAY.get(item, item) for item in predicted_families) or "报告未明确"
    return {
        "status": status,
        "status_display": status_display,
        "truth_summary": ground_truth.get("summary", ""),
        "prediction_summary": prediction_text[:600],
        "predicted_families": predicted_families,
        "predicted_family_display": predicted_family_display,
        "predicted_grade": predicted_grade,
        "predicted_grade_display": LABEL_GRADE_DISPLAY.get(predicted_grade, predicted_grade),
        "items": [
            {
                "key": "lesion_family",
                "label": "病变家族",
                "truth": ground_truth.get("family_display", ""),
                "prediction": predicted_family_display,
                "match": bool(family_match),
            },
            {
                "key": "grade",
                "label": "Grade",
                "truth": ground_truth.get("grade_display", ""),
                "prediction": LABEL_GRADE_DISPLAY.get(predicted_grade, "报告未明确"),
                "match": grade_match,
            },
        ],
    }


def _default_trace_rubric_summary():
    summary = []
    for label in LABEL_ORDER:
        rubric = TRACE_LABEL_RUBRIC.get(label, {})
        color = TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])
        summary.append(
            {
                "region_semantic": label,
                "display_name": LABEL_SHORT.get(label, label),
                "default_priority": rubric.get("default_priority"),
                "require_high_magnification": rubric.get("default_high_mag"),
                "color_fill": color["fill"],
                "color_stroke": color["stroke"],
            }
        )
    return summary


def _score_from_priority(priority, max_priority=4):
    priority = _safe_int(priority, 0)
    max_priority = max(1, _safe_int(max_priority, 4))
    if priority <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(priority) / float(max_priority))), 3)


def _agreement_uncertainty(agreement_status):
    return {
        "strong_agreement": 0.12,
        "risk_disagreement": 0.78,
        "low_value_disagreement": 0.36,
        GLOBAL_SCREENING_SINGLE_MODEL_STATUS: 0.5,
    }.get(str(agreement_status or "").strip(), 0.5)


def _build_dashboard_patch(raw_patch, cell, cluster, box):
    patch = dict(raw_patch or {})
    row_col = _normalize_patch_id(patch.get("patch_id") or [patch.get("row"), patch.get("col")])
    if row_col is None and cell is not None:
        row_col = _normalize_patch_id([cell.get("row_id"), cell.get("col_id")])
    if row_col is not None:
        patch["patch_id"] = [int(row_col[0]), int(row_col[1])]
        patch.setdefault("row", int(row_col[0]))
        patch.setdefault("col", int(row_col[1]))
    if cell is not None and (not patch.get("bbox_level0") or not isinstance(patch.get("bbox_level0"), list)):
        patch["bbox_level0"] = _grid_cell_bbox_level0(cell)
    cluster_label = str((cluster or {}).get("l", patch.get("region_semantic", "")) or "unknown")
    conch_label = str(patch.get("conch_region_semantic", "")).strip()
    patho_label = str(patch.get("pathoreasoner_r1_region_semantic", "")).strip()
    dual_mode = _looks_like_dual_model_global_screening_patch(patch)
    if dual_mode:
        expected = derive_global_screening_fusion(conch_label, patho_label)
        if expected:
            if not str(patch.get("agreement_status", "")).strip():
                patch["agreement_status"] = expected["agreement_status"]
            if not str(patch.get("score_origin", "")).strip():
                patch["score_origin"] = expected["score_origin"]
            if not str(patch.get("fusion_reasoning", "")).strip():
                patch["fusion_reasoning"] = expected["fusion_reasoning"]
            if "diagnostic_priority" not in patch or patch.get("diagnostic_priority") in (None, ""):
                patch["diagnostic_priority"] = expected["diagnostic_priority"]
            if "require_high_magnification" not in patch:
                patch["require_high_magnification"] = expected["require_high_magnification"]
            if not str(patch.get("region_semantic", "")).strip():
                patch["region_semantic"] = expected["region_semantic"]
            if not str(patch.get("conch_region_semantic", "")).strip():
                patch["conch_region_semantic"] = expected["conch_region_semantic"]
            if not str(patch.get("pathoreasoner_r1_region_semantic", "")).strip():
                patch["pathoreasoner_r1_region_semantic"] = expected["pathoreasoner_r1_region_semantic"]
    else:
        if not str(patch.get("agreement_status", "")).strip():
            patch["agreement_status"] = GLOBAL_SCREENING_SINGLE_MODEL_STATUS
        if not str(patch.get("score_origin", "")).strip():
            patch["score_origin"] = GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN
        if not str(patch.get("conch_region_semantic", "")).strip():
            patch["conch_region_semantic"] = "not_available_in_this_run"
        if not str(patch.get("pathoreasoner_r1_region_semantic", "")).strip():
            patch["pathoreasoner_r1_region_semantic"] = str(patch.get("region_semantic", cluster_label) or cluster_label)
        if not str(patch.get("fusion_reasoning", "")).strip():
            patch["fusion_reasoning"] = "Single-model trace output expanded from cluster-level assignment."
    if not str(patch.get("region_semantic", "")).strip():
        patch["region_semantic"] = cluster_label
    patch["cluster_id"] = str(patch.get("cluster_id", "") or (cluster or {}).get("cluster_id", ""))
    patch["review_stage"] = str(patch.get("review_stage", "") or (cluster or {}).get("review_stage", ""))
    patch["diagnostic_priority"] = _safe_int(
        patch.get("diagnostic_priority"),
        _safe_int((cluster or {}).get("s"), 0),
    )
    if "require_high_magnification" not in patch:
        patch["require_high_magnification"] = bool((cluster or {}).get("d", False))
    if box:
        patch.setdefault("tissue_coverage_score", box.get("score"))
    if "score" not in patch or patch.get("score") is None:
        patch["score"] = _score_from_priority(patch["diagnostic_priority"], 5 if dual_mode else 4)
    if "uncertainty_score" not in patch or patch.get("uncertainty_score") is None:
        patch["uncertainty_score"] = _agreement_uncertainty(patch.get("agreement_status"))
    if not str(patch.get("score_origin", "")).strip():
        patch["score_origin"] = GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN if dual_mode else GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN
    return patch


def build_demo_dashboard_payload():
    grid_rows = 5
    grid_cols = 5
    thumbnail_size = [1000, 1000]
    cell_size = 180
    gap = 12
    grid_cells = []
    selected_patch_ids = []
    patch_assignments = []
    high_mag_assets = []
    label_layout = [
        ["background_artifact_stroma", "normal_mucosa", "ssl_suspicious_mucosa", "ssl_suspicious_mucosa", "background_artifact_stroma"],
        ["normal_mucosa", "conventional_adenoma_like", "ssl_suspicious_mucosa", "inflammatory_polyp_like", "background_artifact_stroma"],
        ["normal_mucosa", "conventional_adenoma_like", "conventional_adenoma_like", "inflammatory_polyp_like", "background_artifact_stroma"],
        ["background_artifact_stroma", "normal_mucosa", "inflammatory_polyp_like", "normal_mucosa", "background_artifact_stroma"],
        ["background_artifact_stroma", "background_artifact_stroma", "normal_mucosa", "background_artifact_stroma", "background_artifact_stroma"],
    ]
    score_layout = [
        [0.08, 0.24, 0.92, 0.52, 0.06],
        [0.31, 0.81, 0.57, 0.63, 0.12],
        [0.45, 0.78, 0.74, 0.41, 0.07],
        [0.09, 0.34, 0.49, 0.29, 0.11],
        [0.04, 0.05, 0.22, 0.03, 0.02],
    ]
    uncertainty_layout = [
        [0.04, 0.16, 0.18, 0.79, 0.05],
        [0.42, 0.27, 0.83, 0.32, 0.08],
        [0.58, 0.24, 0.21, 0.77, 0.09],
        [0.05, 0.33, 0.66, 0.28, 0.06],
        [0.03, 0.04, 0.14, 0.03, 0.02],
    ]
    agreement_layout = [
        ["strong_agreement", "strong_agreement", "strong_agreement", "risk_disagreement", "strong_agreement"],
        ["low_value_disagreement", "strong_agreement", "risk_disagreement", "strong_agreement", "strong_agreement"],
        ["low_value_disagreement", "strong_agreement", "strong_agreement", "risk_disagreement", "strong_agreement"],
        ["strong_agreement", "strong_agreement", "risk_disagreement", "strong_agreement", "strong_agreement"],
        ["strong_agreement", "strong_agreement", "strong_agreement", "strong_agreement", "strong_agreement"],
    ]
    patho_layout = [
        ["background_artifact_stroma", "normal_mucosa", "ssl_suspicious_mucosa", "conventional_adenoma_like", "background_artifact_stroma"],
        ["normal_mucosa", "conventional_adenoma_like", "conventional_adenoma_like", "inflammatory_polyp_like", "background_artifact_stroma"],
        ["background_artifact_stroma", "conventional_adenoma_like", "conventional_adenoma_like", "normal_mucosa", "background_artifact_stroma"],
        ["background_artifact_stroma", "normal_mucosa", "normal_mucosa", "normal_mucosa", "background_artifact_stroma"],
        ["background_artifact_stroma", "background_artifact_stroma", "normal_mucosa", "background_artifact_stroma", "background_artifact_stroma"],
    ]

    for row in range(grid_rows):
        for col in range(grid_cols):
            x = gap + col * (cell_size + gap)
            y = gap + row * (cell_size + gap)
            grid_cells.append(
                {
                    "patch_id": [row, col],
                    "row_id": row,
                    "col_id": col,
                    "is_selected": True,
                    "thumbnail_top_left_x": x,
                    "thumbnail_top_left_y": y,
                    "thumbnail_width": cell_size,
                    "thumbnail_height": cell_size,
                }
            )
            selected_patch_ids.append([row, col])
            label = label_layout[row][col]
            score = score_layout[row][col]
            uncertainty = uncertainty_layout[row][col]
            agreement = agreement_layout[row][col]
            conch_label = label
            patho_label = patho_layout[row][col]
            fusion_reasoning = "Stable morphology consensus."
            if agreement == "risk_disagreement":
                if label == "ssl_suspicious_mucosa":
                    fusion_reasoning = "serrated vs conventional conflict around crypt contour and surface maturation."
                elif label == "inflammatory_polyp_like":
                    fusion_reasoning = "lesion-positive vs background disagreement in low-signal tissue edge."
                else:
                    fusion_reasoning = "discordant gland pattern review required."
            diagnostic_priority = expected_global_screening_priority(label, agreement)
            patch_assignments.append(
                {
                    "patch_id": [row, col],
                    "row": row,
                    "col": col,
                    "bbox_level0": [col * 256, row * 256, (col + 1) * 256, (row + 1) * 256],
                    "score": score,
                    "uncertainty_score": uncertainty,
                    "region_semantic": label,
                    "diagnostic_priority": diagnostic_priority,
                    "require_high_magnification": expected_global_screening_require_high_magnification(label, agreement),
                    "agreement_status": agreement,
                    "conch_region_semantic": conch_label,
                    "pathoreasoner_r1_region_semantic": patho_label,
                    "fusion_reasoning": fusion_reasoning,
                    "score_origin": GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN if agreement != GLOBAL_SCREENING_SINGLE_MODEL_STATUS else GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN,
                    "high_mag_ref": "patch_{0}_{1}_20x".format(row, col),
                }
            )
            high_mag_assets.append(
                {
                    "asset_id": "patch_{0}_{1}_20x".format(row, col),
                    "patch_id": [row, col],
                    "magnification": 20.0,
                    "image_url": "",
                    "width": 1024,
                    "height": 1024,
                    "data_label": label,
                }
            )

    return {
        "case_id": "demo_case_ssl_001",
        "slide_id": "demo_slide_a",
        "selected_patch_ids": selected_patch_ids,
        "grid_metadata": {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "patch_size": 256,
            "overview_magnification": 5.0,
            "tile_stride": 128,
            "selected_patch_count": len(selected_patch_ids),
            "generated_at": "2026-06-01T12:00:00Z",
            "model_version": "global_screening_v2_demo",
            "cropped_thumbnail_size": thumbnail_size,
            "grid_cells": grid_cells,
        },
        "trace_rubric": {
            "mode": "dynamic_priority_consensus",
            "labels": _default_trace_rubric_summary(),
        },
        "overview_image": {
            "image_url": "",
            "width": thumbnail_size[0],
            "height": thumbnail_size[1],
        },
        "patch_assignments": patch_assignments,
        "high_mag_assets": high_mag_assets,
        "audit_log": [
            {
                "audit_id": "audit_0001",
                "patch_id": [0, 3],
                "original_region_semantic": "ssl_suspicious_mucosa",
                "corrected_region_semantic": "conventional_adenoma_like",
                "original_diagnostic_priority": 4,
                "corrected_diagnostic_priority": 3,
                "original_require_high_magnification": True,
                "corrected_require_high_magnification": True,
                "operator": "reviewer_demo",
                "comment": "initial review leaned more conventional",
                "created_at": "2026-06-01T12:34:56Z",
            }
        ],
        "contract_validation": {
            "status": "warning",
            "violations": [
                {
                    "type": "missing_patch",
                    "patch_id": [4, 4],
                    "message": "Patch is selected but omitted in candidate payload.",
                },
                {
                    "type": "out_of_bounds_patch",
                    "patch_id": [6, 1],
                    "row": 6,
                    "col": 1,
                    "message": "Patch coordinates exceed grid boundary.",
                },
            ],
        },
        "orphan_assets": [
            {
                "patch_id": [6, 1],
                "reason": "out_of_bounds_patch",
                "score": 0.52,
                "region_semantic": "conventional_adenoma_like",
                "preview_ref": "orphan_patch_6_1_20x",
            }
        ],
        "trace_clusters": [
            {
                "cluster_id": "grid_group_00",
                "cluster_label": "ssl_suspicious_mucosa",
                "cluster_priority": 4,
                "workflow_branch": "serrated",
                "require_high_magnification": True,
            },
            {
                "cluster_id": "grid_group_01",
                "cluster_label": "conventional_adenoma_like",
                "cluster_priority": 3,
                "workflow_branch": "conventional",
                "require_high_magnification": True,
            },
            {
                "cluster_id": "grid_group_02",
                "cluster_label": "inflammatory_polyp_like",
                "cluster_priority": 2,
                "workflow_branch": "non_serrated",
                "require_high_magnification": False,
            },
        ],
        "navigation_steps": [
            {
                "step_id": "step_00",
                "x": 640,
                "y": 180,
                "m": "5x",
                "action": "inspect",
                "region_size_level0": 2048,
                "need_to_see": "serrated mucosal context",
                "review_goal": "serrated_lesion_assessment",
                "stage_gate": "mucosa_or_serrated",
                "metadata": {
                    "cluster_id": "grid_group_00",
                    "cluster_label": "ssl_suspicious_mucosa",
                    "cluster_priority": 4,
                    "workflow_branch": "serrated",
                    "patch_id": [0, 2],
                    "action": "inspect",
                },
            },
            {
                "step_id": "step_01",
                "x": 640,
                "y": 360,
                "m": "20x",
                "action": "inspect",
                "region_size_level0": 512,
                "need_to_see": "basal crypt serration",
                "review_goal": "abnormal_crypt_assessment",
                "stage_gate": "abnormal_crypt",
                "metadata": {
                    "cluster_id": "grid_group_00",
                    "cluster_label": "ssl_suspicious_mucosa",
                    "cluster_priority": 4,
                    "workflow_branch": "serrated",
                    "patch_id": [1, 2],
                    "action": "inspect",
                },
            },
            {
                "step_id": "step_02",
                "x": 400,
                "y": 360,
                "m": "5x",
                "action": "inspect",
                "region_size_level0": 2048,
                "need_to_see": "adenomatous gland crowding",
                "review_goal": "conventional_adenoma_assessment",
                "stage_gate": "conventional_adenoma",
                "metadata": {
                    "cluster_id": "grid_group_01",
                    "cluster_label": "conventional_adenoma_like",
                    "cluster_priority": 3,
                    "workflow_branch": "conventional",
                    "patch_id": [1, 1],
                    "action": "inspect",
                },
            },
            {
                "step_id": "step_03",
                "x": 760,
                "y": 540,
                "m": "5x",
                "action": "inspect",
                "region_size_level0": 2048,
                "need_to_see": "reactive mucosal context",
                "review_goal": "non_serrated_overview_assessment",
                "stage_gate": "non_serrated_context",
                "metadata": {
                    "cluster_id": "grid_group_02",
                    "cluster_label": "inflammatory_polyp_like",
                    "cluster_priority": 2,
                    "workflow_branch": "non_serrated",
                    "patch_id": [2, 3],
                    "action": "inspect",
                },
            },
            {
                "step_id": "step_04",
                "x": 760,
                "y": 540,
                "m": "5x",
                "action": "stop",
                "region_size_level0": 2048,
                "need_to_see": "stop navigation and summarize",
                "review_goal": "integrated_impression",
                "stage_gate": "end",
                "metadata": {
                    "action": "stop",
                },
            },
        ],
        "observe_assets_by_step": [
            {
                "step_id": "step_00",
                "cluster_id": "grid_group_00",
                "review_goal": "serrated_lesion_assessment",
                "need_to_see": "serrated mucosal context",
                "stage_gate": "mucosa_or_serrated",
                "overview_image": _high_mag_svg("ssl overview", "[0,2]"),
                "local_image": _high_mag_svg("ssl local", "[0,2]"),
                "detail_image": _high_mag_svg("ssl detail", "[0,2]"),
            },
            {
                "step_id": "step_01",
                "cluster_id": "grid_group_00",
                "review_goal": "abnormal_crypt_assessment",
                "need_to_see": "basal crypt serration",
                "stage_gate": "abnormal_crypt",
                "overview_image": _high_mag_svg("ssl overview", "[1,2]"),
                "local_image": _high_mag_svg("ssl local", "[1,2]"),
                "detail_image": _high_mag_svg("ssl detail", "[1,2]"),
            },
            {
                "step_id": "step_02",
                "cluster_id": "grid_group_01",
                "review_goal": "conventional_adenoma_assessment",
                "need_to_see": "adenomatous gland crowding",
                "stage_gate": "conventional_adenoma",
                "overview_image": _high_mag_svg("conventional overview", "[1,1]"),
                "local_image": _high_mag_svg("conventional local", "[1,1]"),
                "detail_image": _high_mag_svg("conventional detail", "[1,1]"),
            },
            {
                "step_id": "step_03",
                "cluster_id": "grid_group_02",
                "review_goal": "non_serrated_overview_assessment",
                "need_to_see": "reactive mucosal context",
                "stage_gate": "non_serrated_context",
                "overview_image": _high_mag_svg("inflammatory overview", "[2,3]"),
                "local_image": _high_mag_svg("inflammatory local", "[2,3]"),
                "detail_image": _high_mag_svg("inflammatory detail", "[2,3]"),
            },
        ],
        "playback_state": {
            "current_step_index": 0,
            "is_playing": False,
            "speed": 1.0,
        },
        "observations": [
            {
                "step_id": "step_00",
                "observation": "当前 grid 对应的 5x 基础视野内可见锯齿状表面结构，局部黏膜呈 pale flat 外观。",
                "reasoning": "这一视野更偏向 serrated pathway，需要在同一目标上进一步放大查看基底隐窝形态。",
                "level_1_findings": ["serrated_surface_pattern"],
                "level_2_findings": [],
                "level_3_findings": [],
                "stage_decision": "supports_serrated_lesion",
                "confidence": 0.71,
                "metadata": {
                    "cluster_id": "grid_group_00",
                    "review_goal": "serrated_lesion_assessment",
                    "stage_gate": "mucosa_or_serrated",
                    "need_to_see": "basal crypt serration",
                },
            },
            {
                "step_id": "step_01",
                "observation": "20x 真正放大后可见基底部隐窝扩张与轻度分支，支持 abnormal crypt 方向。",
                "reasoning": "Junior 认为当前证据已足以支持 abnormal_crypt 分支继续推进。",
                "level_1_findings": [],
                "level_2_findings": ["basal_dilatation", "crypt_branching"],
                "level_3_findings": [],
                "stage_decision": "supports_abnormal_crypt",
                "confidence": 0.76,
                "metadata": {
                    "cluster_id": "grid_group_00",
                    "review_goal": "abnormal_crypt_assessment",
                    "stage_gate": "abnormal_crypt",
                    "need_to_see": "boot-shaped crypt base",
                },
            },
            {
                "step_id": "step_02",
                "observation": "传统腺瘤样区域未进一步强化，Chief 倾向当前已可形成筛查级总结。",
                "reasoning": "局部视野未提供足以推翻 serrated 主链的新增证据。",
                "level_1_findings": [],
                "level_2_findings": [],
                "level_3_findings": ["architectural_crowding"],
                "stage_decision": "serrated_dysplasia_not_supported_or_indeterminate",
                "confidence": 0.69,
                "metadata": {
                    "cluster_id": "grid_group_01",
                    "review_goal": "integrated_impression",
                    "stage_gate": "end",
                    "need_to_see": "integrate branch evidence",
                },
            },
        ],
        "global_reviews": [
            {
                "review_id": "global_review_0001",
                "source_step_id": "step_00",
                "decision": "continue",
                "continue_reason": "当前只确认到 serrated surface pattern，仍需 crypt-base morphology。",
                "chief_confidence": 0.74,
                "resolved_branch_state": {
                    "serrated": "supported",
                    "abnormal_crypt": "unresolved",
                    "conventional": "unresolved",
                    "dysplasia": "unresolved"
                },
                "sufficient_evidence": [],
                "unresolved_questions": [
                    "Need basal crypt evidence to distinguish SSL from reactive serration."
                ],
                "next_visual_target": {
                    "target_cluster_id": "grid_group_00",
                    "target_branch": "serrated",
                    "target_region_semantic": "ssl_suspicious_mucosa",
                    "target_morphology_prompt": [
                        "look for basal crypt dilatation",
                        "seek boot-shaped crypt base"
                    ],
                    "preferred_magnification": 20.0,
                    "priority_reason": "Current decision depends on crypt-base morphology."
                },
                "branch_correction_reason": "",
                "chief_thinking": "Chief 认为当前表面锯齿化已成立，但仍缺 crypt-base 证据，不能过早停。"
            },
            {
                "review_id": "global_review_0002",
                "source_step_id": "step_01",
                "decision": "continue",
                "continue_reason": "abnormal crypt 证据正在增强，但仍需检查传统腺瘤分支是否足以改变总结。",
                "chief_confidence": 0.79,
                "resolved_branch_state": {
                    "serrated": "supported",
                    "abnormal_crypt": "supported",
                    "conventional": "unresolved",
                    "dysplasia": "unresolved"
                },
                "sufficient_evidence": [],
                "unresolved_questions": [
                    "Need to rule out a stronger conventional explanation."
                ],
                "next_visual_target": {
                    "target_cluster_id": "grid_group_01",
                    "target_branch": "conventional",
                    "target_region_semantic": "conventional_adenoma_like",
                    "target_morphology_prompt": [
                        "compare gland crowding with serrated branch evidence"
                    ],
                    "preferred_magnification": 5.0,
                    "priority_reason": "Chief wants a final branch-level reconciliation before early stop."
                },
                "branch_correction_reason": "",
                "chief_thinking": "Chief 暂时不排除 conventional，要求回到另一条主线做最后比对。"
            },
            {
                "review_id": "global_review_0003",
                "source_step_id": "step_02",
                "decision": "early_stop",
                "continue_reason": "",
                "chief_confidence": 0.83,
                "resolved_branch_state": {
                    "serrated": "supported",
                    "abnormal_crypt": "supported",
                    "conventional": "opposed",
                    "dysplasia": "unresolved"
                },
                "sufficient_evidence": [
                    "serrated surface pattern already supported",
                    "basal crypt change supports abnormal crypt branch",
                    "additional conventional review did not overturn serrated-dominant explanation"
                ],
                "unresolved_questions": [],
                "next_visual_target": None,
                "branch_correction_reason": "",
                "chief_thinking": "Chief 认为继续扫描的边际收益已低，可以 early stop 并输出整合报告。"
            }
        ],
        "observe_report": {
            "hierarchical_prediction": {
                "serrated_lesion_assessment": "supported",
                "abnormal_crypt_assessment": "supported",
                "conventional_adenoma_assessment": "opposed",
                "final_case_assessment": "SSL-like pathway favored at screening level"
            },
            "serrated_checklist": [
                {"label": "serrated_surface_pattern", "confidence": 0.74}
            ],
            "abnormal_crypt_checklist": [
                {"label": "basal_dilatation", "confidence": 0.76},
                {"label": "crypt_branching", "confidence": 0.72}
            ],
            "conventional_adenoma_checklist": [],
            "serrated_dysplasia_checklist": [],
            "conventional_dysplasia_checklist": [],
            "dysplasia_checklist": [],
            "integrated_report": {
                "summary": "Chief favors a serrated-dominant explanation with abnormal crypt support and no stronger conventional override.",
                "recommendations": [
                    "Document SSL-favoring screening impression.",
                    "Keep branch evidence auditable for retrospective review."
                ]
            }
        },
        "observation_contract_validation": {
            "status": "ok",
            "violations": []
        },
        "navigation_contract_validation": {
            "status": "ok",
            "violations": [],
        },
    }


def _default_score_from_priority(priority):
    priority = _safe_int(priority, 0)
    if priority <= 0:
        return 0.0
    return round(min(1.0, max(0.0, float(priority) / 4.0)), 3)


def _parse_magnification_from_path(path):
    path = str(path or "")
    marker = "_mag"
    if marker not in path:
        return None
    tail = path.split(marker, 1)[1]
    token = []
    for char in tail:
        if char.isdigit() or char == ".":
            token.append(char)
        else:
            break
    if not token:
        return None
    try:
        return float("".join(token))
    except Exception:
        return None


def _best_bundle_image(observation):
    best = None
    metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
    bundle_paths = metadata.get("bundle_image_paths", [])
    if not isinstance(bundle_paths, list):
        bundle_paths = []
    for path in bundle_paths:
        if not path:
            continue
        magnification = _parse_magnification_from_path(path)
        role_rank = 0
        text = str(path).lower()
        if "detail" in text:
            role_rank = 3
        elif "local" in text:
            role_rank = 2
        elif "overview" in text:
            role_rank = 1
        candidate = {
            "path": str(path),
            "magnification": magnification if magnification is not None else 0.0,
            "role_rank": role_rank,
        }
        if best is None or (candidate["magnification"], candidate["role_rank"]) > (best["magnification"], best["role_rank"]):
            best = candidate
    crop_path = observation.get("crop_path")
    if crop_path and best is None:
        best = {
            "path": str(crop_path),
            "magnification": _parse_magnification_from_path(crop_path) or _safe_float(metadata.get("magnification"), 0.0),
            "role_rank": 0,
        }
    return best


def _pick_best_cluster_observation(cluster_records):
    best = None
    for index, observation in enumerate(cluster_records):
        image = _best_bundle_image(observation)
        metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
        candidate = {
            "cluster_id": str(metadata.get("cluster_id", "")),
            "observation": observation,
            "image": image,
            "confidence": _safe_float(observation.get("confidence"), 0.5),
            "backend": str(metadata.get("backend", "")),
            "review_goal": str(metadata.get("review_goal", "")),
            "stage_decision": str(observation.get("stage_decision", "")),
        }
        rank = (
            candidate["image"]["magnification"] if candidate["image"] else 0.0,
            candidate["image"]["role_rank"] if candidate["image"] else 0,
            candidate["confidence"],
            index,
        )
        if best is None:
            best = (rank, candidate)
            continue
        if rank > best[0]:
            best = (rank, candidate)
    return best[1] if best else None


def _cluster_observation_index(observations):
    mapping = {}
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
        cluster_id = str(metadata.get("cluster_id", "")).strip()
        if not cluster_id:
            continue
        mapping.setdefault(cluster_id, []).append(observation)
    return {cluster_id: _pick_best_cluster_observation(records) for cluster_id, records in mapping.items()}


def _observations_from_events(case_dir):
    events_path = Path(case_dir) / "events.jsonl"
    if not events_path.exists():
        return []
    rows = []
    try:
        events = read_jsonl(events_path)
    except Exception:
        return []
    seen = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("state", "")).strip() != "OBSERVE":
            continue
        if str(event.get("status", "")).strip() != "ok":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        step_id = str(payload.get("step_id", "")).strip()
        if not step_id or step_id in seen:
            continue
        seen.add(step_id)
        rows.append(payload)
    return rows


def _case_file_index(case_dir):
    lookup = {}
    for path in Path(case_dir).rglob("*"):
        if not path.is_file():
            continue
        lookup.setdefault(path.name, str(path.resolve()))
    return lookup


def _resolve_existing_case_asset_path(case_dir, raw_path, file_index):
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return ""
    candidate = Path(raw_path)
    if candidate.exists():
        return str(candidate.resolve())
    relative_candidate = Path(case_dir) / raw_path
    if relative_candidate.exists():
        return str(relative_candidate.resolve())
    return file_index.get(candidate.name, raw_path)


def _grid_cell_bbox_level0(cell):
    x1 = _safe_int(cell.get("level0_top_left_x"), 0)
    y1 = _safe_int(cell.get("level0_top_left_y"), 0)
    width = _safe_int(cell.get("level0_width"), 0)
    height = _safe_int(cell.get("level0_height"), 0)
    return [x1, y1, x1 + width, y1 + height]


def build_dashboard_payload_from_harness_case(case_dir):
    case_dir = Path(case_dir).resolve()
    trace_dir = case_dir / "trace"
    grid_input_dir = trace_dir / "grid_input"
    observe_dir = case_dir / "observe"
    navigation_dir = case_dir / "navigation"

    case_result_path = case_dir / "case_result.json"
    run_metadata_path = case_dir / "run_metadata.json"
    trace_clusters_path = trace_dir / "trace_clusters.json"
    observation_records_path = observe_dir / "observation_records.json"
    pathology_report_path = observe_dir / "pathological_report.json"
    reasoning_state_path = observe_dir / "reasoning_state.json"
    navigation_steps_path = navigation_dir / "navigation_steps.json"
    crop_manifest_path = observe_dir / "crop_manifest.json"
    boxes_json_candidates = sorted(trace_dir.glob("*_grid_input_boxes.json"))
    grid_json_candidates = sorted(grid_input_dir.glob("*.json"))
    grid_image_candidates = sorted(list(grid_input_dir.glob("*.jpg")) + list(grid_input_dir.glob("*.png")))

    if not grid_json_candidates:
        raise FileNotFoundError("No grid metadata JSON found under {0}".format(grid_input_dir))
    if not grid_image_candidates:
        raise FileNotFoundError("No grid thumbnail image found under {0}".format(grid_input_dir))
    if not trace_clusters_path.exists():
        raise FileNotFoundError("trace_clusters.json not found under {0}".format(trace_dir))

    grid_meta = read_json(grid_json_candidates[0])
    case_result = read_json(case_result_path) if case_result_path.exists() else {}
    run_metadata = read_json(run_metadata_path) if run_metadata_path.exists() else {}
    trace_clusters_payload = read_json(trace_clusters_path)
    observation_records_payload = read_json(observation_records_path) if observation_records_path.exists() else {}
    pathology_report = read_json(pathology_report_path) if pathology_report_path.exists() else {}
    reasoning_state = read_json(reasoning_state_path) if reasoning_state_path.exists() else {}
    navigation_steps = read_json(navigation_steps_path) if navigation_steps_path.exists() else {}
    crop_manifest = read_json(crop_manifest_path) if crop_manifest_path.exists() else {}
    boxes_payload = read_json(boxes_json_candidates[0]) if boxes_json_candidates else {}
    file_index = _case_file_index(case_dir)

    clusters = trace_clusters_payload.get("clusters", []) if isinstance(trace_clusters_payload, dict) else []
    grid_lookup = _grid_cell_lookup(grid_meta)
    selected_patch_ids = _selected_patch_ids_from_payload({"grid_metadata": grid_meta})
    selected_patch_lookup = {_normalize_patch_id(item) for item in selected_patch_ids}
    observations = observation_records_payload.get("observations", []) if isinstance(observation_records_payload, dict) else []
    if not observations:
        observations = _observations_from_events(case_dir)
    cluster_observations = _cluster_observation_index(observations)
    observe_assets_by_step = _merge_observe_assets_by_step(
        _build_observe_assets_by_step(observations, case_dir, file_index),
        _build_observe_assets_from_crop_manifest(crop_manifest, case_dir, file_index),
    )
    boxes_lookup = {}
    for item in boxes_payload.get("boxes", []) if isinstance(boxes_payload, dict) else []:
        if not isinstance(item, dict):
            continue
        row_col = _normalize_patch_id(item.get("patch_id"))
        if row_col is not None:
            boxes_lookup[row_col] = item

    backend_name = ""
    if observations:
        first_meta = observations[0].get("metadata", {}) if isinstance(observations[0].get("metadata"), dict) else {}
        backend_name = str(first_meta.get("backend", ""))
    if not backend_name:
        attempts = trace_clusters_payload.get("backend_attempts", []) if isinstance(trace_clusters_payload, dict) else []
        if attempts and isinstance(attempts[0], dict):
            backend_name = str(attempts[0].get("backend", ""))

    cluster_by_patch = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        patch_ids = cluster.get("patch_ids_ordered") or ((cluster.get("metadata", {}) or {}).get("grid_id_list")) or []
        if not patch_ids:
            patch_ids = [patch.get("patch_id") for patch in cluster.get("patches_thumb", []) if isinstance(patch, dict)]
        for patch_id in patch_ids:
            row_col = _normalize_patch_id(patch_id)
            if row_col is not None and row_col not in cluster_by_patch:
                cluster_by_patch[row_col] = cluster

    raw_assignment_payload = trace_clusters_payload.get("patch_assignments", {})
    raw_assignment_lookup = {}
    if isinstance(raw_assignment_payload, dict):
        for item in raw_assignment_payload.get("patches", []) if isinstance(raw_assignment_payload.get("patches", []), list) else []:
            if not isinstance(item, dict):
                continue
            row_col = _normalize_patch_id(item.get("patch_id"))
            if row_col is not None:
                raw_assignment_lookup[row_col] = item
    elif isinstance(raw_assignment_payload, list):
        for item in raw_assignment_payload:
            if not isinstance(item, dict):
                continue
            row_col = _normalize_patch_id(item.get("patch_id"))
            if row_col is not None:
                raw_assignment_lookup[row_col] = item

    patch_assignments = []
    high_mag_assets = []
    asset_refs = {}
    dual_model_mode_seen = False
    for row_col in selected_patch_ids:
        cell = grid_lookup.get(row_col)
        if not cell:
            continue
        cluster = cluster_by_patch.get(row_col, {})
        cluster_id = str(cluster.get("cluster_id", "")).strip()
        observation_info = cluster_observations.get(cluster_id)
        observation_confidence = observation_info["confidence"] if observation_info else None
        stage_decision = observation_info["stage_decision"] if observation_info else ""
        review_goal = observation_info["review_goal"] if observation_info else ""
        image = observation_info["image"] if observation_info else None
        high_mag_ref = ""
        if image and image.get("path"):
            resolved_image_path = _resolve_existing_case_asset_path(case_dir, image.get("path"), file_index)
            if resolved_image_path:
                high_mag_ref = "{0}_high_mag".format(cluster_id or "patch_{0}_{1}".format(int(row_col[0]), int(row_col[1])))
            if resolved_image_path and high_mag_ref not in asset_refs:
                asset_refs[high_mag_ref] = True
                representative_patch = [int(row_col[0]), int(row_col[1])]
                high_mag_assets.append(
                    {
                        "asset_id": high_mag_ref,
                        "patch_id": representative_patch,
                        "magnification": image.get("magnification", 0.0),
                        "image_path": resolved_image_path,
                        "width": 256,
                        "height": 256,
                        "data_label": str((raw_assignment_lookup.get(row_col) or {}).get("region_semantic") or cluster.get("l", "unknown")),
                    }
                )
        raw_patch = dict(raw_assignment_lookup.get(row_col, {}))
        if not raw_patch:
            raw_patch = {
                "patch_id": [int(row_col[0]), int(row_col[1])],
                "region_semantic": str(cluster.get("l", "unknown")),
                "diagnostic_priority": _safe_int(cluster.get("s"), 0),
                "require_high_magnification": bool(cluster.get("d", False) or (cluster.get("metadata", {}) or {}).get("requires_high_magnification", False)),
                "agreement_status": GLOBAL_SCREENING_SINGLE_MODEL_STATUS,
                "conch_region_semantic": "not_available_in_this_run",
                "pathoreasoner_r1_region_semantic": str(cluster.get("l", "unknown")),
                "fusion_reasoning": str(cluster.get("desc", "")).strip() or "Single-model trace output expanded from cluster-level assignment.",
                "score_origin": GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN,
            }
        elif cluster and not cluster_id:
            cluster_id = str((cluster.get("cluster_id", "") or "")).strip()
        patch = _build_dashboard_patch(raw_patch, cell, cluster, boxes_lookup.get(row_col, {}))
        if _looks_like_dual_model_global_screening_patch(patch):
            dual_model_mode_seen = True
        if high_mag_ref:
            patch["high_mag_ref"] = high_mag_ref
        if observation_confidence is not None:
            patch["observation_confidence"] = observation_confidence
            if "uncertainty_score" not in patch or patch.get("uncertainty_score") is None:
                patch["uncertainty_score"] = round(max(0.0, 1.0 - observation_confidence), 3)
        if stage_decision:
            patch["stage_decision"] = stage_decision
        if review_goal:
            patch["review_goal"] = review_goal
        if cluster_id:
            patch["cluster_id"] = cluster_id
        patch.setdefault("tissue_coverage_score", boxes_lookup.get(row_col, {}).get("score"))
        if "score_origin" not in patch or not str(patch.get("score_origin", "")).strip():
            patch["score_origin"] = GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN if _looks_like_dual_model_global_screening_patch(patch) else GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN
        if "score" not in patch or patch.get("score") is None:
            max_priority = 5 if patch.get("score_origin") == GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN else 4
            patch["score"] = _score_from_priority(patch.get("diagnostic_priority", 0), max_priority=max_priority)
        patch_assignments.append(patch)

    violation_status = "ok"
    if selected_patch_lookup and len(patch_assignments) != len(selected_patch_lookup):
        violation_status = "warning"

    integrated_report = ""
    recommendations = []
    if isinstance(pathology_report.get("integrated_report"), dict):
        integrated_report = str((pathology_report.get("integrated_report") or {}).get("summary", ""))
        recommendations = list((pathology_report.get("integrated_report") or {}).get("recommendations", []))
    elif isinstance(case_result.get("integrated_report"), str):
        integrated_report = str(case_result.get("integrated_report", ""))

    trace_rubric = {
        "mode": "dynamic_priority_consensus" if dual_model_mode_seen else "single_model_harness_trace",
        "labels": _default_trace_rubric_summary(),
        "score_note": (
            "score is derived from consensus-fused diagnostic_priority because this harness run emits dual-model audit fields."
            if dual_model_mode_seen
            else "score is derived from diagnostic_priority because this harness run does not emit a native patch-level 0-1 diagnostic score."
        ),
    }

    grid_metadata = dict(grid_meta)
    grid_metadata["run_id"] = case_dir.parent.name
    grid_metadata["experiment_tag"] = case_dir.parent.parent.name if case_dir.parent.parent != case_dir.parent else case_dir.parent.name
    grid_metadata["input_mode"] = ((run_metadata.get("case", {}) or {}).get("input_mode")) if isinstance(run_metadata.get("case", {}), dict) else None
    grid_metadata["backend"] = backend_name

    trace_clusters = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        metadata = cluster.get("metadata", {}) if isinstance(cluster.get("metadata"), dict) else {}
        trace_clusters.append(
            {
                "cluster_id": str(cluster.get("cluster_id", "")).strip(),
                "cluster_label": str(cluster.get("l", "")).strip(),
                "cluster_priority": _safe_int(cluster.get("s"), 0),
                "workflow_branch": str(metadata.get("workflow_branch", "")).strip(),
                "require_high_magnification": bool(cluster.get("d", False) or metadata.get("requires_high_magnification", False)),
                "bbox_level0": list(cluster.get("cluster_bbox_level0", {}).values()) if isinstance(cluster.get("cluster_bbox_level0"), dict) else None,
            }
        )

    normalized_navigation_steps = []
    for index, step in enumerate(navigation_steps.get("steps", []) if isinstance(navigation_steps, dict) else []):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
        normalized_navigation_steps.append(
            {
                "step_id": str(step.get("step_id", "")).strip() or "step_{0}".format(index),
                "x": step.get("x"),
                "y": step.get("y"),
                "m": _normalize_navigation_m(step.get("m"), step.get("region_size_level0")),
                "raw_m": step.get("m"),
                "action": _normalize_navigation_action(step),
                "region_size_level0": _safe_int(step.get("region_size_level0"), 0),
                "need_to_see": str(step.get("need_to_see", "")).strip(),
                "review_goal": str(step.get("review_goal", "")).strip(),
                "stage_gate": str(step.get("stage_gate", "")).strip(),
                "metadata": metadata,
            }
        )

    navigation_contract_validation = _build_navigation_contract_validation(normalized_navigation_steps, observe_assets_by_step, grid_meta)
    trace_cluster_lookup = {item["cluster_id"]: item for item in trace_clusters if item.get("cluster_id")}
    raw_global_reviews = observation_records_payload.get("global_reviews", []) if isinstance(observation_records_payload, dict) else []
    chief_reviews = _read_chief_reviews(case_dir)
    global_reviews = raw_global_reviews if raw_global_reviews else (chief_reviews if chief_reviews else _build_fallback_global_reviews(observations, trace_cluster_lookup, reasoning_state, pathology_report))
    if raw_global_reviews and chief_reviews:
        global_reviews = _merge_chief_debug_into_reviews(global_reviews, chief_reviews)
    observation_contract_validation = _build_observation_contract_validation(observations, global_reviews, pathology_report)
    case_id_value = str(case_result.get("case_id") or (run_metadata.get("case", {}) or {}).get("case_id") or case_dir.name)
    ground_truth_label = _ground_truth_label_for_case(case_id_value)
    final_report_comparison = _compare_final_report_with_ground_truth(ground_truth_label, pathology_report, case_result)

    payload = {
        "case_id": case_id_value,
        "slide_id": str((run_metadata.get("case", {}) or {}).get("slide_id") or case_dir.name),
        "source_case_dir": str(case_dir),
        "source_run_dir": str(case_dir.parent),
        "selected_patch_ids": [[int(item[0]), int(item[1])] for item in selected_patch_ids],
        "grid_metadata": grid_metadata,
        "trace_rubric": trace_rubric,
        "overview_image": {
            "image_path": str(grid_image_candidates[0]),
            "width": _safe_int(grid_meta.get("cropped_thumbnail_size", [0, 0])[0], 0),
            "height": _safe_int(grid_meta.get("cropped_thumbnail_size", [0, 0])[1], 0),
        },
        "patch_assignments": patch_assignments,
        "high_mag_assets": high_mag_assets,
        "audit_log": [],
        "contract_validation": {
            "status": violation_status,
            "violations": [],
        },
        "orphan_assets": [],
        "trace_clusters": trace_clusters,
        "navigation_steps": normalized_navigation_steps,
        "observe_assets_by_step": observe_assets_by_step,
        "observations": observations,
        "global_reviews": global_reviews,
        "chief_reviews": chief_reviews,
        "crop_manifest": crop_manifest,
        "case_result": case_result,
        "observe_report": pathology_report,
        "reasoning_state": reasoning_state,
        "ground_truth_label": ground_truth_label,
        "final_report_comparison": final_report_comparison,
        "playback_state": {
            "current_step_index": 0,
            "is_playing": False,
            "speed": 1.0,
        },
        "navigation_contract_validation": navigation_contract_validation,
        "observation_contract_validation": observation_contract_validation,
        "dashboard_notes": {
            "integrated_report": integrated_report,
            "recommendations": recommendations,
            "navigation_step_count": len((navigation_steps.get("steps") or [])) if isinstance(navigation_steps, dict) else 0,
            "observation_step_count": len(observations),
        },
    }
    return payload


def _patch_key(item):
    row_col = _normalize_patch_id(item)
    if row_col is None:
        return None
    return "{0}_{1}".format(int(row_col[0]), int(row_col[1]))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        if default is None:
            return None
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _grade_score_bucket(score):
    score = _safe_float(score)
    if score < 0.3:
        return "lt_03"
    if score <= 0.7:
        return "mid_03_07"
    return "gt_07"


def _alpha_from_score(score):
    score = _safe_float(score)
    return round(max(0.16, min(0.92, 0.18 + score * 0.72)), 3)


def _rgba_with_alpha(base_rgba, alpha):
    text = str(base_rgba or "")
    if not text.startswith("rgba("):
        return text
    inner = text[5:-1]
    parts = [item.strip() for item in inner.split(",")]
    if len(parts) != 4:
        return text
    return "rgba({0}, {1}, {2}, {3})".format(parts[0], parts[1], parts[2], alpha)


def _gradient_for_label(label):
    palette = {
        "ssl_suspicious_mucosa": ("#f97316", "#dc2626"),
        "conventional_adenoma_like": ("#f59e0b", "#b45309"),
        "inflammatory_polyp_like": ("#14b8a6", "#0f766e"),
        "normal_mucosa": ("#22c55e", "#15803d"),
        "background_artifact_stroma": ("#64748b", "#334155"),
        "unknown": ("#94a3b8", "#64748b"),
    }
    start, end = palette.get(label, palette["unknown"])
    return "linear-gradient(135deg, {0}, {1})".format(start, end)


def _overview_background_svg(width, height):
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="wash" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f5e6cf"/>
      <stop offset="50%" stop-color="#e8c8d2"/>
      <stop offset="100%" stop-color="#d4d9b5"/>
    </linearGradient>
    <radialGradient id="focus" cx="55%" cy="42%" r="60%">
      <stop offset="0%" stop-color="rgba(255,255,255,0.92)"/>
      <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
    </radialGradient>
    <filter id="blur" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur stdDeviation="28" />
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="url(#wash)" />
  <path d="M24 280 C132 180, 260 200, 372 302 S 640 430, 818 296 S 970 160, 976 186 L 976 1000 L 0 1000 L 0 390 C 92 358, 170 332, 24 280 Z" fill="#f8ead7" opacity="0.72"/>
  <path d="M30 320 C162 184, 296 228, 440 338 S 702 452, 940 308" fill="none" stroke="#be8c7b" stroke-width="54" stroke-linecap="round" opacity="0.3" filter="url(#blur)"/>
  <path d="M44 330 C168 208, 308 250, 450 350 S 712 444, 924 320" fill="none" stroke="#c6816d" stroke-width="20" stroke-linecap="round" opacity="0.4"/>
  <path d="M118 386 C180 352, 252 350, 316 388" fill="none" stroke="#d59d77" stroke-width="10" stroke-linecap="round" opacity="0.45"/>
  <path d="M420 410 C502 372, 576 382, 652 424" fill="none" stroke="#d59d77" stroke-width="10" stroke-linecap="round" opacity="0.45"/>
  <path d="M716 370 C776 334, 858 334, 916 380" fill="none" stroke="#d59d77" stroke-width="10" stroke-linecap="round" opacity="0.45"/>
  <circle cx="550" cy="300" r="240" fill="url(#focus)" opacity="0.6"/>
</svg>
""".strip().format(width=int(width), height=int(height))
    return "data:image/svg+xml;utf8," + svg.replace("#", "%23").replace("\n", "")


def _high_mag_svg(label, patch_text):
    label = str(label or "unknown")
    patch_text = str(patch_text or "patch")
    svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
  <defs>
    <linearGradient id="base" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#fff3e4"/>
      <stop offset="55%" stop-color="#f0cfdb"/>
      <stop offset="100%" stop-color="#e8d9c7"/>
    </linearGradient>
    <filter id="soft" x="-10%" y="-10%" width="120%" height="120%">
      <feGaussianBlur stdDeviation="6" />
    </filter>
  </defs>
  <rect width="1024" height="1024" fill="url(#base)" />
  <g opacity="0.65">
    <path d="M64 664 C178 460, 320 388, 470 460 S 704 664, 904 592" fill="none" stroke="#b55f54" stroke-width="86" stroke-linecap="round" filter="url(#soft)"/>
    <path d="M90 682 C198 484, 336 434, 486 496 S 708 666, 890 614" fill="none" stroke="#cf8574" stroke-width="30" stroke-linecap="round"/>
  </g>
  <g fill="none" stroke="#a74f49" stroke-width="14" opacity="0.5">
    <path d="M170 684 C214 584, 258 544, 314 568 C358 590, 356 660, 388 728" />
    <path d="M328 714 C372 604, 418 564, 468 584 C520 606, 520 684, 548 744" />
    <path d="M498 738 C540 626, 586 590, 640 610 C686 628, 694 696, 720 760" />
  </g>
  <g font-family="Georgia, serif" fill="#4b2f35">
    <text x="68" y="110" font-size="42" font-weight="700">{patch_text}</text>
    <text x="68" y="162" font-size="28">{label}</text>
  </g>
</svg>
""".strip().format(label=escape(label), patch_text=escape(patch_text))
    return "data:image/svg+xml;utf8," + svg.replace("#", "%23").replace("\n", "")


def _materialize_asset_url(raw_value, output_dir, subdir, source_root=None):
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if value.startswith("data:") or value.startswith("http://") or value.startswith("https://"):
        return value
    source_root = Path(source_root).resolve() if source_root else None
    candidate = Path(value)
    if not candidate.is_absolute() and source_root is not None:
        candidate = source_root / candidate
    if not candidate.exists():
        return value
    asset_dir = ensure_dir(Path(output_dir) / "assets" / subdir)
    target = asset_dir / candidate.name
    if candidate.resolve() != target.resolve():
        shutil.copy2(str(candidate), str(target))
    return str(Path("assets") / subdir / candidate.name)


def _script_json_text(payload):
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _selected_patch_ids_from_payload(payload):
    selected = []
    for item in payload.get("selected_patch_ids", []):
        row_col = _normalize_patch_id(item)
        if row_col is not None and row_col not in selected:
            selected.append(row_col)
    if selected:
        return selected
    grid_meta = payload.get("grid_metadata", {})
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict) or not cell.get("is_selected"):
            continue
        row_col = _normalize_patch_id(cell.get("patch_id") or [cell.get("row_id"), cell.get("col_id")])
        if row_col is not None and row_col not in selected:
            selected.append(row_col)
    return selected


def _grid_cell_lookup(grid_meta):
    lookup = {}
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict):
            continue
        row_col = _normalize_patch_id(cell.get("patch_id") or [cell.get("row_id"), cell.get("col_id")])
        if row_col is None:
            continue
        lookup[row_col] = cell
    return lookup


def _grid_canvas_size(grid_meta):
    size = grid_meta.get("cropped_thumbnail_size", [])
    if isinstance(size, list) and len(size) >= 2:
        return max(1, _safe_int(size[0], 1)), max(1, _safe_int(size[1], 1))
    cell_lookup = _grid_cell_lookup(grid_meta)
    max_x = 0
    max_y = 0
    for cell in cell_lookup.values():
        max_x = max(max_x, _safe_int(cell.get("thumbnail_top_left_x")) + _safe_int(cell.get("thumbnail_width")))
        max_y = max(max_y, _safe_int(cell.get("thumbnail_top_left_y")) + _safe_int(cell.get("thumbnail_height")))
    return max(1, max_x), max(1, max_y)


def _overview_level0_bbox(grid_meta):
    bbox = grid_meta.get("level0_crop_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return {
            "x1": _safe_int(bbox[0]),
            "y1": _safe_int(bbox[1]),
            "x2": _safe_int(bbox[2]),
            "y2": _safe_int(bbox[3]),
        }
    slide_w = 0
    slide_h = 0
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict):
            continue
        slide_w = max(slide_w, _safe_int(cell.get("level0_top_left_x")) + _safe_int(cell.get("level0_width")))
        slide_h = max(slide_h, _safe_int(cell.get("level0_top_left_y")) + _safe_int(cell.get("level0_height")))
    return {"x1": 0, "y1": 0, "x2": max(1, slide_w), "y2": max(1, slide_h)}


def _normalize_navigation_m(raw_m, region_size_level0):
    text = str(raw_m).strip().lower()
    if "20" in text:
        return "20x"
    if "10" in text:
        return "10x"
    if "5" in text:
        return "5x"
    value = _safe_float(raw_m, None)
    if value is not None:
        if value >= 20:
            return "20x"
        if value >= 10:
            return "10x"
        return "5x"
    region_size_level0 = _safe_int(region_size_level0, 0)
    if region_size_level0 and region_size_level0 <= 1024:
        return "10x"
    return "5x"


def _normalize_navigation_action(step):
    metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
    action = str(step.get("action") or metadata.get("action") or "").strip().lower()
    if action in {"inspect", "stop"}:
        return action
    stage_gate = str(step.get("stage_gate", "")).strip().lower()
    review_goal = str(step.get("review_goal", "")).strip().lower()
    if stage_gate == "end" or review_goal == "integrated_impression":
        return "stop"
    return "inspect"


def _observe_role_from_path(path):
    text = str(path or "").lower()
    if "detail" in text:
        return "detail_image"
    if "local" in text:
        return "local_image"
    if "overview" in text:
        return "overview_image"
    magnification = _parse_magnification_from_path(path)
    if magnification is None:
        return None
    if magnification >= 5.0:
        return "detail_image"
    if magnification >= 2.0:
        return "local_image"
    return "overview_image"


def _build_observe_assets_by_step(observations, case_dir, file_index):
    items = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
        step_id = str(observation.get("step_id", "")).strip()
        if not step_id:
            continue
        item = {
            "step_id": step_id,
            "cluster_id": str(metadata.get("cluster_id", "")).strip(),
            "review_goal": str(metadata.get("review_goal", "")).strip(),
            "need_to_see": str(metadata.get("need_to_see", "")).strip(),
            "stage_gate": str(metadata.get("stage_gate", "")).strip(),
            "overview_image": "",
            "local_image": "",
            "detail_image": "",
        }
        bundle_paths = metadata.get("bundle_image_paths", [])
        if not isinstance(bundle_paths, list):
            bundle_paths = []
        for path in bundle_paths:
            role = _observe_role_from_path(path)
            if not role:
                continue
            resolved = _resolve_existing_case_asset_path(case_dir, path, file_index)
            if resolved and not item.get(role):
                item[role] = resolved
        crop_path = observation.get("crop_path")
        if crop_path:
            role = _observe_role_from_path(crop_path) or "detail_image"
            resolved = _resolve_existing_case_asset_path(case_dir, crop_path, file_index)
            if resolved and not item.get(role):
                item[role] = resolved
        items.append(item)
    return items


def _build_observe_assets_from_crop_manifest(crop_manifest, case_dir, file_index):
    rows = []
    crops = crop_manifest.get("crops", []) if isinstance(crop_manifest, dict) else []
    for crop in crops if isinstance(crops, list) else []:
        if not isinstance(crop, dict):
            continue
        metadata = crop.get("metadata", {}) if isinstance(crop.get("metadata"), dict) else {}
        source_step_id = str(metadata.get("source_step_id") or metadata.get("view_bundle_id") or "").strip()
        if not source_step_id:
            step_text = str(crop.get("step_id", "")).strip()
            source_step_id = step_text.rsplit("__", 1)[0] if "__" in step_text else step_text
        if not source_step_id:
            continue
        item = {
            "step_id": source_step_id,
            "cluster_id": str(metadata.get("cluster_id", "")).strip(),
            "review_goal": "",
            "need_to_see": str(crop.get("need_to_see", "")).strip(),
            "stage_gate": "",
            "overview_image": "",
            "local_image": "",
            "detail_image": "",
            "raw_m": crop.get("m"),
            "region_size_level0": crop.get("region_size_level0"),
            "is_dynamic": str(source_step_id).startswith("step_dyn_"),
            "generated_by": str(metadata.get("generated_by", "")).strip(),
            "gate_source_step": str(metadata.get("gate_source_step", "")).strip(),
            "gate_source": str(metadata.get("gate_source", "")).strip(),
        }
        role = str(metadata.get("image_role") or "").strip().lower()
        path = crop.get("image_path")
        resolved = _resolve_existing_case_asset_path(case_dir, path, file_index)
        if resolved:
            if role == "overview":
                item["overview_image"] = resolved
            elif role == "local":
                item["local_image"] = resolved
            elif role == "detail":
                item["detail_image"] = resolved
            else:
                item[_observe_role_from_path(resolved) or "detail_image"] = resolved
        rows.append(item)
    return rows


def _merge_observe_assets_by_step(primary_rows, secondary_rows):
    merged = {}
    for rows in [primary_rows, secondary_rows]:
        for item in rows if isinstance(rows, list) else []:
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("step_id", "")).strip()
            if not step_id:
                continue
            row = merged.setdefault(
                step_id,
                {
                    "step_id": step_id,
                    "cluster_id": "",
                    "review_goal": "",
                    "need_to_see": "",
                    "stage_gate": "",
                    "overview_image": "",
                    "local_image": "",
                    "detail_image": "",
                },
            )
            for key, value in item.items():
                if key in {"overview_image", "local_image", "detail_image"}:
                    if value and not row.get(key):
                        row[key] = value
                elif value not in (None, "", [], {}):
                    row[key] = value
    return [merged[key] for key in sorted(merged.keys())]


def _read_text_if_exists(path):
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _read_chief_reviews(case_dir):
    review_dir = Path(case_dir) / "observe" / "chief_reviews"
    if not review_dir.exists():
        return []
    rows = []
    for response_path in sorted(review_dir.glob("*_chief_response.json")):
        try:
            review = read_json(response_path)
        except Exception:
            review = {}
        if not isinstance(review, dict):
            review = {}
        stem = response_path.name[: -len("_chief_response.json")]
        row = dict(review)
        step_id = str(row.get("source_step_id") or stem).strip()
        row["source_step_id"] = step_id
        row.setdefault("review_id", stem)
        row.setdefault("decision", "")
        row["chief_thinking"] = str(row.get("thought_text") or _read_text_if_exists(review_dir / "{0}_chief_thought_text.txt".format(stem))).strip()
        row["answer_candidate_text"] = str(row.get("answer_candidate_text") or _read_text_if_exists(review_dir / "{0}_chief_answer_candidate.txt".format(stem))).strip()
        row["parse_error"] = str(row.get("parse_error") or _read_text_if_exists(review_dir / "{0}_chief_parse_error.txt".format(stem))).strip()
        row["raw_text"] = str(row.get("raw_generated_text") or _read_text_if_exists(review_dir / "{0}_chief_raw_text.txt".format(stem))).strip()
        row.setdefault("metadata", {})
        if isinstance(row["metadata"], dict):
            row["metadata"].setdefault("chief_debug_response_json", str(response_path))
            row["metadata"].setdefault("chief_debug_raw_text", str(review_dir / "{0}_chief_raw_text.txt".format(stem)))
        rows.append(row)
    return rows


def _merge_chief_debug_into_reviews(global_reviews, chief_reviews):
    debug_by_step = {str(item.get("source_step_id", "")).strip(): item for item in chief_reviews if isinstance(item, dict)}
    merged = []
    for review in global_reviews if isinstance(global_reviews, list) else []:
        if not isinstance(review, dict):
            continue
        row = dict(review)
        step_id = str(row.get("source_step_id", "")).strip()
        debug = debug_by_step.get(step_id, {})
        for key in ["chief_thinking", "answer_candidate_text", "parse_error", "raw_text", "raw_generated_text", "review_source", "model_name", "round_trip_ms"]:
            if debug.get(key) not in (None, "") and row.get(key) in (None, ""):
                row[key] = debug.get(key)
        if isinstance(row.get("metadata"), dict) and isinstance(debug.get("metadata"), dict):
            merged_metadata = dict(row["metadata"])
            merged_metadata.update({key: value for key, value in debug["metadata"].items() if value not in (None, "")})
            row["metadata"] = merged_metadata
        merged.append(row)
    return merged


def _resolve_navigation_observe_assets(observe_assets_by_step, output_dir, source_root=None):
    rows = []
    for item in observe_assets_by_step if isinstance(observe_assets_by_step, list) else []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        for key in ["overview_image", "local_image", "detail_image"]:
            row[key] = _materialize_asset_url(item.get(key), output_dir, "observe", source_root=source_root)
        rows.append(row)
    return rows


def _navigation_cluster_lookup(trace_clusters):
    lookup = {}
    for cluster in trace_clusters if isinstance(trace_clusters, list) else []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id", "")).strip()
        if cluster_id:
            lookup[cluster_id] = cluster
    return lookup


def _build_navigation_contract_validation(navigation_steps, observe_assets_by_step, grid_meta):
    violations = []
    step_rows = navigation_steps if isinstance(navigation_steps, list) else []
    observe_lookup = {str(item.get("step_id", "")).strip(): item for item in observe_assets_by_step if isinstance(item, dict)}
    bbox = _overview_level0_bbox(grid_meta)
    if not step_rows:
        return {
            "status": "warning",
            "violations": [
                {
                    "type": "empty_navigation_steps",
                    "message": "navigation step 序列为空。",
                }
            ],
        }
    seen_stop = False
    for index, step in enumerate(step_rows):
        step_id = str(step.get("step_id", "")).strip() or "step_{0}".format(index)
        action = _normalize_navigation_action(step)
        if action == "stop":
            seen_stop = True
        m = _normalize_navigation_m(step.get("m"), step.get("region_size_level0"))
        if m not in {"5x", "10x", "20x"}:
            violations.append({"type": "invalid_magnification", "step_id": step_id, "message": "非法倍率。"})
        x = _safe_float(step.get("x"), None)
        y = _safe_float(step.get("y"), None)
        if x is None or y is None or x < bbox["x1"] or x > bbox["x2"] or y < bbox["y1"] or y > bbox["y2"]:
            violations.append({"type": "out_of_bounds_step", "step_id": step_id, "message": "step 坐标越界。"})
        if not str(step.get("review_goal", "")).strip():
            violations.append({"type": "missing_review_goal", "step_id": step_id, "message": "缺失 review_goal。"})
        if not str(step.get("stage_gate", "")).strip():
            violations.append({"type": "missing_stage_gate", "step_id": step_id, "message": "缺失 stage_gate。"})
        if not observe_lookup.get(step_id) and action != "stop":
            violations.append({"type": "missing_observe_asset", "step_id": step_id, "message": "缺失关联 Observe crop。"})
    if not seen_stop:
        violations.append({"type": "missing_stop_step", "message": "step 序列中没有 stop 节点。"})
    return {
        "status": "warning" if violations else "ok",
        "violations": violations,
    }


def _build_navigation_view_model(payload, canvas_width, canvas_height):
    navigation_steps = payload.get("navigation_steps", [])
    trace_clusters = payload.get("trace_clusters", [])
    observe_assets = payload.get("observe_assets_by_step", [])
    grid_meta = payload.get("grid_metadata", {})
    bbox = _overview_level0_bbox(grid_meta)
    bbox_w = max(1, bbox["x2"] - bbox["x1"])
    bbox_h = max(1, bbox["y2"] - bbox["y1"])
    cluster_lookup = _navigation_cluster_lookup(trace_clusters)
    observe_by_step = {str(item.get("step_id", "")).strip(): item for item in observe_assets if isinstance(item, dict)}
    steps = []
    planned_cluster_ids = set()
    observed_cluster_ids = set()
    previous_center = None
    for index, step in enumerate(navigation_steps if isinstance(navigation_steps, list) else []):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
        cluster_id = str(metadata.get("cluster_id", "")).strip()
        cluster = cluster_lookup.get(cluster_id, {})
        action = _normalize_navigation_action(step)
        normalized_m = _normalize_navigation_m(step.get("m"), step.get("region_size_level0"))
        color = NAVIGATION_MAG_COLORS.get(normalized_m, NAVIGATION_MAG_COLORS["unknown"])
        x = _safe_float(step.get("x"), None)
        y = _safe_float(step.get("y"), None)
        if (x is None or y is None) and previous_center is not None:
            x, y = previous_center
        if x is None:
            x = float(bbox["x1"])
        if y is None:
            y = float(bbox["y1"])
        previous_center = (x, y)
        center_x_percent = round(100.0 * (x - bbox["x1"]) / float(bbox_w), 4)
        center_y_percent = round(100.0 * (y - bbox["y1"]) / float(bbox_h), 4)
        region_size_level0 = _safe_int(step.get("region_size_level0"), 256)
        fov_w_percent = round(100.0 * region_size_level0 / float(bbox_w), 4)
        fov_h_percent = round(100.0 * region_size_level0 / float(bbox_h), 4)
        fov_left_percent = round(center_x_percent - fov_w_percent / 2.0, 4)
        fov_top_percent = round(center_y_percent - fov_h_percent / 2.0, 4)
        observe_asset = observe_by_step.get(str(step.get("step_id", "")).strip()) or {}
        if cluster_id:
            planned_cluster_ids.add(cluster_id)
        if observe_asset.get("cluster_id"):
            observed_cluster_ids.add(str(observe_asset.get("cluster_id")))
        steps.append(
            {
                "step_id": str(step.get("step_id", "")).strip() or "step_{0}".format(index),
                "step_index": index,
                "x": x,
                "y": y,
                "m": normalized_m,
                "raw_m": step.get("m"),
                "action": action,
                "action_display": NAVIGATION_ACTION_DISPLAY.get(action, action),
                "region_size_level0": region_size_level0,
                "need_to_see": str(step.get("need_to_see", "")).strip(),
                "review_goal": str(step.get("review_goal", "")).strip(),
                "stage_gate": str(step.get("stage_gate", "")).strip(),
                "cluster_id": cluster_id,
                "cluster_label": str(metadata.get("cluster_label", "") or cluster.get("cluster_label", "") or cluster.get("l", "")),
                "cluster_priority": _safe_int(metadata.get("cluster_priority"), _safe_int(cluster.get("cluster_priority"), _safe_int(cluster.get("s"), 0))),
                "workflow_branch": str(metadata.get("workflow_branch", "") or cluster.get("workflow_branch", "")),
                "patch_id": metadata.get("patch_id"),
                "node_size": max(12, 10 + _safe_int(metadata.get("cluster_priority"), _safe_int(cluster.get("s"), 0)) * 3),
                "stroke": color["stroke"],
                "fill": color["fill"],
                "center_x_percent": center_x_percent,
                "center_y_percent": center_y_percent,
                "fov_left_percent": fov_left_percent,
                "fov_top_percent": fov_top_percent,
                "fov_width_percent": fov_w_percent,
                "fov_height_percent": fov_h_percent,
                "overview_image": str(observe_asset.get("overview_image", "")),
                "local_image": str(observe_asset.get("local_image", "")),
                "detail_image": str(observe_asset.get("detail_image", "")),
                "observe_asset_step_id": str(observe_asset.get("step_id", "")),
                "is_observed": bool(observe_asset.get("overview_image") or observe_asset.get("local_image") or observe_asset.get("detail_image")),
                "is_stop": action == "stop",
            }
        )
    lines = []
    for previous, current in zip(steps, steps[1:]):
        lines.append(
            {
                "from_step_id": previous["step_id"],
                "to_step_id": current["step_id"],
                "x1": previous["center_x_percent"],
                "y1": previous["center_y_percent"],
                "x2": current["center_x_percent"],
                "y2": current["center_y_percent"],
            }
        )
    clusters = []
    for cluster in trace_clusters if isinstance(trace_clusters, list) else []:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id", "")).strip()
        cluster_priority = _safe_int(cluster.get("cluster_priority"), _safe_int(cluster.get("s"), 0))
        planned = cluster_id in planned_cluster_ids if cluster_id else False
        observed = cluster_id in observed_cluster_ids if cluster_id else False
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": str(cluster.get("cluster_label", "") or cluster.get("l", "")),
                "cluster_priority": cluster_priority,
                "workflow_branch": str(cluster.get("workflow_branch", "") or (cluster.get("metadata", {}) or {}).get("workflow_branch", "")),
                "require_high_magnification": bool(cluster.get("require_high_magnification", False) or cluster.get("d", False)),
                "planned_in_path": planned,
                "observed": observed,
                "missed_high_priority": (not planned) and cluster_priority >= 3,
            }
        )
    return {
        "steps": steps,
        "lines": lines,
        "clusters": clusters,
        "contract_validation": payload.get("navigation_contract_validation", {"status": "ok", "violations": []}),
        "initial_step_id": steps[0]["step_id"] if steps else None,
        "total_steps": len(steps),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
    }


def _review_goal_branch(review_goal):
    review_goal = str(review_goal or "").strip().lower()
    if review_goal in {"serrated_lesion_assessment", "abnormal_crypt_assessment", "serrated_dysplasia_assessment"}:
        return "serrated"
    if review_goal in {"conventional_adenoma_assessment", "conventional_dysplasia_assessment"}:
        return "conventional"
    return "non_serrated"


def _preferred_contract_magnification(observation):
    metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
    bundle = metadata.get("bundle_image_paths", [])
    if isinstance(bundle, list):
        for path in bundle:
            mag = _parse_magnification_from_path(path)
            if mag and mag >= 5.0:
                return 20.0
    mag = _safe_float(metadata.get("magnification"), 1.0)
    return 20.0 if mag >= 5.0 else 5.0


def _update_branch_state_from_stage_decision(state, stage_decision):
    stage_decision = str(stage_decision or "").strip()
    next_state = dict(state)
    if stage_decision == "supports_serrated_lesion":
        next_state["serrated"] = "supported"
    elif stage_decision == "leans_non_serrated_or_indeterminate":
        next_state["serrated"] = "opposed"
    elif stage_decision == "supports_abnormal_crypt":
        next_state["abnormal_crypt"] = "supported"
    elif stage_decision == "serrated_but_no_support_for_abnormal_crypt":
        next_state["abnormal_crypt"] = "opposed"
    elif stage_decision == "supports_conventional_adenoma":
        next_state["conventional"] = "supported"
    elif stage_decision == "conventional_adenoma_indeterminate_or_opposed":
        next_state["conventional"] = "opposed"
    elif stage_decision in {"serrated_dysplasia_supported", "conventional_dysplasia_supported"}:
        next_state["dysplasia"] = "supported"
    elif stage_decision in {
        "serrated_dysplasia_not_supported_or_indeterminate",
        "conventional_dysplasia_not_supported_or_indeterminate",
    }:
        next_state["dysplasia"] = "opposed"
    return next_state


def _synthetic_chief_thinking(observation, reasoning_state):
    metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
    parts = [
        "Chief 汇总当前局部观察，并与累计分支状态进行比对。",
        "当前 cluster_id: {0}".format(metadata.get("cluster_id", "-")),
        "review_goal: {0}".format(metadata.get("review_goal", "-")),
        "stage_decision: {0}".format(observation.get("stage_decision", "-")),
    ]
    hypotheses = reasoning_state.get("hypotheses", []) if isinstance(reasoning_state, dict) else []
    if hypotheses:
        parts.append("当前全局假设: {0}".format(", ".join(str(item) for item in hypotheses[:3])))
    return "\n".join(parts)


def _build_fallback_global_reviews(observations, trace_cluster_lookup, reasoning_state, observe_report):
    reviews = []
    current_state = {
        "serrated": "unresolved",
        "abnormal_crypt": "unresolved",
        "conventional": "unresolved",
        "dysplasia": "unresolved",
    }
    supporting_evidence = list(reasoning_state.get("supporting_evidence", [])) if isinstance(reasoning_state, dict) else []
    observe_report_summary = ""
    integrated_report = observe_report.get("integrated_report", {}) if isinstance(observe_report, dict) else {}
    if isinstance(integrated_report, dict):
        observe_report_summary = str(integrated_report.get("summary", "")).strip()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            continue
        metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
        step_id = str(observation.get("step_id", "")).strip() or "step_{0}".format(index)
        stage_decision = str(observation.get("stage_decision", "")).strip()
        current_state = _update_branch_state_from_stage_decision(current_state, stage_decision)
        next_observation = observations[index + 1] if index + 1 < len(observations) else None
        if next_observation:
            next_meta = next_observation.get("metadata", {}) if isinstance(next_observation.get("metadata"), dict) else {}
            target_cluster_id = str(next_meta.get("cluster_id", "")).strip()
            target_cluster = trace_cluster_lookup.get(target_cluster_id, {})
            target_region_semantic = str(target_cluster.get("cluster_label", "") or target_cluster.get("l", "") or next_meta.get("need_to_see", "")).strip()
            if target_region_semantic not in LABEL_ORDER:
                target_region_semantic = "normal_mucosa"
            next_visual_target = {
                "target_cluster_id": target_cluster_id or "unknown_cluster",
                "target_branch": _review_goal_branch(next_meta.get("review_goal", "")),
                "target_region_semantic": target_region_semantic,
                "target_morphology_prompt": [str(next_meta.get("need_to_see", "")).strip()] if str(next_meta.get("need_to_see", "")).strip() else [],
                "preferred_magnification": _preferred_contract_magnification(next_observation),
                "priority_reason": "Chief 认为当前证据尚不足以早停，需要继续沿后续视角补强证据链。",
            }
            review = {
                "review_id": "global_review_{0:04d}".format(index + 1),
                "source_step_id": step_id,
                "decision": "continue",
                "continue_reason": "当前全局证据尚未闭合，需要继续扫描下一观察目标。",
                "chief_confidence": _safe_float(observation.get("confidence"), 0.5),
                "resolved_branch_state": dict(current_state),
                "sufficient_evidence": [],
                "unresolved_questions": [str(next_meta.get("need_to_see", "")).strip()] if str(next_meta.get("need_to_see", "")).strip() else [],
                "next_visual_target": next_visual_target,
                "branch_correction_reason": "",
                "chief_thinking": _synthetic_chief_thinking(observation, reasoning_state),
                "is_fallback_review": True,
            }
        else:
            evidence = [str(item) for item in supporting_evidence if str(item).strip()]
            if observe_report_summary:
                evidence.insert(0, observe_report_summary)
            if not evidence:
                evidence = [str(observation.get("reasoning", "")).strip() or "当前轨迹已完成，Chief 允许 early stop。"]
            review = {
                "review_id": "global_review_{0:04d}".format(index + 1),
                "source_step_id": step_id,
                "decision": "early_stop",
                "continue_reason": "",
                "chief_confidence": _safe_float(observation.get("confidence"), 0.5),
                "resolved_branch_state": dict(current_state),
                "sufficient_evidence": evidence,
                "unresolved_questions": [],
                "next_visual_target": None,
                "branch_correction_reason": "",
                "chief_thinking": _synthetic_chief_thinking(observation, reasoning_state) + "\nChief 判定轨迹已完成，可生成最终报告。",
                "is_fallback_review": True,
            }
        reviews.append(review)
    return reviews


def _build_observation_contract_validation(observations, global_reviews, observe_report):
    violations = []
    observations = observations if isinstance(observations, list) else []
    global_reviews = global_reviews if isinstance(global_reviews, list) else []
    if not global_reviews:
        violations.append({"type": "missing_global_reviews", "message": "缺失 global_reviews，当前界面使用 fallback chief review。"})
    if len(global_reviews) > len(observations):
        violations.append({"type": "review_observation_length_mismatch", "message": "global_reviews 数量不能超过 observations。"})
    observation_step_ids = {str(item.get("step_id", "")).strip() for item in observations if isinstance(item, dict)}
    for review in global_reviews:
        step_id = str(review.get("source_step_id", "")).strip()
        if step_id not in observation_step_ids:
            violations.append({"type": "source_step_id_mismatch", "step_id": step_id, "message": "global_review.source_step_id 未指向已有 observation.step_id。"})
            continue
        if str(review.get("source_step_id", "")).strip() != step_id:
            violations.append({"type": "source_step_id_mismatch", "step_id": step_id, "message": "global_review.source_step_id 与 observation.step_id 不一致。"})
        decision = str(review.get("decision", "")).strip()
        if decision == "continue":
            if not str(review.get("continue_reason", "")).strip():
                violations.append({"type": "missing_continue_reason", "step_id": step_id, "message": "decision=continue 但缺失 continue_reason。"})
            if review.get("next_visual_target") in (None, {}, []):
                violations.append({"type": "missing_next_visual_target", "step_id": step_id, "message": "decision=continue 但缺失 next_visual_target。"})
        if decision == "early_stop":
            evidence = review.get("sufficient_evidence", [])
            if not evidence:
                violations.append({"type": "missing_sufficient_evidence", "step_id": step_id, "message": "decision=early_stop 但 sufficient_evidence 为空。"})
    if not any(str(item.get("decision", "")).strip() == "early_stop" for item in global_reviews):
        violations.append({"type": "missing_early_stop", "message": "最终没有检测到 early_stop。"})
    if not observe_report:
        violations.append({"type": "missing_observe_report", "message": "缺失 observe_report。"})
    return {
        "status": "warning" if violations else "ok",
        "violations": violations,
    }


def _build_observation_view_model(payload):
    observations = payload.get("observations", [])
    global_reviews = payload.get("global_reviews", [])
    observe_assets = payload.get("observe_assets_by_step", [])
    observe_report = payload.get("observe_report", {}) if isinstance(payload.get("observe_report"), dict) else {}
    contract_validation = payload.get("observation_contract_validation", {"status": "ok", "violations": []})
    observe_by_step = {str(item.get("step_id", "")).strip(): item for item in observe_assets if isinstance(item, dict)}
    review_by_step = {str(item.get("source_step_id", "")).strip(): item for item in global_reviews if isinstance(item, dict)}
    cards = []
    selected_step_id = None
    unlocked = any(str(review.get("decision", "")).strip() == "early_stop" for review in global_reviews if isinstance(review, dict))
    for index, observation in enumerate(observations if isinstance(observations, list) else []):
        if not isinstance(observation, dict):
            continue
        review = review_by_step.get(str(observation.get("step_id", "")).strip(), {})
        metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
        assets = observe_by_step.get(str(observation.get("step_id", "")).strip(), {})
        decision = str(review.get("decision", "")).strip()
        card = {
            "step_id": str(observation.get("step_id", "")).strip() or "step_{0}".format(index),
            "cluster_id": str(metadata.get("cluster_id", "")).strip(),
            "review_goal": str(metadata.get("review_goal", "")).strip(),
            "stage_gate": str(metadata.get("stage_gate", "")).strip(),
            "chief_confidence": _safe_float(review.get("chief_confidence"), observation.get("confidence", 0.5)),
            "observation_confidence": _safe_float(observation.get("confidence"), 0.5),
            "observation": str(observation.get("observation", "")).strip(),
            "reasoning": str(observation.get("reasoning", "")).strip(),
            "level_1_findings": list(observation.get("level_1_findings", [])),
            "level_2_findings": list(observation.get("level_2_findings", [])),
            "level_3_findings": list(observation.get("level_3_findings", [])),
            "stage_decision": str(observation.get("stage_decision", "")).strip(),
            "decision": decision or "continue",
            "continue_reason": str(review.get("continue_reason", "")).strip(),
            "resolved_branch_state": dict(review.get("resolved_branch_state", {})) if isinstance(review.get("resolved_branch_state"), dict) else {},
            "sufficient_evidence": list(review.get("sufficient_evidence", [])),
            "unresolved_questions": list(review.get("unresolved_questions", [])),
            "next_visual_target": review.get("next_visual_target") if isinstance(review.get("next_visual_target"), dict) else None,
            "branch_correction_reason": str(review.get("branch_correction_reason", "")).strip(),
            "chief_thinking": str(review.get("chief_thinking", "")).strip(),
            "overview_image": str(assets.get("overview_image", "")),
            "local_image": str(assets.get("local_image", "")),
            "detail_image": str(assets.get("detail_image", "")),
            "decision_class": "early-stop" if decision == "early_stop" else "continue",
            "is_early_stop": decision == "early_stop",
            "memory_observation_count": index + 1,
            "memory_review_count": min(index + 1, len(global_reviews)),
        }
        cards.append(card)
        if selected_step_id is None:
            selected_step_id = card["step_id"]

    branch_state = {"serrated": "unresolved", "abnormal_crypt": "unresolved", "conventional": "unresolved", "dysplasia": "unresolved"}
    if cards:
        for key, value in cards[-1]["resolved_branch_state"].items():
            if key in branch_state and value in BRANCH_STATE_DISPLAY:
                branch_state[key] = value

    report_sections = []
    for key in [
        "serrated_checklist",
        "abnormal_crypt_checklist",
        "conventional_adenoma_checklist",
        "serrated_dysplasia_checklist",
        "conventional_dysplasia_checklist",
        "dysplasia_checklist",
    ]:
        value = observe_report.get(key, [])
        if isinstance(value, list):
            report_sections.append({"key": key, "items": value})
    integrated_report = observe_report.get("integrated_report", {}) if isinstance(observe_report, dict) else {}
    if isinstance(integrated_report, dict):
        integrated_summary = str(integrated_report.get("summary", "")).strip()
        integrated_recommendations = list(integrated_report.get("recommendations", []))
    else:
        integrated_summary = str(integrated_report).strip()
        integrated_recommendations = []
    return {
        "cards": cards,
        "branch_state": branch_state,
        "selected_step_id": selected_step_id,
        "unlocked": unlocked,
        "contract_validation": contract_validation,
        "report_sections": report_sections,
        "hierarchical_prediction": observe_report.get("hierarchical_prediction", {}),
        "integrated_summary": integrated_summary,
        "integrated_recommendations": integrated_recommendations,
    }


def _report_summary_from_sources(observe_report, case_result):
    observe_report = observe_report if isinstance(observe_report, dict) else {}
    case_result = case_result if isinstance(case_result, dict) else {}
    integrated = observe_report.get("integrated_report")
    if isinstance(integrated, dict):
        summary = str(integrated.get("summary", "")).strip()
        recommendations = list(integrated.get("recommendations", [])) if isinstance(integrated.get("recommendations", []), list) else []
    else:
        summary = str(integrated or "").strip()
        recommendations = []
    if not summary and isinstance(case_result.get("integrated_report"), str):
        summary = str(case_result.get("integrated_report", "")).strip()
    prediction = observe_report.get("hierarchical_prediction") or case_result.get("hierarchical_prediction") or {}
    return {
        "hierarchical_prediction": prediction if isinstance(prediction, dict) else {},
        "summary": summary,
        "recommendations": recommendations,
        "report_ready": bool(summary or prediction),
    }


def _case_story_stage_label(observation):
    metadata = observation.get("metadata", {}) if isinstance(observation.get("metadata"), dict) else {}
    step_id = str(observation.get("step_id", "")).strip()
    return {
        "step_id": step_id,
        "cluster_id": str(metadata.get("cluster_id", "")).strip(),
        "cluster_label": str(metadata.get("cluster_label", "")).strip(),
        "workflow_branch": str(metadata.get("workflow_branch", "")).strip(),
        "review_goal": str(metadata.get("review_goal", "")).strip(),
        "stage_gate": str(metadata.get("stage_gate", "")).strip(),
        "stage_decision": str(observation.get("stage_decision", "")).strip(),
        "magnification": metadata.get("magnification"),
        "is_dynamic": step_id.startswith("step_dyn_"),
    }


def _build_case_story_view_model(payload):
    observations = payload.get("observations", []) if isinstance(payload.get("observations"), list) else []
    navigation_steps = payload.get("navigation_steps", []) if isinstance(payload.get("navigation_steps"), list) else []
    trace_clusters = payload.get("trace_clusters", []) if isinstance(payload.get("trace_clusters"), list) else []
    observe_assets = payload.get("observe_assets_by_step", []) if isinstance(payload.get("observe_assets_by_step"), list) else []
    global_reviews = payload.get("global_reviews", []) if isinstance(payload.get("global_reviews"), list) else []
    observe_report = payload.get("observe_report", {}) if isinstance(payload.get("observe_report"), dict) else {}
    case_result = payload.get("case_result", {}) if isinstance(payload.get("case_result"), dict) else {}
    reasoning_state = payload.get("reasoning_state", {}) if isinstance(payload.get("reasoning_state"), dict) else {}

    assets_by_step = {str(item.get("step_id", "")).strip(): item for item in observe_assets if isinstance(item, dict)}
    review_by_step = {str(item.get("source_step_id", "")).strip(): item for item in global_reviews if isinstance(item, dict)}
    navigation_by_step = {str(item.get("step_id", "")).strip(): item for item in navigation_steps if isinstance(item, dict)}
    observed_step_ids = {str(item.get("step_id", "")).strip() for item in observations if isinstance(item, dict)}

    branch_counts = {"serrated": 0, "abnormal_crypt": 0, "conventional": 0, "dysplasia": 0, "inflammatory": 0, "background": 0}
    story_steps = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            continue
        stage = _case_story_stage_label(observation)
        step_id = stage["step_id"] or "obs_{0}".format(index)
        assets = assets_by_step.get(step_id, {})
        review = review_by_step.get(step_id, {})
        nav_step = navigation_by_step.get(step_id, {})
        branch = stage["workflow_branch"] or _review_goal_branch(stage["review_goal"])
        if branch in branch_counts:
            branch_counts[branch] += 1
        if "dysplasia" in stage["stage_decision"] or "dysplasia" in stage["review_goal"]:
            branch_counts["dysplasia"] += 1
        story_steps.append(
            {
                "step_id": step_id,
                "index": index,
                "cluster_id": stage["cluster_id"],
                "cluster_label": stage["cluster_label"],
                "workflow_branch": branch,
                "review_goal": stage["review_goal"],
                "stage_gate": stage["stage_gate"],
                "stage_decision": stage["stage_decision"],
                "magnification": stage["magnification"],
                "is_dynamic": bool(stage["is_dynamic"] or assets.get("is_dynamic")),
                "generated_by": str(assets.get("generated_by", "")).strip(),
                "gate_source_step": str(assets.get("gate_source_step", "")).strip(),
                "gate_source": str(assets.get("gate_source", "")).strip(),
                "observation": str(observation.get("observation", "")).strip(),
                "reasoning": str(observation.get("reasoning", "")).strip(),
                "next_step": str(observation.get("next_step", "")).strip(),
                "confidence": _safe_float(observation.get("confidence"), 0.0),
                "overview_image": str(assets.get("overview_image", "")),
                "local_image": str(assets.get("local_image", "")),
                "detail_image": str(assets.get("detail_image", "")),
                "navigation_status": "planned_observed" if nav_step else "dynamic_observed",
                "chief_review": {
                    "decision": str(review.get("decision", "")).strip(),
                    "continue_reason": str(review.get("continue_reason", "")).strip(),
                    "chief_confidence": _safe_float(review.get("chief_confidence"), 0.0),
                    "resolved_branch_state": dict(review.get("resolved_branch_state", {})) if isinstance(review.get("resolved_branch_state"), dict) else {},
                    "sufficient_evidence": list(review.get("sufficient_evidence", [])) if isinstance(review.get("sufficient_evidence", []), list) else [],
                    "unresolved_questions": list(review.get("unresolved_questions", [])) if isinstance(review.get("unresolved_questions", []), list) else [],
                    "next_visual_target": review.get("next_visual_target") if isinstance(review.get("next_visual_target"), dict) else None,
                    "branch_correction_reason": str(review.get("branch_correction_reason", "")).strip(),
                    "thought_text": str(review.get("chief_thinking", "")).strip(),
                    "answer_candidate_text": str(review.get("answer_candidate_text", "")).strip(),
                    "parse_error": str(review.get("parse_error", "")).strip(),
                    "review_source": str(review.get("review_source", "")).strip(),
                    "model_name": str(review.get("model_name", "")).strip(),
                    "round_trip_ms": _safe_int(review.get("round_trip_ms"), 0),
                },
            }
        )

    planned_steps = []
    for index, step in enumerate(navigation_steps):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata", {}) if isinstance(step.get("metadata"), dict) else {}
        step_id = str(step.get("step_id", "")).strip() or "step_{0}".format(index)
        action = _normalize_navigation_action(step)
        planned_steps.append(
            {
                "step_id": step_id,
                "index": index,
                "m": step.get("m"),
                "action": action,
                "cluster_id": str(metadata.get("cluster_id", "")).strip(),
                "cluster_label": str(metadata.get("cluster_label", "")).strip(),
                "review_goal": str(step.get("review_goal", "")).strip(),
                "stage_gate": str(step.get("stage_gate", "")).strip(),
                "is_observed": step_id in observed_step_ids,
                "is_stop": action == "stop",
            }
        )

    clusters = []
    observed_clusters = {step["cluster_id"] for step in story_steps if step.get("cluster_id")}
    planned_clusters = {step["cluster_id"] for step in planned_steps if step.get("cluster_id")}
    for cluster in trace_clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id", "")).strip()
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": str(cluster.get("cluster_label", "") or cluster.get("l", "")).strip(),
                "cluster_priority": _safe_int(cluster.get("cluster_priority"), _safe_int(cluster.get("s"), 0)),
                "workflow_branch": str(cluster.get("workflow_branch", "") or (cluster.get("metadata", {}) or {}).get("workflow_branch", "")).strip(),
                "planned": cluster_id in planned_clusters,
                "observed": cluster_id in observed_clusters,
            }
        )

    report = _report_summary_from_sources(observe_report, case_result)
    report["ground_truth_label"] = payload.get("ground_truth_label", {})
    report["final_report_comparison"] = payload.get("final_report_comparison", {})
    final_status = str(reasoning_state.get("stop_reason") or case_result.get("status") or case_result.get("final_status") or "unknown")
    chief_parse_errors = sum(1 for step in story_steps if step["chief_review"].get("parse_error"))
    chief_model_reviews = sum(1 for step in story_steps if step["chief_review"].get("review_source") == "chief_model")
    return {
        "summary": {
            "case_id": str(payload.get("case_id", "")),
            "final_status": final_status,
            "observation_count": len(story_steps),
            "navigation_count": len(planned_steps),
            "dynamic_step_count": sum(1 for step in story_steps if step.get("is_dynamic")),
            "chief_model_review_count": chief_model_reviews,
            "chief_parse_error_count": chief_parse_errors,
            "report_ready": report["report_ready"],
            "branch_counts": branch_counts,
        },
        "steps": story_steps,
        "planned_steps": planned_steps,
        "clusters": clusters,
        "report": report,
        "selected_step_id": story_steps[0]["step_id"] if story_steps else None,
    }


def _normalize_contract_validation(payload):
    selected = set(_selected_patch_ids_from_payload(payload))
    grid_meta = payload.get("grid_metadata", {})
    grid_rows = _safe_int(grid_meta.get("grid_rows"), 0)
    grid_cols = _safe_int(grid_meta.get("grid_cols"), 0)
    patch_assignments = payload.get("patch_assignments", [])
    contract_validation = payload.get("contract_validation", {}) if isinstance(payload.get("contract_validation"), dict) else {}
    violations = list(contract_validation.get("violations", [])) if isinstance(contract_validation.get("violations"), list) else []
    seen = set()
    duplicate_ids = set()
    orphan_assets = list(payload.get("orphan_assets", [])) if isinstance(payload.get("orphan_assets"), list) else []
    orphan_keys = {_patch_key(item.get("patch_id")) for item in orphan_assets if isinstance(item, dict)}
    valid_assignments = []

    for patch in patch_assignments:
        if not isinstance(patch, dict):
            continue
        row_col = _normalize_patch_id(patch.get("patch_id") or [patch.get("row"), patch.get("col")])
        if row_col is None:
            violations.append(
                {
                    "type": "illegal_patch_id",
                    "patch_id": patch.get("patch_id"),
                    "message": "Patch ID 无法被规范化解析。",
                }
            )
            continue
        key = _patch_key(row_col)
        if key in seen:
            duplicate_ids.add(key)
            violations.append(
                {
                    "type": "duplicate_patch",
                    "patch_id": list(row_col),
                    "message": "同一个 Patch 在 patch_assignments 中出现了多次。",
                }
            )
            continue
        seen.add(key)
        row = _safe_int(patch.get("row"), row_col[0])
        col = _safe_int(patch.get("col"), row_col[1])
        if row < 0 or col < 0 or row >= grid_rows or col >= grid_cols:
            if key not in orphan_keys:
                orphan_assets.append(
                    {
                        "patch_id": list(row_col),
                        "reason": "out_of_bounds_patch",
                        "score": patch.get("score"),
                        "region_semantic": patch.get("region_semantic"),
                        "preview_ref": patch.get("high_mag_ref", ""),
                    }
                )
                orphan_keys.add(key)
            violations.append(
                {
                    "type": "out_of_bounds_patch",
                    "patch_id": list(row_col),
                    "row": row,
                    "col": col,
                    "message": "Patch 坐标超出了 grid_rows/grid_cols 边界。",
                }
            )
            continue
        if selected and row_col not in selected:
            if key not in orphan_keys:
                orphan_assets.append(
                    {
                        "patch_id": list(row_col),
                        "reason": "orphan_patch",
                        "score": patch.get("score"),
                        "region_semantic": patch.get("region_semantic"),
                        "preview_ref": patch.get("high_mag_ref", ""),
                    }
                )
                orphan_keys.add(key)
            violations.append(
                {
                    "type": "orphan_patch",
                    "patch_id": list(row_col),
                    "message": "该 Patch 不属于 selected_patch_ids。",
                }
            )
            continue
        patch["row"] = row
        patch["col"] = col
        valid_assignments.append(patch)

    assigned = {_normalize_patch_id(patch.get("patch_id") or [patch.get("row"), patch.get("col")]) for patch in valid_assignments}
    missing_ids = []
    for row_col in sorted(selected):
        if row_col not in assigned:
            missing_ids.append(list(row_col))
            violations.append(
                {
                    "type": "missing_patch",
                    "patch_id": list(row_col),
                    "message": "该 Patch 已被选中，但在 patch_assignments 中缺失。",
                }
            )

    type_counts = {}
    deduped = []
    seen_violation = set()
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        key = json.dumps(violation, sort_keys=True, ensure_ascii=False)
        if key in seen_violation:
            continue
        seen_violation.add(key)
        vtype = str(violation.get("type", "unknown"))
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
        deduped.append(violation)

    status = "ok"
    if deduped:
        status = contract_validation.get("status") or "warning"

    return {
        "selected_patch_ids": [list(item) for item in selected],
        "valid_patch_assignments": valid_assignments,
        "missing_patch_ids": missing_ids,
        "duplicate_patch_ids": sorted(list(duplicate_ids)),
        "orphan_assets": orphan_assets,
        "contract_validation": {
            "status": status,
            "violations": deduped,
            "counts": type_counts,
        },
    }


def _serialize_patch(patch, cell, canvas_width, canvas_height, assets_by_ref, audited_keys):
    row_col = _normalize_patch_id(patch.get("patch_id") or [patch.get("row"), patch.get("col")])
    key = _patch_key(row_col)
    label = str(patch.get("region_semantic", "unknown"))
    style = TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])
    left = 100.0 * _safe_float(cell.get("thumbnail_top_left_x")) / float(canvas_width)
    top = 100.0 * _safe_float(cell.get("thumbnail_top_left_y")) / float(canvas_height)
    width = 100.0 * _safe_float(cell.get("thumbnail_width")) / float(canvas_width)
    height = 100.0 * _safe_float(cell.get("thumbnail_height")) / float(canvas_height)
    score = _safe_float(patch.get("score"))
    uncertainty = _safe_float(patch.get("uncertainty_score"), None if patch.get("uncertainty_score") is None else patch.get("uncertainty_score"))
    agreement = str(patch.get("agreement_status", "strong_agreement"))
    high_mag = assets_by_ref.get(str(patch.get("high_mag_ref", "")), {})
    preview_url = high_mag.get("resolved_image_url", "")
    return {
        "patch_key": key,
        "patch_id": list(row_col) if row_col is not None else patch.get("patch_id"),
        "patch_text": "[{0},{1}]".format(int(row_col[0]), int(row_col[1])) if row_col is not None else str(patch.get("patch_id")),
        "row": _safe_int(patch.get("row")),
        "col": _safe_int(patch.get("col")),
        "cluster_id": str(patch.get("cluster_id", "")),
        "score_origin": str(patch.get("score_origin", "")),
        "label": label,
        "label_short": LABEL_SHORT.get(label, label),
        "score": score,
        "score_bucket": _grade_score_bucket(score),
        "uncertainty_score": uncertainty,
        "agreement_status": agreement,
        "diagnostic_priority": _safe_int(patch.get("diagnostic_priority"), 0),
        "require_high_magnification": bool(patch.get("require_high_magnification", False)),
        "bbox_level0": patch.get("bbox_level0", []),
        "conch_region_semantic": str(patch.get("conch_region_semantic", "")),
        "pathoreasoner_r1_region_semantic": str(patch.get("pathoreasoner_r1_region_semantic", "")),
        "fusion_reasoning": str(patch.get("fusion_reasoning", "")),
        "high_mag_ref": str(patch.get("high_mag_ref", "")),
        "high_mag_url": preview_url,
        "left": round(left, 4),
        "top": round(top, 4),
        "width": round(width, 4),
        "height": round(height, 4),
        "fill": _rgba_with_alpha(style["fill"], _alpha_from_score(score)),
        "stroke": style["stroke"],
        "text": style["text"],
        "audited": key in audited_keys,
        "is_uncertain_focus": agreement == "risk_disagreement" or (0.4 <= score <= 0.6),
    }


def _serialize_missing_patch(row_col, cell, canvas_width, canvas_height):
    left = 100.0 * _safe_float(cell.get("thumbnail_top_left_x")) / float(canvas_width)
    top = 100.0 * _safe_float(cell.get("thumbnail_top_left_y")) / float(canvas_height)
    width = 100.0 * _safe_float(cell.get("thumbnail_width")) / float(canvas_width)
    height = 100.0 * _safe_float(cell.get("thumbnail_height")) / float(canvas_height)
    return {
        "patch_key": _patch_key(row_col),
        "patch_text": "[{0},{1}]".format(int(row_col[0]), int(row_col[1])),
        "left": round(left, 4),
        "top": round(top, 4),
        "width": round(width, 4),
        "height": round(height, 4),
        "type": "missing_patch",
    }


def _serialize_orphan_assets(orphan_assets):
    rows = []
    for item in orphan_assets:
        if not isinstance(item, dict):
            continue
        row_col = _normalize_patch_id(item.get("patch_id"))
        rows.append(
            {
                "patch_key": _patch_key(row_col) if row_col is not None else str(item.get("patch_id")),
                "patch_text": "[{0},{1}]".format(int(row_col[0]), int(row_col[1])) if row_col is not None else str(item.get("patch_id")),
                "reason": str(item.get("reason", "orphan_patch")),
                "score": _safe_float(item.get("score")),
                "region_semantic": str(item.get("region_semantic", "unknown")),
                "preview_ref": str(item.get("preview_ref", "")),
            }
        )
    return rows


def _compute_stats(serialized_patches, missing_count, orphan_count, audit_log):
    label_counts = {label: 0 for label in LABEL_ORDER}
    high_priority_count = 0
    disagreement_count = 0
    uncertain_count = 0
    for patch in serialized_patches:
        label = patch["label"]
        if label in label_counts:
            label_counts[label] += 1
        if patch["diagnostic_priority"] >= 3:
            high_priority_count += 1
        if patch["agreement_status"] == "risk_disagreement":
            disagreement_count += 1
        if patch["is_uncertain_focus"]:
            uncertain_count += 1
    return {
        "total_patches": len(serialized_patches),
        "filtered_patches": len(serialized_patches),
        "high_priority_patches": high_priority_count,
        "audited_patches": len(audit_log),
        "disagreement_patches": disagreement_count,
        "uncertain_focus_patches": uncertain_count,
        "missing_patches": missing_count,
        "orphan_patches": orphan_count,
        "label_counts": label_counts,
    }


def _resolve_asset_urls(payload, output_dir, source_root=None):
    overview = dict(payload.get("overview_image", {})) if isinstance(payload.get("overview_image"), dict) else {}
    high_mag_assets = []
    for asset in payload.get("high_mag_assets", []):
        if isinstance(asset, dict):
            high_mag_assets.append(dict(asset))

    overview_ref = overview.get("image_path") or overview.get("image_url")
    overview_url = _materialize_asset_url(overview_ref, output_dir, "overview", source_root=source_root)
    if not overview_url:
        width = _safe_int(overview.get("width"), 1000) or 1000
        height = _safe_int(overview.get("height"), 1000) or 1000
        overview["resolved_image_url"] = _overview_background_svg(width, height)
    else:
        overview["resolved_image_url"] = overview_url

    for asset in high_mag_assets:
        image_ref = asset.get("image_path") or asset.get("image_url")
        image_url = _materialize_asset_url(image_ref, output_dir, "high_mag", source_root=source_root)
        if image_url:
            asset["resolved_image_url"] = image_url
            continue
        row_col = _normalize_patch_id(asset.get("patch_id"))
        patch_text = "[{0},{1}]".format(int(row_col[0]), int(row_col[1])) if row_col is not None else str(asset.get("patch_id"))
        asset["resolved_image_url"] = _high_mag_svg(asset.get("data_label"), patch_text)
    return overview, high_mag_assets


def _html_for_dashboard(context):
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #efe7da;
      --surface: rgba(255, 250, 243, 0.88);
      --surface-strong: #fffaf2;
      --surface-alt: rgba(255, 244, 231, 0.94);
      --ink: #271713;
      --muted: #7b6a64;
      --line: rgba(109, 74, 63, 0.18);
      --accent: #9f3f2f;
      --accent-soft: rgba(159, 63, 47, 0.12);
      --danger: #b91c1c;
      --warning: #b45309;
      --green: #166534;
      --shadow: 0 18px 40px rgba(86, 48, 39, 0.14);
      --round: 20px;
      --sidebar-left: 320px;
      --sidebar-right: 400px;
      --sidebar-left-width: 320px;
      --sidebar-right-width: 400px;
      --resizer-width: 18px;
      --serif: "Noto Serif SC", "Songti SC", "STSong", "Iowan Old Style", Georgia, serif;
      --sans: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "Avenir Next", "Segoe UI", Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background:
      radial-gradient(circle at top left, rgba(248,219,192,0.75), transparent 28%),
      radial-gradient(circle at 90% 10%, rgba(221,177,170,0.42), transparent 24%),
      linear-gradient(180deg, #f7efe5 0%, #ebdfd1 100%);
      color: var(--ink); font: 14px/1.45 var(--sans); min-height: 100%;
    }}
    body {{ padding: 18px; }}
    .dashboard-page {{
      display: grid;
      gap: 16px;
    }}
    .view-switcher-panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--round);
      box-shadow: var(--shadow);
      padding: 16px 18px;
    }}
    .dashboard-error-banner {{
      display: none;
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(185, 28, 28, 0.24);
      background: rgba(254, 242, 242, 0.9);
      color: #991b1b;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .dashboard-error-banner.is-visible {{
      display: block;
    }}
    .view-switcher-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .view-tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .view-tab {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      font: 700 13px/1 var(--sans);
      cursor: pointer;
    }}
    .view-tab.is-active {{
      border-color: var(--accent);
      background: linear-gradient(135deg, rgba(159,63,47,0.15), rgba(245,158,11,0.1));
      color: var(--accent);
    }}
    .view-description {{
      color: var(--muted);
      font-size: 13px;
    }}
    .data-source-banner {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.68);
      color: var(--ink);
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    .dashboard-shell {{
      display: grid;
      grid-template-columns: var(--sidebar-left-width) var(--resizer-width) minmax(0, 1fr) var(--resizer-width) var(--sidebar-right-width);
      gap: 0;
      align-items: start;
    }}
    .dashboard-shell.is-left-collapsed {{
      grid-template-columns: 0px var(--resizer-width) minmax(0, 1fr) var(--resizer-width) var(--sidebar-right-width);
    }}
    .dashboard-shell.is-right-collapsed {{
      grid-template-columns: var(--sidebar-left-width) var(--resizer-width) minmax(0, 1fr) var(--resizer-width) 0px;
    }}
    .dashboard-shell.is-left-collapsed.is-right-collapsed {{
      grid-template-columns: 0px var(--resizer-width) minmax(0, 1fr) var(--resizer-width) 0px;
    }}
    .dashboard-column {{
      min-width: 0;
      overflow: hidden;
      transition: opacity 160ms ease, transform 160ms ease;
    }}
    .left-column {{ margin-right: 10px; }}
    .center-column {{ margin: 0 10px; }}
    .right-column {{ margin-left: 10px; }}
    .dashboard-shell.is-left-collapsed .left-column {{
      opacity: 0;
      pointer-events: none;
      transform: translateX(-12px);
    }}
    .dashboard-shell.is-right-collapsed .right-column {{
      opacity: 0;
      pointer-events: none;
      transform: translateX(12px);
    }}
    .column-resizer {{
      position: relative;
      min-height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: col-resize;
      user-select: none;
      touch-action: none;
    }}
    .column-resizer::before {{
      content: "";
      position: absolute;
      top: 18px;
      bottom: 18px;
      left: 50%;
      width: 2px;
      transform: translateX(-50%);
      border-radius: 999px;
      background: linear-gradient(180deg, rgba(109,74,63,0.12), rgba(159,63,47,0.42), rgba(109,74,63,0.12));
      transition: background 120ms ease, box-shadow 120ms ease;
    }}
    .column-resizer:hover::before,
    .column-resizer.is-dragging::before {{
      background: linear-gradient(180deg, rgba(159,63,47,0.18), rgba(159,63,47,0.8), rgba(159,63,47,0.18));
      box-shadow: 0 0 0 4px rgba(159,63,47,0.08);
    }}
    .resizer-toggle {{
      position: relative;
      z-index: 1;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,250,243,0.95);
      color: var(--accent);
      font: 700 11px/1 var(--sans);
      cursor: pointer;
      box-shadow: 0 8px 18px rgba(63,31,26,0.08);
    }}
    .panel {{
      background: var(--surface);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: var(--round);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-header {{
      padding: 18px 20px 12px;
      border-bottom: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.54), rgba(255,255,255,0.12)),
        linear-gradient(135deg, rgba(255,244,230,0.94), rgba(255,239,227,0.72));
    }}
    .panel-header-actions {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .panel-title {{
      margin: 0;
      font: 700 19px/1.15 var(--serif);
      letter-spacing: 0.01em;
    }}
    .panel-kicker {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
    }}
    .panel-body {{ padding: 18px 20px 22px; }}
    .panel-collapse-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 10px;
      background: rgba(255,255,255,0.74);
      color: var(--accent);
      font: 700 12px/1 var(--sans);
      cursor: pointer;
      white-space: nowrap;
    }}
    .stack {{ display: grid; gap: 14px; }}
    .view-pane {{
      display: none;
    }}
    .view-pane.is-active {{
      display: grid;
      gap: 14px;
    }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .meta-card, .stat-card, .control-card, .legend-item, .violation-item, .audit-item, .orphan-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.52);
    }}
    .meta-card, .stat-card, .control-card, .violation-item, .audit-item, .orphan-card {{ padding: 12px 13px; }}
    .meta-label, .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      margin-bottom: 6px;
    }}
    .meta-value, .stat-value {{
      display: block;
      font-size: 15px;
      font-weight: 700;
    }}
    .meta-value {{
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: break-word;
      line-height: 1.35;
    }}
    .rubric-list, .stats-grid, .toggle-grid, .filter-grid, .legend-list, .detail-list {{
      display: grid;
      gap: 10px;
    }}
    .rubric-list {{
      grid-template-columns: 1fr;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
    }}
    .legend-swatch {{
      width: 16px;
      height: 16px;
      border-radius: 999px;
      border: 2px solid rgba(0,0,0,0.1);
      flex: none;
    }}
    .legend-copy strong {{
      display: block;
      font-size: 13px;
    }}
    .legend-copy span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,0.82);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, background 120ms ease;
      user-select: none;
    }}
    .chip:hover {{ transform: translateY(-1px); box-shadow: 0 10px 18px rgba(63, 31, 26, 0.08); }}
    .chip.is-active {{ border-color: var(--accent); background: var(--accent-soft); color: var(--accent); }}
    .chip-color-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: none;
    }}
    .toggle-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .toggle-button {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
      background: rgba(255,255,255,0.74);
      text-align: left;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
    }}
    .toggle-button.is-active {{
      border-color: var(--accent);
      background: linear-gradient(135deg, rgba(159,63,47,0.15), rgba(245,158,11,0.1));
      transform: translateY(-1px);
    }}
    .toggle-button strong {{
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .toggle-button span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .stats-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .center-column {{
      display: grid;
      gap: 16px;
    }}
    .hero {{
      padding: 22px 24px 8px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font: 700 30px/1.05 var(--serif);
    }}
    .hero p {{
      margin: 0 0 16px;
      color: var(--muted);
      max-width: 78ch;
    }}
    .hero-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      padding-bottom: 14px;
    }}
    .hero-pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.6);
      font-size: 12px;
    }}
    .canvas-panel {{
      padding: 0;
      overflow: hidden;
    }}
    .canvas-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,255,255,0));
    }}
    .canvas-toolbar .left, .canvas-toolbar .right {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }}
    .canvas-stage-wrap {{
      padding: 18px;
    }}
    .canvas-scroll {{
      position: relative;
      width: 100%;
      height: min(76vh, 920px);
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid rgba(79, 53, 47, 0.2);
      background:
        linear-gradient(135deg, rgba(65,34,28,0.14), rgba(30,20,18,0.06)),
        #f5eadf;
    }}
    .canvas-stage {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform-origin: center center;
      cursor: grab;
      user-select: none;
    }}
    .canvas-stage.is-dragging {{ cursor: grabbing; }}
    .overview-image {{
      display: block;
      width: 100%;
      height: 100%;
      border-radius: 20px;
      object-fit: cover;
      pointer-events: none;
    }}
    .overlay-layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .patch-cell, .missing-cell {{
      position: absolute;
      border-radius: 14px;
      border: 2px solid;
      pointer-events: auto;
      padding: 0;
      overflow: hidden;
      transition: opacity 120ms ease, transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease, filter 120ms ease;
    }}
    .patch-cell {{
      background: transparent;
      cursor: pointer;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
    }}
    .patch-cell.is-selected {{
      box-shadow: 0 0 0 3px rgba(255,250,244,0.92), 0 0 0 6px rgba(159,63,47,0.55);
      z-index: 5;
    }}
    .patch-cell.is-audited::after {{
      content: "";
      position: absolute;
      inset: 5px;
      border-radius: 10px;
      border: 2px dashed rgba(255,255,255,0.88);
      opacity: 0.9;
    }}
    .patch-cell.is-uncertain {{
      box-shadow: 0 0 0 3px rgba(255,255,255,0.92), 0 0 26px rgba(159,63,47,0.3);
    }}
    .patch-cell.is-muted {{
      opacity: 0.1 !important;
      filter: saturate(0.5);
    }}
    .patch-cell.is-dimmed {{
      opacity: 0.16 !important;
    }}
    .patch-cell.is-hidden {{
      display: none;
    }}
    .patch-cell:hover {{ transform: translateY(-1px); }}
    .patch-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 7px;
      border-bottom-right-radius: 10px;
      background: rgba(255,255,255,0.88);
      color: #2b1915;
      font-size: 11px;
      font-weight: 800;
      backdrop-filter: blur(6px);
    }}
    .patch-corner {{
      position: absolute;
      right: 8px;
      bottom: 8px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 26px;
      padding: 4px 6px;
      border-radius: 999px;
      background: rgba(35,24,22,0.72);
      color: #fff7f0;
      font-size: 11px;
      font-weight: 700;
      backdrop-filter: blur(6px);
    }}
    .missing-cell {{
      border-color: #111827;
      background:
        repeating-linear-gradient(135deg, rgba(17,24,39,0.95) 0 7px, rgba(255,255,255,0.12) 7px 14px);
      cursor: default;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.16);
    }}
    .canvas-footer {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 14px 18px 18px;
      border-top: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0), rgba(255,255,255,0.4));
    }}
    .footer-note {{
      color: var(--muted);
      font-size: 12px;
    }}
    .violation-banner {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(185,28,28,0.2);
      background: rgba(255, 239, 239, 0.84);
      color: #7f1d1d;
    }}
    .violation-list, .audit-list, .orphan-list {{
      display: grid;
      gap: 10px;
      max-height: 220px;
      overflow: auto;
      padding-right: 2px;
    }}
    .violation-item strong, .audit-item strong, .orphan-card strong {{
      display: block;
      margin-bottom: 4px;
      font-size: 13px;
    }}
    .tiny {{
      color: var(--muted);
      font-size: 12px;
    }}
    .inspector-image-frame {{
      position: relative;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.7), rgba(248,235,221,0.95));
      aspect-ratio: 1 / 1;
    }}
    .inspector-image {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .inspector-image-badge {{
      position: absolute;
      top: 14px;
      left: 14px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(17,24,39,0.76);
      color: #fff8f0;
      font-size: 12px;
      font-weight: 700;
      backdrop-filter: blur(8px);
    }}
    .detail-list {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .detail-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.5);
    }}
    .detail-card strong {{
      display: block;
      margin-bottom: 6px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .label-compare-card {{
      margin-top: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(245,238,225,0.66));
    }}
    .label-compare-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .label-status {{
      display: inline-flex;
      align-items: center;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      font-size: 11px;
      font-weight: 900;
      white-space: nowrap;
    }}
    .label-status.match,
    .label-status.family_match {{
      color: #14532d;
      border-color: rgba(22,101,52,0.24);
      background: rgba(220,252,231,0.9);
    }}
    .label-status.partial,
    .label-status.no_prediction {{
      color: #854d0e;
      border-color: rgba(180,83,9,0.24);
      background: rgba(254,249,195,0.95);
    }}
    .label-status.mismatch {{
      color: #7f1d1d;
      border-color: rgba(153,27,27,0.24);
      background: rgba(254,226,226,0.95);
    }}
    .label-status.missing_label {{
      color: #475569;
      border-color: rgba(100,116,139,0.24);
      background: rgba(241,245,249,0.95);
    }}
    .label-compare-grid {{
      display: grid;
      gap: 9px;
    }}
    .label-compare-row {{
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      padding: 10px;
      border: 1px solid rgba(217,224,220,0.78);
      border-radius: 14px;
      background: rgba(255,255,255,0.58);
    }}
    .label-compare-row strong {{
      font-size: 12px;
      color: var(--muted);
    }}
    .label-compare-values {{
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: #3f2b27;
      overflow-wrap: anywhere;
    }}
    .reason-panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.54);
    }}
    .reason-compare {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .reason-model {{
      border-radius: 16px;
      padding: 12px;
      color: #2b1915;
      font-weight: 700;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .reason-model.conch {{
      background: linear-gradient(135deg, rgba(255,224,102,0.48), rgba(251,191,36,0.22));
    }}
    .reason-model.patho {{
      background: linear-gradient(135deg, rgba(254,178,178,0.48), rgba(248,113,113,0.22));
    }}
    .reason-text {{
      margin: 0;
      color: #3f2b27;
      white-space: pre-wrap;
      font-size: 13px;
    }}
    .highlight-token {{
      display: inline;
      padding: 0 2px;
      border-radius: 5px;
      background: rgba(159,63,47,0.18);
      font-weight: 800;
    }}
    .audit-grid {{
      display: grid;
      gap: 10px;
    }}
    .audit-classes {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .audit-class {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
      background: rgba(255,255,255,0.74);
      text-align: left;
      cursor: pointer;
      font-weight: 700;
    }}
    .audit-class.is-selected {{
      border-color: var(--accent);
      background: rgba(159,63,47,0.12);
      color: var(--accent);
    }}
    .checkbox-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.72);
    }}
    textarea {{
      width: 100%;
      min-height: 86px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
      font: inherit;
      background: rgba(255,255,255,0.9);
      color: var(--ink);
    }}
    .button-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 10px 14px;
      font: 700 12px/1 var(--sans);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      background: rgba(255,255,255,0.86);
      color: var(--ink);
    }}
    .button.primary {{
      background: linear-gradient(135deg, #9f3f2f, #c16335);
      color: #fff7f0;
      border-color: rgba(159,63,47,0.4);
    }}
    .button.subtle.is-active {{
      border-color: var(--accent);
      color: var(--accent);
      background: rgba(159,63,47,0.08);
    }}
    .hotkeys {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .hotkey {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 9px 10px;
      background: rgba(255,255,255,0.68);
      font-size: 12px;
    }}
    .hotkey strong {{
      display: inline-block;
      margin-right: 6px;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .empty-state {{
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.46);
    }}
    .orphan-drawer summary {{
      list-style: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .orphan-drawer summary::-webkit-details-marker {{ display: none; }}
    .orphan-count {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 28px;
      height: 28px;
      border-radius: 999px;
      padding: 0 9px;
      background: rgba(185,28,28,0.12);
      color: #991b1b;
      font-size: 12px;
      font-weight: 800;
    }}
    .layout-note {{
      padding: 0 20px 16px;
      color: var(--muted);
      font-size: 12px;
    }}
    .story-summary-grid,
    .story-branch-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .story-stat,
    .story-cluster-card,
    .story-step-button,
    .story-report-card,
    .story-debug-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.58);
      padding: 12px 13px;
    }}
    .story-stat strong {{
      display: block;
      font-size: 18px;
      margin-bottom: 4px;
    }}
    .story-cluster-card.is-observed {{
      border-color: rgba(21,128,61,0.34);
      background: rgba(236,253,245,0.85);
    }}
    .story-step-list,
    .story-cluster-list,
    .story-timeline,
    .story-debug-list {{
      display: grid;
      gap: 10px;
    }}
    .story-step-button {{
      text-align: left;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }}
    .story-step-button:hover {{
      transform: translateY(-1px);
    }}
    .story-step-button.is-active {{
      border-color: var(--accent);
      background: rgba(159,63,47,0.1);
    }}
    .story-step-button.is-dynamic {{
      border-style: dashed;
    }}
    .story-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.62);
      overflow: hidden;
    }}
    .story-card.is-active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(159,63,47,0.08);
    }}
    .story-card.is-dynamic {{
      border-style: dashed;
    }}
    .story-card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
    }}
    .story-card-body {{
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr);
      gap: 14px;
      padding: 16px;
    }}
    .story-image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .story-image-frame {{
      position: relative;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      aspect-ratio: 1 / 1;
      background: rgba(255,255,255,0.62);
    }}
    .story-image-frame img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .story-image-label {{
      position: absolute;
      top: 10px;
      left: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(17,24,39,0.74);
      color: #fff8f0;
      font-size: 11px;
      font-weight: 700;
    }}
    .story-badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .story-badge {{
      display: inline-flex;
      align-items: center;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }}
    .story-badge.dynamic {{
      color: #9f3f2f;
      border-color: rgba(159,63,47,0.26);
      background: rgba(159,63,47,0.08);
    }}
    .story-dialogue {{
      display: grid;
      gap: 12px;
    }}
    .story-dialogue-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 13px;
      background: rgba(255,255,255,0.62);
    }}
    .story-inspector-image {{
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.6);
      object-fit: cover;
      aspect-ratio: 1 / 1;
    }}
    .story-debug-card details {{
      margin-top: 10px;
    }}
    .story-debug-card pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 260px;
      overflow: auto;
      font-size: 11px;
      color: var(--muted);
    }}
    .nav-cluster-list,
    .nav-step-list {{
      display: grid;
      gap: 10px;
    }}
    .nav-cluster-card,
    .nav-step-item,
    .nav-intent-card,
    .nav-stage-card,
    .nav-validation-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.56);
      padding: 12px 13px;
    }}
    .nav-cluster-card.is-missed {{
      border-color: rgba(185,28,28,0.35);
      background: rgba(255, 239, 239, 0.82);
    }}
    .nav-cluster-card.is-planned {{
      border-color: rgba(21, 128, 61, 0.28);
    }}
    .nav-step-item {{
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }}
    .nav-step-item:hover {{
      transform: translateY(-1px);
    }}
    .nav-step-item.is-active {{
      border-color: var(--accent);
      background: rgba(159,63,47,0.1);
    }}
    .nav-step-head,
    .nav-cluster-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
    }}
    .nav-step-badge,
    .nav-cluster-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 30px;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }}
    .nav-status-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .nav-status-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      font-size: 11px;
      font-weight: 700;
    }}
    .nav-hero-note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }}
    .trajectory-panel {{
      padding: 0;
      overflow: hidden;
    }}
    .trajectory-stage-wrap {{
      padding: 18px;
    }}
    .trajectory-stage-shell {{
      position: relative;
      width: 100%;
      height: min(76vh, 920px);
      border-radius: 24px;
      overflow: hidden;
      border: 1px solid rgba(79, 53, 47, 0.2);
      background:
        linear-gradient(135deg, rgba(65,34,28,0.14), rgba(30,20,18,0.06)),
        #f5eadf;
    }}
    .trajectory-stage {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform-origin: center center;
      cursor: grab;
      user-select: none;
    }}
    .trajectory-stage.is-dragging {{
      cursor: grabbing;
    }}
    .trajectory-overview-image {{
      display: block;
      width: 100%;
      height: 100%;
      border-radius: 20px;
      object-fit: cover;
      pointer-events: none;
    }}
    .trajectory-svg {{
      position: absolute;
      inset: 0;
      overflow: visible;
      pointer-events: none;
    }}
    .trajectory-segment {{
      fill: none;
      stroke: rgba(159,63,47,0.72);
      stroke-width: 3.5;
      stroke-linecap: round;
      stroke-dasharray: 8 7;
      animation: trajectory-flow 1.6s linear infinite;
      pointer-events: none;
    }}
    .trajectory-segment.is-stop {{
      stroke: rgba(100,116,139,0.78);
    }}
    @keyframes trajectory-flow {{
      from {{ stroke-dashoffset: 0; }}
      to {{ stroke-dashoffset: -30; }}
    }}
    .trajectory-step-layer {{
      position: absolute;
      inset: 0;
      pointer-events: none;
    }}
    .trajectory-step-node {{
      position: absolute;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid;
      background: rgba(255,255,255,0.92);
      box-shadow: 0 10px 24px rgba(63,31,26,0.16);
      cursor: pointer;
      pointer-events: auto;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-weight: 800;
      color: var(--ink);
      transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    .trajectory-step-node:hover {{
      transform: translate(-50%, -50%) scale(1.05);
    }}
    .trajectory-step-node.is-active {{
      box-shadow: 0 0 0 3px rgba(255,250,244,0.92), 0 0 0 6px rgba(159,63,47,0.45);
      z-index: 4;
    }}
    .trajectory-step-node.is-stop {{
      border-style: dashed;
    }}
    .trajectory-fov-box {{
      position: absolute;
      border: 2px solid rgba(37,99,235,0.92);
      background: rgba(37,99,235,0.08);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.9) inset;
      border-radius: 8px;
      pointer-events: none;
      transition: left 220ms ease, top 220ms ease, width 220ms ease, height 220ms ease, opacity 160ms ease;
      opacity: 0;
      z-index: 2;
    }}
    .trajectory-fov-box.is-visible {{
      opacity: 1;
    }}
    .trajectory-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,255,255,0));
    }}
    .trajectory-toolbar .left,
    .trajectory-toolbar .right {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
    }}
    .timeline-strip {{
      display: grid;
      grid-template-columns: auto auto auto 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 14px 18px 18px;
      border-top: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0), rgba(255,255,255,0.4));
    }}
    .timeline-range {{
      width: 100%;
    }}
    .nav-dual-viewer {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .nav-image-frame {{
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.7), rgba(248,235,221,0.95));
      aspect-ratio: 1 / 1;
    }}
    .nav-image {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .nav-image-label {{
      position: absolute;
      top: 12px;
      left: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(17,24,39,0.74);
      color: #fff8f0;
      font-size: 11px;
      font-weight: 700;
    }}
    .nav-info-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .obs-branch-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .obs-branch-card,
    .obs-memory-card,
    .obs-validation-card,
    .obs-junior-card,
    .obs-chief-card,
    .obs-report-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.56);
      padding: 12px 13px;
    }}
    .obs-branch-card.supported {{
      border-color: rgba(21, 128, 61, 0.35);
      background: rgba(236, 253, 245, 0.9);
    }}
    .obs-branch-card.opposed {{
      border-color: rgba(185, 28, 28, 0.35);
      background: rgba(254, 242, 242, 0.9);
    }}
    .obs-branch-card.unresolved {{
      border-color: rgba(180, 83, 9, 0.35);
      background: rgba(255, 251, 235, 0.9);
    }}
    .obs-branch-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
    }}
    .obs-branch-state {{
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
    }}
    .obs-memory-list,
    .obs-report-list {{
      display: grid;
      gap: 10px;
    }}
    .obs-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.58);
      overflow: hidden;
    }}
    .obs-card.is-active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(159,63,47,0.08);
    }}
    .obs-card.early-stop {{
      border-color: rgba(245, 158, 11, 0.72);
      box-shadow: 0 0 0 3px rgba(245,158,11,0.15);
    }}
    .obs-card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,255,255,0));
      cursor: pointer;
    }}
    .obs-card-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .obs-badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.82);
      font-size: 11px;
      font-weight: 700;
    }}
    .obs-banner {{
      padding: 10px 14px;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: #fffaf2;
    }}
    .obs-banner.continue {{
      background: linear-gradient(135deg, rgba(180,83,9,0.92), rgba(234,179,8,0.72));
    }}
    .obs-banner.early-stop {{
      background: linear-gradient(135deg, rgba(22,163,74,0.92), rgba(245,158,11,0.78));
    }}
    .obs-card-body {{
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(340px, 1.05fr);
      gap: 14px;
      padding: 16px;
    }}
    .obs-image-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .obs-image-frame {{
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,0.7), rgba(248,235,221,0.95));
      aspect-ratio: 1 / 1;
    }}
    .obs-image {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}
    .obs-image-label {{
      position: absolute;
      top: 10px;
      left: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(17,24,39,0.74);
      color: #fff8f0;
      font-size: 11px;
      font-weight: 700;
    }}
    .obs-dialogue {{
      display: grid;
      gap: 12px;
    }}
    .obs-bubble {{
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.74);
      border: 1px solid var(--line);
    }}
    .obs-bubble.junior {{
      border-color: rgba(37, 99, 235, 0.2);
      background: rgba(239, 246, 255, 0.88);
    }}
    .obs-bubble.chief {{
      border-color: rgba(159, 63, 47, 0.24);
      background: rgba(255, 247, 237, 0.9);
    }}
    .obs-findings {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }}
    .obs-finding-chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.84);
    }}
    .obs-finding-chip.level1 {{
      background: rgba(219, 234, 254, 0.95);
    }}
    .obs-finding-chip.level2 {{
      background: rgba(254, 240, 138, 0.95);
    }}
    .obs-finding-chip.level3 {{
      background: rgba(254, 226, 226, 0.95);
    }}
    .obs-chief-thinking {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(31, 41, 55, 0.08);
    }}
    .obs-chief-thinking summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--muted);
    }}
    .obs-chief-thinking pre {{
      white-space: pre-wrap;
      margin: 10px 0 0;
      font: 12px/1.5 var(--sans);
      color: #374151;
    }}
    .obs-next-target,
    .obs-evidence-box {{
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.72);
    }}
    .obs-report-locked {{
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
      background: rgba(255,255,255,0.46);
    }}
    .obs-report-tree {{
      display: grid;
      gap: 8px;
    }}
    .obs-tree-node {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255,255,255,0.8);
    }}
    .obs-report-accordion details {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.82);
      padding: 10px 12px;
    }}
    .obs-report-accordion summary {{
      cursor: pointer;
      font-weight: 700;
    }}
    @media (max-width: 1180px) {{
      .dashboard-shell {{
        grid-template-columns: 1fr;
      }}
      .column-resizer,
      .panel-collapse-button {{
        display: none;
      }}
      .left-column,
      .center-column,
      .right-column {{
        margin: 0;
      }}
      .center-column {{
        order: -1;
      }}
    }}
    @media (max-width: 720px) {{
      body {{ padding: 10px; }}
      .panel-header, .panel-body, .hero, .canvas-toolbar, .canvas-stage-wrap, .canvas-footer {{ padding-left: 14px; padding-right: 14px; }}
      .meta-grid, .stats-grid, .detail-list, .reason-compare, .audit-classes, .toggle-grid, .hotkeys {{
        grid-template-columns: 1fr;
      }}
      .canvas-scroll {{ height: 62vh; }}
    }}
  </style>
</head>
<body>
  <div class="dashboard-page">
  <section class="view-switcher-panel">
    <div class="view-switcher-row">
      <div class="view-tabs">
        <button class="view-tab is-active" type="button" id="view-tab-case-story" data-view-tab="case_story">Case Story</button>
        <button class="view-tab" type="button" id="view-tab-screening" data-view-tab="screening">Global Screening</button>
        <button class="view-tab" type="button" id="view-tab-navigation" data-view-tab="navigation">Navigation Replay</button>
        <button class="view-tab" type="button" id="view-tab-observation" data-view-tab="observation">Raw Observation</button>
      </div>
      <div class="view-description" id="view-description">按真实 Observation 顺序串联 Trace、Navigation、Chief Review 与最终报告。</div>
    </div>
    <div class="data-source-banner">
      <strong>当前数据源：</strong> {data_source_label}
    </div>
    <div class="dashboard-error-banner" id="dashboard-error-banner"></div>
  </section>
  <div class="dashboard-shell" id="dashboard-shell">
    <aside class="panel dashboard-column left-column" id="left-column">
      <div class="panel-header">
        <div class="panel-header-actions">
          <div>
            <p class="panel-kicker">元数据与控制</p>
            <h2 class="panel-title">实验上下文</h2>
          </div>
          <button class="panel-collapse-button" type="button" id="toggle-left-header">收起左栏</button>
        </div>
      </div>
      <div class="panel-body stack">
        <div class="view-pane is-active" id="pane-case-story-left">
          <section class="stack">
            <div class="control-card">
              <p class="panel-kicker">Case 状态</p>
              <div class="story-summary-grid" id="story-summary-grid"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">Trace Clusters</p>
              <div class="story-cluster-list" id="story-cluster-list"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">Evidence Steps</p>
              <div class="story-step-list" id="story-step-list"></div>
            </div>
          </section>
        </div>
        <div class="view-pane" id="pane-screening-left">
        <section>
          <div class="meta-grid">{metadata_cards}</div>
        </section>
        <section class="stack">
          <div>
            <p class="panel-kicker">Trace 规则</p>
            <div class="legend-list">{rubric_cards}</div>
          </div>
        </section>
        <section class="stack">
          <div class="control-card">
            <p class="panel-kicker">视图模式</p>
            <div class="toggle-grid">
              <button class="toggle-button is-active" id="mode-default" data-view-mode="default_score_mode">
                <strong>default_score_mode</strong>
                <span>以主分数驱动热力强调</span>
              </button>
              <button class="toggle-button" id="mode-entropy" data-view-mode="entropy_disagreement_mode">
                <strong>entropy_disagreement_mode</strong>
                <span>优先排查灰区与冲突区</span>
              </button>
            </div>
          </div>
          <div class="control-card">
            <p class="panel-kicker">审计模式</p>
            <div class="toggle-grid">
              <button class="toggle-button is-active" id="audit-standard" data-audit-mode="standard">
                <strong>standard</strong>
                <span>适合手动确认与补充备注</span>
              </button>
              <button class="toggle-button" id="audit-turbo" data-audit-mode="turbo_audit">
                <strong>turbo_audit</strong>
                <span>自动保存并跳转到下一格</span>
              </button>
            </div>
          </div>
          <div class="control-card">
            <p class="panel-kicker">类别筛选</p>
            <div class="chip-row" id="label-filter-row">{label_filter_buttons}</div>
          </div>
          <div class="control-card">
            <p class="panel-kicker">分数筛选</p>
            <div class="chip-row" id="score-filter-row">
              <button class="chip is-active" data-score-filter="all">全部分数</button>
              <button class="chip" data-score-filter="lt_03">&lt; 0.3</button>
              <button class="chip" data-score-filter="mid_03_07">0.3 - 0.7</button>
              <button class="chip" data-score-filter="gt_07">&gt; 0.7</button>
            </div>
          </div>
          <div class="control-card">
            <p class="panel-kicker">附加筛选</p>
            <div class="chip-row" id="extra-filter-row">
              <button class="chip" data-extra-filter="require_high_mag">需要高倍</button>
              <button class="chip" data-extra-filter="risk_disagreement">高风险分歧</button>
              <button class="chip" data-extra-filter="priority_high">优先级 3+</button>
            </div>
          </div>
        </section>
        <section>
          <p class="panel-kicker">统计摘要</p>
          <div class="stats-grid" id="stats-grid">{stats_cards}</div>
        </section>
        <section class="stack">
          <div class="control-card">
            <p class="panel-kicker">校验状态</p>
            <div class="violation-banner">
              <strong>{validation_status}</strong>
              <span class="tiny">{validation_summary}</span>
            </div>
          </div>
          <div class="violation-list">{violation_items}</div>
        </section>
        </div>
        <div class="view-pane" id="pane-navigation-left">
          <section class="stack">
            <div class="control-card">
              <p class="panel-kicker">导航目标概览</p>
              <div class="nav-cluster-list" id="navigation-cluster-list"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">步骤树</p>
              <div class="nav-step-list" id="navigation-step-list"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">导航校验</p>
              <div class="nav-validation-card">
                <div class="violation-banner">
                  <strong id="navigation-validation-status">-</strong>
                  <span class="tiny" id="navigation-validation-summary">-</span>
                </div>
                <div class="violation-list" id="navigation-violation-list" style="margin-top:10px;"></div>
              </div>
            </div>
          </section>
        </div>
        <div class="view-pane" id="pane-observation-left">
          <section class="stack">
            <div class="control-card">
              <p class="panel-kicker">分支状态晴雨表</p>
              <div class="obs-branch-grid" id="observation-branch-grid"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">动态记忆追踪</p>
              <div class="obs-memory-list" id="observation-memory-list"></div>
            </div>
            <div class="control-card">
              <p class="panel-kicker">Observation Contract</p>
              <div class="obs-validation-card">
                <div class="violation-banner">
                  <strong id="observation-validation-status">-</strong>
                  <span class="tiny" id="observation-validation-summary">-</span>
                </div>
                <div class="violation-list" id="observation-violation-list" style="margin-top:10px;"></div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </aside>

    <div class="column-resizer" id="left-resizer" data-side="left" title="拖拽调整左栏宽度">
      <button class="resizer-toggle" type="button" id="toggle-left-rail" aria-label="收起或展开左栏">收</button>
    </div>

    <main class="center-column">
      <div class="view-pane is-active" id="pane-case-story-center">
        <section class="panel hero">
          <p class="panel-kicker">Case Story</p>
          <h1>新版 Agent 证据故事线</h1>
          <p>以真实观察顺序串联低倍筛查、导航计划、动态追加观察、Chief Review 与最终报告。</p>
          <div class="hero-strip">
            <div class="hero-pill"><strong>病例</strong> {case_id}</div>
            <div class="hero-pill"><strong>Observation</strong> <span id="story-observation-count">0</span></div>
            <div class="hero-pill"><strong>动态 Step</strong> <span id="story-dynamic-count">0</span></div>
            <div class="hero-pill"><strong>最终报告</strong> <span id="story-report-ready">-</span></div>
          </div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <div>
              <p class="panel-kicker">Evidence Timeline</p>
              <h2 class="panel-title">Trace → Navigate → Observe → Chief</h2>
            </div>
          </div>
          <div class="panel-body">
            <div class="story-timeline" id="story-timeline"></div>
          </div>
        </section>
      </div>
      <div class="view-pane" id="pane-screening-center">
      <section class="panel hero">
        <p class="panel-kicker">全局热力画布</p>
        <h1>Agent 实验可视化看板</h1>
        <p>这是历史 Trace/Navigate/Observe artifact 的兼容审计界面，不构成当前 AgentFlow 合同。</p>
        <div class="hero-strip">
          <div class="hero-pill"><strong>病例</strong> {case_id}</div>
          <div class="hero-pill"><strong>切片</strong> {slide_id}</div>
          <div class="hero-pill"><strong>已选 Patch</strong> {selected_patch_count}</div>
          <div class="hero-pill"><strong>渲染方式</strong> 静态 HTML 叠层</div>
        </div>
      </section>

      <section class="panel canvas-panel">
        <div class="canvas-toolbar">
          <div class="left">
            <button class="button subtle" id="zoom-out">缩小</button>
            <button class="button subtle" id="zoom-reset">重置</button>
            <button class="button subtle" id="zoom-in">放大</button>
          </div>
          <div class="right">
            <div class="hero-pill"><strong>Canvas 阈值</strong> {rendering_threshold}</div>
            <div class="hero-pill"><strong>当前底座</strong> {rendering_base}</div>
          </div>
        </div>
        <div class="canvas-stage-wrap">
          <div class="canvas-scroll" id="canvas-scroll">
            <div class="canvas-stage" id="canvas-stage" style="width:{canvas_width}px;height:{canvas_height}px;">
              <img src="{overview_image_url}" alt="全局预览图" class="overview-image" />
              <div class="overlay-layer">
                {patch_cells}
                {missing_cells}
              </div>
            </div>
          </div>
        </div>
        <div class="canvas-footer">
          <div class="footer-note">
            <strong>不确定性 / 分歧模式：</strong>开启后，灰区 Patch 与 <code>risk_disagreement</code> Patch 会被提升为最强视觉焦点，而高把握度 Patch 会被弱化。
          </div>
          <div class="footer-note" id="hover-readout">悬停或点击 Patch 以查看详情。</div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <p class="panel-kicker">契约异常兜底</p>
          <h2 class="panel-title">孤儿资产抽屉</h2>
        </div>
        <div class="panel-body">
          <details class="orphan-drawer" {orphan_open}>
            <summary>
              <span>非法或越界资产会被隔离到这里，避免污染主解剖空间画布。</span>
              <span class="orphan-count">{orphan_count}</span>
            </summary>
            <div class="orphan-list" style="margin-top:12px;">{orphan_cards}</div>
          </details>
        </div>
      </section>
      </div>

      <div class="view-pane" id="pane-navigation-center">
        <section class="panel hero">
          <p class="panel-kicker">电子地图导航</p>
          <h1>GPS Navigation View</h1>
          <p>这一视图回答的核心问题不是“这个 patch 属于什么类”，而是 Agent 是怎么走到这里、为什么在这里停下，以及为什么在这里切倍率。</p>
          <div class="hero-strip">
            <div class="hero-pill"><strong>病例</strong> {case_id}</div>
            <div class="hero-pill"><strong>总步骤</strong> <span id="navigation-total-steps">0</span></div>
            <div class="hero-pill"><strong>当前步骤</strong> <span id="navigation-current-step-pill">-</span></div>
            <div class="hero-pill"><strong>播放速度</strong> <span id="navigation-current-speed-pill">1x</span></div>
          </div>
          <div class="nav-hero-note">轨迹节点映射 step 序列，颜色映射倍率，节点大小映射 cluster_priority，FOV 框映射 `region_size_level0`。</div>
        </section>

        <section class="panel trajectory-panel">
          <div class="trajectory-toolbar">
            <div class="left">
              <button class="button subtle" id="nav-zoom-out">缩小</button>
              <button class="button subtle" id="nav-zoom-reset">重置</button>
              <button class="button subtle" id="nav-zoom-in">放大</button>
            </div>
            <div class="right">
              <div class="hero-pill"><strong>节点颜色</strong> 5x 蓝色 / 20x 红色</div>
              <div class="hero-pill"><strong>节点大小</strong> Cluster Priority</div>
            </div>
          </div>
          <div class="trajectory-stage-wrap">
            <div class="trajectory-stage-shell" id="navigation-canvas-scroll">
              <div class="trajectory-stage" id="navigation-canvas-stage" style="width:{canvas_width}px;height:{canvas_height}px;">
                <img src="{overview_image_url}" alt="导航底图" class="trajectory-overview-image" />
                <svg class="trajectory-svg" viewBox="0 0 100 100" preserveAspectRatio="none" id="trajectory-svg">
                  <defs>
                    <marker id="trajectory-arrow" markerWidth="1" markerHeight="1" refX="0.9" refY="0.5" orient="auto">
                      <path d="M0,0 L1,0.5 L0,1 z" fill="#9f3f2f"></path>
                    </marker>
                  </defs>
                  <g id="trajectory-line-group"></g>
                </svg>
                <div class="trajectory-step-layer" id="trajectory-step-layer"></div>
                <div class="trajectory-fov-box" id="trajectory-fov-box"></div>
              </div>
            </div>
          </div>
          <div class="timeline-strip">
            <button class="button" type="button" id="timeline-prev">上一步</button>
            <button class="button primary" type="button" id="timeline-play">播放</button>
            <button class="button" type="button" id="timeline-next">下一步</button>
            <input class="timeline-range" type="range" min="0" max="0" value="0" id="timeline-range" />
            <span class="tiny" id="timeline-status">Step 0 / 0</span>
            <select class="button" id="timeline-speed">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="2">2x</option>
            </select>
          </div>
        </section>
      </div>
      <div class="view-pane" id="pane-observation-center">
        <section class="panel hero">
          <p class="panel-kicker">多视图多模态推演时间轴</p>
          <h1>Observation / Reasoning View</h1>
          <p>这里展示的是 `Junior Screener` 的局部观察与 `Chief Pathologist` 的全局复核如何在每一步形成对话，并逐步推导到 `continue` 或 `early_stop`。</p>
          <div class="hero-strip">
            <div class="hero-pill"><strong>病例</strong> {case_id}</div>
            <div class="hero-pill"><strong>Observation 步数</strong> <span id="observation-total-steps">0</span></div>
            <div class="hero-pill"><strong>当前选中</strong> <span id="observation-current-step-pill">-</span></div>
            <div class="hero-pill"><strong>最终状态</strong> <span id="observation-final-state-pill">-</span></div>
          </div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <div class="panel-header-actions">
              <div>
                <p class="panel-kicker">双角色质询时间轴</p>
                <h2 class="panel-title">Junior-Chief Dialogue Timeline</h2>
              </div>
            </div>
          </div>
          <div class="panel-body">
            <div class="obs-report-list" id="observation-card-list"></div>
          </div>
        </section>
      </div>
    </main>

    <div class="column-resizer" id="right-resizer" data-side="right" title="拖拽调整右栏宽度">
      <button class="resizer-toggle" type="button" id="toggle-right-rail" aria-label="收起或展开右栏">收</button>
    </div>

    <aside class="panel dashboard-column right-column" id="right-column">
      <div class="panel-header">
        <div class="panel-header-actions">
          <div>
            <p class="panel-kicker">Patch 检查器</p>
            <h2 class="panel-title">单点复核</h2>
          </div>
          <button class="panel-collapse-button" type="button" id="toggle-right-header">收起右栏</button>
        </div>
      </div>
      <div class="layout-note">快捷键：<code>1-5</code> 重分类，<code>[</code>/<code>]</code> 上一个/下一个，<code>H</code> 切换高倍，<code>Enter</code> 保存，<code>Esc</code> 取消，<code>Ctrl/Cmd + Z</code> 撤销。</div>
      <div class="panel-body stack">
        <div class="view-pane is-active" id="pane-case-story-right">
          <section>
            <img src="" alt="Case Story 当前视图" id="story-inspector-image" class="story-inspector-image" />
          </section>
          <section class="story-report-card">
            <p class="panel-kicker">当前证据节点</p>
            <div class="detail-list">
              <div class="detail-card"><strong>Step</strong><span id="story-detail-step">-</span></div>
              <div class="detail-card"><strong>Cluster</strong><span id="story-detail-cluster">-</span></div>
              <div class="detail-card"><strong>Branch</strong><span id="story-detail-branch">-</span></div>
              <div class="detail-card"><strong>Decision</strong><span id="story-detail-decision">-</span></div>
            </div>
          </section>
          <section class="story-debug-card">
            <p class="panel-kicker">Chief Review / Debug</p>
            <div id="story-chief-panel"></div>
          </section>
          <section class="story-report-card">
            <p class="panel-kicker">Final Report</p>
            <div id="story-final-report"></div>
          </section>
        </div>
        <div class="view-pane" id="pane-screening-right">
        <section class="inspector-image-frame">
          <img src="" alt="高倍预览图" id="inspector-image" class="inspector-image" />
          <div class="inspector-image-badge" id="inspector-image-badge">尚未选中 Patch</div>
        </section>
        <section class="detail-list">
          <div class="detail-card"><strong>Patch</strong><span id="detail-patch-id">-</span></div>
          <div class="detail-card"><strong>位置</strong><span id="detail-location">-</span></div>
          <div class="detail-card"><strong>Cluster</strong><span id="detail-cluster-id">-</span></div>
          <div class="detail-card"><strong>分数来源</strong><span id="detail-score-origin">-</span></div>
          <div class="detail-card"><strong>分数</strong><span id="detail-score">-</span></div>
          <div class="detail-card"><strong>不确定性</strong><span id="detail-uncertainty">-</span></div>
          <div class="detail-card"><strong>语义类别</strong><span id="detail-semantic">-</span></div>
          <div class="detail-card"><strong>优先级</strong><span id="detail-priority">-</span></div>
          <div class="detail-card"><strong>需要高倍</strong><span id="detail-highmag">-</span></div>
          <div class="detail-card"><strong>一致性</strong><span id="detail-agreement">-</span></div>
        </section>
        <section class="reason-panel">
          <p class="panel-kicker">推理高亮</p>
          <div class="reason-compare">
            <div class="reason-model conch"><strong>CONCH</strong><div id="detail-conch">-</div></div>
            <div class="reason-model patho"><strong>PathoReasoner-R1</strong><div id="detail-patho">-</div></div>
          </div>
          <p class="reason-text" id="detail-fusion">请选择一个 Patch 以查看融合推理说明。</p>
        </section>
        <section class="audit-grid">
          <p class="panel-kicker">审计控制</p>
          <div class="audit-classes" id="audit-class-grid">{audit_class_buttons}</div>
          <div class="checkbox-row">
            <input type="checkbox" id="audit-highmag-toggle" />
            <label for="audit-highmag-toggle">需要高倍复核</label>
          </div>
          <textarea id="audit-comment" placeholder="可选审计备注。在 turbo 模式下，只有你想留下说明时才需要填写。"></textarea>
          <div class="button-row">
            <button class="button primary" id="save-audit-button">保存审计</button>
            <button class="button" id="cancel-audit-button">取消当前修改</button>
            <button class="button subtle" id="undo-audit-button">撤销上次审计</button>
          </div>
        </section>
        <section class="stack">
          <div>
            <p class="panel-kicker">最近审计记录</p>
            <div class="audit-list" id="audit-log-list">{audit_items}</div>
          </div>
          <div>
            <p class="panel-kicker">快捷键</p>
            <div class="hotkeys">
              <div class="hotkey"><strong>1</strong> 背景/伪影</div>
              <div class="hotkey"><strong>2</strong> 正常黏膜</div>
              <div class="hotkey"><strong>3</strong> 炎性样</div>
              <div class="hotkey"><strong>4</strong> 传统腺瘤样</div>
              <div class="hotkey"><strong>5</strong> SSL 可疑</div>
              <div class="hotkey"><strong>[ ]</strong> 上一个 / 下一个</div>
              <div class="hotkey"><strong>H</strong> 切换高倍</div>
              <div class="hotkey"><strong>Enter</strong> 保存</div>
              <div class="hotkey"><strong>Esc</strong> 取消</div>
              <div class="hotkey"><strong>Ctrl/Cmd+Z</strong> 撤销</div>
            </div>
          </div>
        </section>
        </div>
        <div class="view-pane" id="pane-navigation-right">
          <section class="nav-dual-viewer">
            <div class="nav-image-frame">
              <img src="" alt="导航主视图" id="nav-primary-image" class="nav-image" />
              <div class="nav-image-label" id="nav-primary-label">主视图</div>
            </div>
            <div class="nav-image-frame">
              <img src="" alt="导航辅助视图" id="nav-secondary-image" class="nav-image" />
              <div class="nav-image-label" id="nav-secondary-label">辅助视图</div>
            </div>
          </section>
          <section class="nav-info-grid">
            <div class="detail-card"><strong>Step</strong><span id="nav-step-id">-</span></div>
            <div class="detail-card"><strong>倍率</strong><span id="nav-step-mag">-</span></div>
            <div class="detail-card"><strong>动作</strong><span id="nav-step-action">-</span></div>
            <div class="detail-card"><strong>Cluster</strong><span id="nav-step-cluster">-</span></div>
            <div class="detail-card"><strong>优先级</strong><span id="nav-step-priority">-</span></div>
            <div class="detail-card"><strong>坐标</strong><span id="nav-step-coords">-</span></div>
          </section>
          <section class="nav-intent-card">
            <p class="panel-kicker">临床意图卡片</p>
            <p><strong>Review Goal</strong></p>
            <p class="reason-text" id="nav-review-goal">-</p>
            <p style="margin-top:10px;"><strong>Need To See</strong></p>
            <p class="reason-text" id="nav-need-to-see">-</p>
          </section>
          <section class="nav-stage-card">
            <p class="panel-kicker">Stage Gate Indicator</p>
            <div class="violation-banner" id="nav-stage-gate-banner">
              <strong id="nav-stage-gate">-</strong>
              <span class="tiny" id="nav-stage-note">-</span>
            </div>
          </section>
          <section class="nav-validation-card">
            <p class="panel-kicker">导航审计重点</p>
            <div class="tiny">这里审查的不是 patch 分类，而是 Agent 在当前 step 的临床意图、倍率选择和停靠逻辑是否合理。</div>
          </section>
        </div>
        <div class="view-pane" id="pane-observation-right">
          <section class="obs-report-card">
            <p class="panel-kicker">最终报告与终点审计</p>
            <div id="observation-report-root"></div>
          </section>
        </div>
      </div>
    </aside>
  </div>
  </div>
  <script id="dashboard-data" type="application/json">{dashboard_json}</script>
  <script>
    const DASHBOARD = JSON.parse(document.getElementById("dashboard-data").textContent);
    const LABELS = {labels_json};
    const LABEL_SHORT = {label_short_json};
    const AGREEMENT_DISPLAY = {agreement_display_json};
    const SCORE_ORIGIN_DISPLAY = {score_origin_display_json};
    const BRANCH_STATE_DISPLAY = {branch_state_display_json};
    const KEY_TO_LABEL = {{
      "1": "background_artifact_stroma",
      "2": "normal_mucosa",
      "3": "inflammatory_polyp_like",
      "4": "conventional_adenoma_like",
      "5": "ssl_suspicious_mucosa"
    }};
    const REASON_TOKENS = ["serrated", "conventional", "inflammatory", "normal", "background", "discordant", "disagreement", "lesion-positive"];

    const state = {{
      activeView: "case_story",
      activeCaseStoryStepId: (DASHBOARD.case_story && DASHBOARD.case_story.selected_step_id) || null,
      selectedPatchKey: DASHBOARD.initial_selected_patch_key || null,
      viewMode: "default_score_mode",
      auditMode: "standard",
      scoreFilter: "all",
      extraFilters: new Set(),
      labelFilters: new Set(LABELS),
      pendingAudit: null,
      auditLog: [...(DASHBOARD.audit_log || [])],
      patchMap: new Map((DASHBOARD.patches || []).map((patch) => [patch.patch_key, cloneValue(patch)])),
      history: [],
      zoom: 1,
      panX: 0,
      panY: 0,
      drag: {{ active: false, startX: 0, startY: 0, originX: 0, originY: 0 }},
      layout: {{
        leftWidth: 320,
        rightWidth: 400,
        leftCollapsed: false,
        rightCollapsed: false,
      }},
      resize: {{
        active: false,
        side: null,
        startX: 0,
        startLeftWidth: 320,
        startRightWidth: 400,
      }},
      navigation: {{
        selectedStepId: (DASHBOARD.navigation && DASHBOARD.navigation.initial_step_id) || null,
        hoverStepId: null,
        isPlaying: false,
        playTimer: null,
        speed: (DASHBOARD.playback_state && Number(DASHBOARD.playback_state.speed)) || 1,
        zoom: 1,
        panX: 0,
        panY: 0,
        drag: {{ active: false, startX: 0, startY: 0, originX: 0, originY: 0 }}
      }}
    }};

    const elements = {{
      viewTabCaseStory: document.getElementById("view-tab-case-story"),
      viewTabScreening: document.getElementById("view-tab-screening"),
      viewTabNavigation: document.getElementById("view-tab-navigation"),
      viewTabObservation: document.getElementById("view-tab-observation"),
      viewDescription: document.getElementById("view-description"),
      dashboardErrorBanner: document.getElementById("dashboard-error-banner"),
      paneCaseStoryLeft: document.getElementById("pane-case-story-left"),
      paneScreeningLeft: document.getElementById("pane-screening-left"),
      paneNavigationLeft: document.getElementById("pane-navigation-left"),
      paneObservationLeft: document.getElementById("pane-observation-left"),
      paneCaseStoryCenter: document.getElementById("pane-case-story-center"),
      paneScreeningCenter: document.getElementById("pane-screening-center"),
      paneNavigationCenter: document.getElementById("pane-navigation-center"),
      paneObservationCenter: document.getElementById("pane-observation-center"),
      paneCaseStoryRight: document.getElementById("pane-case-story-right"),
      paneScreeningRight: document.getElementById("pane-screening-right"),
      paneNavigationRight: document.getElementById("pane-navigation-right"),
      paneObservationRight: document.getElementById("pane-observation-right"),
      dashboardShell: document.getElementById("dashboard-shell"),
      leftColumn: document.getElementById("left-column"),
      rightColumn: document.getElementById("right-column"),
      leftResizer: document.getElementById("left-resizer"),
      rightResizer: document.getElementById("right-resizer"),
      toggleLeftHeader: document.getElementById("toggle-left-header"),
      toggleRightHeader: document.getElementById("toggle-right-header"),
      toggleLeftRail: document.getElementById("toggle-left-rail"),
      toggleRightRail: document.getElementById("toggle-right-rail"),
      hoverReadout: document.getElementById("hover-readout"),
      patchButtons: [...document.querySelectorAll(".patch-cell")],
      missingButtons: [...document.querySelectorAll(".missing-cell")],
      labelFilterRow: document.getElementById("label-filter-row"),
      scoreFilterRow: document.getElementById("score-filter-row"),
      extraFilterRow: document.getElementById("extra-filter-row"),
      modeDefault: document.getElementById("mode-default"),
      modeEntropy: document.getElementById("mode-entropy"),
      auditStandard: document.getElementById("audit-standard"),
      auditTurbo: document.getElementById("audit-turbo"),
      statsGrid: document.getElementById("stats-grid"),
      inspectorImage: document.getElementById("inspector-image"),
      inspectorImageBadge: document.getElementById("inspector-image-badge"),
      detailPatchId: document.getElementById("detail-patch-id"),
      detailLocation: document.getElementById("detail-location"),
      detailClusterId: document.getElementById("detail-cluster-id"),
      detailScoreOrigin: document.getElementById("detail-score-origin"),
      detailScore: document.getElementById("detail-score"),
      detailUncertainty: document.getElementById("detail-uncertainty"),
      detailSemantic: document.getElementById("detail-semantic"),
      detailPriority: document.getElementById("detail-priority"),
      detailHighMag: document.getElementById("detail-highmag"),
      detailAgreement: document.getElementById("detail-agreement"),
      detailConch: document.getElementById("detail-conch"),
      detailPatho: document.getElementById("detail-patho"),
      detailFusion: document.getElementById("detail-fusion"),
      auditClassGrid: document.getElementById("audit-class-grid"),
      auditHighMag: document.getElementById("audit-highmag-toggle"),
      auditComment: document.getElementById("audit-comment"),
      saveAuditButton: document.getElementById("save-audit-button"),
      cancelAuditButton: document.getElementById("cancel-audit-button"),
      undoAuditButton: document.getElementById("undo-audit-button"),
      auditLogList: document.getElementById("audit-log-list"),
      canvasStage: document.getElementById("canvas-stage"),
      canvasScroll: document.getElementById("canvas-scroll"),
      zoomIn: document.getElementById("zoom-in"),
      zoomOut: document.getElementById("zoom-out"),
      zoomReset: document.getElementById("zoom-reset"),
      navigationClusterList: document.getElementById("navigation-cluster-list"),
      navigationStepList: document.getElementById("navigation-step-list"),
      navigationValidationStatus: document.getElementById("navigation-validation-status"),
      navigationValidationSummary: document.getElementById("navigation-validation-summary"),
      navigationViolationList: document.getElementById("navigation-violation-list"),
      navigationTotalSteps: document.getElementById("navigation-total-steps"),
      navigationCurrentStepPill: document.getElementById("navigation-current-step-pill"),
      navigationCurrentSpeedPill: document.getElementById("navigation-current-speed-pill"),
      navCanvasScroll: document.getElementById("navigation-canvas-scroll"),
      navCanvasStage: document.getElementById("navigation-canvas-stage"),
      trajectoryLineGroup: document.getElementById("trajectory-line-group"),
      trajectoryStepLayer: document.getElementById("trajectory-step-layer"),
      trajectoryFovBox: document.getElementById("trajectory-fov-box"),
      navZoomIn: document.getElementById("nav-zoom-in"),
      navZoomOut: document.getElementById("nav-zoom-out"),
      navZoomReset: document.getElementById("nav-zoom-reset"),
      timelinePrev: document.getElementById("timeline-prev"),
      timelinePlay: document.getElementById("timeline-play"),
      timelineNext: document.getElementById("timeline-next"),
      timelineRange: document.getElementById("timeline-range"),
      timelineStatus: document.getElementById("timeline-status"),
      timelineSpeed: document.getElementById("timeline-speed"),
      storySummaryGrid: document.getElementById("story-summary-grid"),
      storyClusterList: document.getElementById("story-cluster-list"),
      storyStepList: document.getElementById("story-step-list"),
      storyTimeline: document.getElementById("story-timeline"),
      storyObservationCount: document.getElementById("story-observation-count"),
      storyDynamicCount: document.getElementById("story-dynamic-count"),
      storyReportReady: document.getElementById("story-report-ready"),
      storyInspectorImage: document.getElementById("story-inspector-image"),
      storyDetailStep: document.getElementById("story-detail-step"),
      storyDetailCluster: document.getElementById("story-detail-cluster"),
      storyDetailBranch: document.getElementById("story-detail-branch"),
      storyDetailDecision: document.getElementById("story-detail-decision"),
      storyChiefPanel: document.getElementById("story-chief-panel"),
      storyFinalReport: document.getElementById("story-final-report"),
      navPrimaryImage: document.getElementById("nav-primary-image"),
      navSecondaryImage: document.getElementById("nav-secondary-image"),
      navPrimaryLabel: document.getElementById("nav-primary-label"),
      navSecondaryLabel: document.getElementById("nav-secondary-label"),
      navStepId: document.getElementById("nav-step-id"),
      navStepMag: document.getElementById("nav-step-mag"),
      navStepAction: document.getElementById("nav-step-action"),
      navStepCluster: document.getElementById("nav-step-cluster"),
      navStepPriority: document.getElementById("nav-step-priority"),
      navStepCoords: document.getElementById("nav-step-coords"),
      navReviewGoal: document.getElementById("nav-review-goal"),
      navNeedToSee: document.getElementById("nav-need-to-see"),
      navStageGate: document.getElementById("nav-stage-gate"),
      navStageNote: document.getElementById("nav-stage-note"),
      navStageGateBanner: document.getElementById("nav-stage-gate-banner"),
      observationBranchGrid: document.getElementById("observation-branch-grid"),
      observationMemoryList: document.getElementById("observation-memory-list"),
      observationValidationStatus: document.getElementById("observation-validation-status"),
      observationValidationSummary: document.getElementById("observation-validation-summary"),
      observationViolationList: document.getElementById("observation-violation-list"),
      observationTotalSteps: document.getElementById("observation-total-steps"),
      observationCurrentStepPill: document.getElementById("observation-current-step-pill"),
      observationFinalStatePill: document.getElementById("observation-final-state-pill"),
      observationCardList: document.getElementById("observation-card-list"),
      observationReportRoot: document.getElementById("observation-report-root"),
    }};

    function round(value, digits = 2) {{
      const factor = Math.pow(10, digits);
      return Math.round((Number(value) + Number.EPSILON) * factor) / factor;
    }}

    function reportDashboardError(label, error) {{
      const message = label + ": " + ((error && error.message) ? error.message : String(error));
      if (elements.dashboardErrorBanner) {{
        elements.dashboardErrorBanner.classList.add("is-visible");
        elements.dashboardErrorBanner.textContent = "Dashboard 初始化异常\\n" + message;
      }}
      if (typeof console !== "undefined" && console.error) {{
        console.error("Dashboard error", label, error);
      }}
    }}

    function safeRun(label, fn) {{
      try {{
        return fn();
      }} catch (error) {{
        reportDashboardError(label, error);
        return null;
      }}
    }}

    function safeClosest(target, selector) {{
      if (!target || typeof target.closest !== "function") return null;
      return target.closest(selector);
    }}

    function cloneValue(value) {{
      return JSON.parse(JSON.stringify(value));
    }}

    function patchByKey(key) {{
      return state.patchMap.get(key) || null;
    }}

    function defaultPriorityFor(label, fallback) {{
      if (DASHBOARD.default_priorities && Object.prototype.hasOwnProperty.call(DASHBOARD.default_priorities, label)) {{
        return DASHBOARD.default_priorities[label];
      }}
      return fallback;
    }}

    function displaySemantic(label) {{
      if (!label) return "-";
      return (LABEL_SHORT[label] || label) + " / " + label;
    }}

    function displayAgreement(value) {{
      if (!value) return "-";
      return AGREEMENT_DISPLAY[value] || value;
    }}

    function displayScoreOrigin(value) {{
      if (!value) return SCORE_ORIGIN_DISPLAY.native_score || "原生分数";
      return SCORE_ORIGIN_DISPLAY[value] || value;
    }}

    function missingImageDataUrl(title, subtitle) {{
      const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">
          <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#fff7ed" />
              <stop offset="100%" stop-color="#fde7d8" />
            </linearGradient>
          </defs>
          <rect width="640" height="640" fill="url(#bg)" />
          <rect x="48" y="48" width="544" height="544" rx="28" fill="none" stroke="#d97706" stroke-dasharray="10 10" stroke-width="3" />
          <text x="320" y="280" text-anchor="middle" font-size="28" font-family="sans-serif" fill="#9a3412" font-weight="700">${{title}}</text>
          <text x="320" y="325" text-anchor="middle" font-size="18" font-family="sans-serif" fill="#7c2d12">${{subtitle}}</text>
        </svg>
      `;
      return "data:image/svg+xml;utf8," + svg.replace(/#/g, "%23").replace(/\\n/g, "");
    }}

    function navigationSteps() {{
      return (DASHBOARD.navigation && DASHBOARD.navigation.steps) || [];
    }}

    function navigationClusters() {{
      return (DASHBOARD.navigation && DASHBOARD.navigation.clusters) || [];
    }}

    function navigationValidation() {{
      return (DASHBOARD.navigation && DASHBOARD.navigation.contract_validation) || {{ status: "ok", violations: [] }};
    }}

    function stepById(stepId) {{
      return navigationSteps().find((step) => step.step_id === stepId) || null;
    }}

    function currentNavigationStep() {{
      return stepById(state.navigation.selectedStepId);
    }}

    function currentNavigationStepIndex() {{
      const steps = navigationSteps();
      return Math.max(0, steps.findIndex((step) => step.step_id === state.navigation.selectedStepId));
    }}

    function stopNavigationPlayback() {{
      if (state.navigation.playTimer) {{
        window.clearInterval(state.navigation.playTimer);
        state.navigation.playTimer = null;
      }}
      state.navigation.isPlaying = false;
      if (elements.timelinePlay) elements.timelinePlay.textContent = "播放";
    }}

    function caseStory() {{
      return DASHBOARD.case_story || {{ summary: {{}}, steps: [], planned_steps: [], clusters: [], report: {{}} }};
    }}

    function caseStorySteps() {{
      return caseStory().steps || [];
    }}

    function caseStoryStepById(stepId) {{
      return caseStorySteps().find((step) => step.step_id === stepId) || null;
    }}

    function currentCaseStoryStep() {{
      return caseStoryStepById(state.activeCaseStoryStepId) || caseStorySteps()[0] || null;
    }}

    function imageForStoryStep(step) {{
      if (!step) return missingImageDataUrl("未选中步骤", "请选择一个 Evidence Step");
      return step.detail_image || step.local_image || step.overview_image || missingImageDataUrl("图像缺失", step.step_id + " 未找到 observe crop");
    }}

    function setSelectedCaseStoryStep(stepId) {{
      const step = caseStoryStepById(stepId);
      if (!step) return;
      state.activeCaseStoryStepId = step.step_id;
      renderCaseStoryStepList();
      renderCaseStoryTimeline();
      renderCaseStoryInspector();
      if (step.step_id && stepById(step.step_id)) {{
        state.navigation.selectedStepId = step.step_id;
        renderNavigationStepList();
        renderTrajectory();
        renderNavigationInspector();
      }}
    }}

    function renderCaseStorySummary() {{
      const summary = caseStory().summary || {{}};
      if (elements.storyObservationCount) elements.storyObservationCount.textContent = String(summary.observation_count || 0);
      if (elements.storyDynamicCount) elements.storyDynamicCount.textContent = String(summary.dynamic_step_count || 0);
      if (elements.storyReportReady) elements.storyReportReady.textContent = summary.report_ready ? "已生成" : "未生成";
      elements.storySummaryGrid.innerHTML = `
        <div class="story-stat"><strong>${{summary.observation_count || 0}}</strong><span class="tiny">Observation 步数</span></div>
        <div class="story-stat"><strong>${{summary.navigation_count || 0}}</strong><span class="tiny">计划导航步数</span></div>
        <div class="story-stat"><strong>${{summary.dynamic_step_count || 0}}</strong><span class="tiny">动态追加 Step</span></div>
        <div class="story-stat"><strong>${{summary.chief_model_review_count || 0}}</strong><span class="tiny">Chief 模型审查</span></div>
        <div class="story-stat"><strong>${{summary.chief_parse_error_count || 0}}</strong><span class="tiny">Chief Parse Error</span></div>
        <div class="story-stat"><strong>${{summary.final_status || "-"}}</strong><span class="tiny">最终状态</span></div>
      `;
    }}

    function renderCaseStoryClusters() {{
      const clusters = caseStory().clusters || [];
      if (!clusters.length) {{
        elements.storyClusterList.innerHTML = '<div class="empty-state">没有 trace clusters。</div>';
        return;
      }}
      elements.storyClusterList.innerHTML = clusters.map((cluster) => {{
        const classes = ["story-cluster-card"];
        if (cluster.observed) classes.push("is-observed");
        return `<div class="${{classes.join(" ")}}">
          <div class="nav-cluster-head"><strong>${{cluster.cluster_id || "-"}}</strong><span class="nav-cluster-badge">P${{cluster.cluster_priority || 0}}</span></div>
          <div class="tiny">${{LABEL_SHORT[cluster.cluster_label] || cluster.cluster_label || "-"}} · ${{cluster.workflow_branch || "-"}}</div>
          <div class="story-badge-row"><span class="story-badge">${{cluster.planned ? "已规划" : "未规划"}}</span><span class="story-badge">${{cluster.observed ? "已观察" : "未观察"}}</span></div>
        </div>`;
      }}).join("");
    }}

    function renderCaseStoryStepList() {{
      const steps = caseStorySteps();
      if (!steps.length) {{
        elements.storyStepList.innerHTML = '<div class="empty-state">没有 observation evidence steps。</div>';
        return;
      }}
      elements.storyStepList.innerHTML = steps.map((step, index) => {{
        const active = step.step_id === state.activeCaseStoryStepId ? " is-active" : "";
        const dynamic = step.is_dynamic ? " is-dynamic" : "";
        const label = LABEL_SHORT[step.cluster_label] || step.cluster_label || step.workflow_branch || "-";
        return `<button type="button" class="story-step-button${{active}}${{dynamic}}" data-story-step-id="${{step.step_id}}">
          <div class="nav-step-head"><strong>[${{index + 1}}] ${{step.step_id}}</strong><span class="nav-step-badge">${{step.magnification || "-"}}x</span></div>
          <div class="tiny">${{step.cluster_id || "-"}} · ${{label}}</div>
          <div class="story-badge-row">${{step.is_dynamic ? '<span class="story-badge dynamic">动态追加</span>' : '<span class="story-badge">计划观察</span>'}}<span class="story-badge">${{step.stage_decision || "-"}}</span></div>
        </button>`;
      }}).join("");
      elements.storyStepList.querySelectorAll("[data-story-step-id]").forEach((node) => {{
        node.addEventListener("click", () => setSelectedCaseStoryStep(node.getAttribute("data-story-step-id")));
      }});
    }}

    function renderCaseStoryTimeline() {{
      const steps = caseStorySteps();
      if (!steps.length) {{
        elements.storyTimeline.innerHTML = '<div class="empty-state">没有 evidence timeline。</div>';
        return;
      }}
      elements.storyTimeline.innerHTML = steps.map((step, index) => {{
        const active = step.step_id === state.activeCaseStoryStepId ? " is-active" : "";
        const dynamic = step.is_dynamic ? " is-dynamic" : "";
        const review = step.chief_review || {{}};
        const decision = review.decision || "no_chief_review";
        const image = imageForStoryStep(step);
        const badges = [
          step.is_dynamic ? '<span class="story-badge dynamic">动态追加</span>' : '<span class="story-badge">计划观察</span>',
          `<span class="story-badge">${{step.workflow_branch || "-"}}</span>`,
          `<span class="story-badge">${{decision}}</span>`,
        ].join("");
        return `<article class="story-card${{active}}${{dynamic}}" data-story-card-id="${{step.step_id}}">
          <div class="story-card-header">
            <div><strong>${{index + 1}}. ${{step.step_id}}</strong><div class="tiny">${{step.cluster_id || "-"}} · ${{step.review_goal || "-"}}</div></div>
            <div class="story-badge-row">${{badges}}</div>
          </div>
          <div class="story-card-body">
            <div class="story-image-grid">
              <div class="story-image-frame"><img src="${{step.overview_image || image}}" alt="overview" /><div class="story-image-label">overview</div></div>
              <div class="story-image-frame"><img src="${{step.detail_image || step.local_image || image}}" alt="detail" /><div class="story-image-label">detail / local</div></div>
            </div>
            <div class="story-dialogue">
              <div class="story-dialogue-card"><strong>Junior Screener</strong><p class="reason-text">${{step.observation || "-"}}</p><p class="reason-text">${{step.reasoning || "-"}}</p></div>
              <div class="story-dialogue-card"><strong>Chief Pathologist</strong><p class="reason-text">Decision: ${{decision}}</p><p class="reason-text">${{review.continue_reason || review.branch_correction_reason || "-"}}</p></div>
            </div>
          </div>
        </article>`;
      }}).join("");
      elements.storyTimeline.querySelectorAll("[data-story-card-id]").forEach((node) => {{
        node.addEventListener("click", () => setSelectedCaseStoryStep(node.getAttribute("data-story-card-id")));
      }});
    }}

    function renderCaseStoryInspector() {{
      const step = currentCaseStoryStep();
      const report = (caseStory().report || {{}});
      if (!step) {{
        elements.storyInspectorImage.src = missingImageDataUrl("无 Evidence Step", "当前 case 没有 observation 记录");
        elements.storyDetailStep.textContent = "-";
        elements.storyDetailCluster.textContent = "-";
        elements.storyDetailBranch.textContent = "-";
        elements.storyDetailDecision.textContent = "-";
        elements.storyChiefPanel.innerHTML = '<div class="empty-state">没有 Chief Review。</div>';
      }} else {{
        const review = step.chief_review || {{}};
        elements.storyInspectorImage.src = imageForStoryStep(step);
        elements.storyDetailStep.textContent = step.step_id || "-";
        elements.storyDetailCluster.textContent = step.cluster_id || "-";
        elements.storyDetailBranch.textContent = step.workflow_branch || "-";
        elements.storyDetailDecision.textContent = review.decision || "-";
        const parseError = review.parse_error ? `<details open><summary>Parse Error</summary><pre>${{review.parse_error}}</pre></details>` : "";
        const thought = review.thought_text ? `<details><summary>Chief Thinking Process</summary><pre>${{review.thought_text}}</pre></details>` : "";
        const answer = review.answer_candidate_text ? `<details><summary>Answer Candidate</summary><pre>${{review.answer_candidate_text}}</pre></details>` : "";
        const nextTarget = review.next_visual_target ? `<div class="story-dialogue-card"><strong>Next Visual Target</strong><p class="reason-text">${{JSON.stringify(review.next_visual_target)}}</p></div>` : "";
        elements.storyChiefPanel.innerHTML = `
          <div class="story-dialogue-card"><strong>${{review.review_source || "chief_review"}}</strong><p class="reason-text">${{review.model_name || "-"}}</p><p class="reason-text">round_trip_ms: ${{review.round_trip_ms || 0}}</p></div>
          <div class="story-dialogue-card"><strong>Continue / Correction</strong><p class="reason-text">${{review.continue_reason || review.branch_correction_reason || "-"}}</p></div>
          ${{nextTarget}}
          ${{parseError}}
          ${{thought}}
          ${{answer}}
        `;
      }}
      const prediction = report.hierarchical_prediction || {{}};
      const predictionHtml = Object.keys(prediction).length
        ? Object.entries(prediction).map(([key, value]) => `<div class="detail-card"><strong>${{key}}</strong><span>${{JSON.stringify(value)}}</span></div>`).join("")
        : '<div class="empty-state">没有 hierarchical_prediction。</div>';
      const truth = report.ground_truth_label || {{}};
      const comparison = report.final_report_comparison || {{}};
      const comparisonItems = comparison.items || [];
      const itemHtml = comparisonItems.length
        ? comparisonItems.map((item) => {{
            const mark = item.match === true ? "匹配" : item.match === false ? "不匹配" : "报告未明确";
            return `<div class="label-compare-row">
              <strong>${{item.label || item.key || "-"}}</strong>
              <div class="label-compare-values">
                <span>真实标签：${{item.truth || "-"}}</span>
                <span>Final Report：${{item.prediction || "-"}}</span>
                <span>状态：${{mark}}</span>
              </div>
            </div>`;
          }}).join("")
        : `<div class="empty-state">${{comparison.truth_summary || "未找到真实标签。"}}</div>`;
      const labelCompareHtml = `
        <div class="label-compare-card">
          <div class="label-compare-head">
            <div>
              <strong>真实标签 vs Final Report</strong>
              <div class="tiny">${{truth.source_path || "data/label/Adenoma_filtered.xlsx"}}</div>
            </div>
            <span class="label-status ${{comparison.status || "missing_label"}}">${{comparison.status_display || "缺少真实标签"}}</span>
          </div>
          <div class="label-compare-grid">
            <div class="label-compare-row">
              <strong>Label</strong>
              <div class="label-compare-values">
                <span>type：${{truth.type || "-"}}</span>
                <span>grade：${{truth.grade || "-"}}</span>
              </div>
            </div>
            ${{itemHtml}}
          </div>
        </div>
      `;
      elements.storyFinalReport.innerHTML = `
        <div class="detail-list">${{predictionHtml}}</div>
        <p class="reason-text" style="margin-top:12px;">${{report.summary || "暂无 integrated report。"}}</p>
        ${{labelCompareHtml}}
      `;
    }}

    function renderCaseStory() {{
      if (!state.activeCaseStoryStepId && caseStorySteps().length) {{
        state.activeCaseStoryStepId = caseStorySteps()[0].step_id;
      }}
      renderCaseStorySummary();
      renderCaseStoryClusters();
      renderCaseStoryStepList();
      renderCaseStoryTimeline();
      renderCaseStoryInspector();
    }}

    function setActiveView(view) {{
      state.activeView = view === "screening" || view === "navigation" || view === "observation" ? view : "case_story";
      const isCaseStory = state.activeView === "case_story";
      const isScreening = state.activeView === "screening";
      const isNavigation = state.activeView === "navigation";
      const isObservation = state.activeView === "observation";
      elements.viewTabCaseStory.classList.toggle("is-active", isCaseStory);
      elements.viewTabScreening.classList.toggle("is-active", isScreening);
      elements.viewTabNavigation.classList.toggle("is-active", isNavigation);
      elements.viewTabObservation.classList.toggle("is-active", isObservation);
      elements.paneCaseStoryLeft.classList.toggle("is-active", isCaseStory);
      elements.paneScreeningLeft.classList.toggle("is-active", isScreening);
      elements.paneNavigationLeft.classList.toggle("is-active", isNavigation);
      elements.paneObservationLeft.classList.toggle("is-active", isObservation);
      elements.paneCaseStoryCenter.classList.toggle("is-active", isCaseStory);
      elements.paneScreeningCenter.classList.toggle("is-active", isScreening);
      elements.paneNavigationCenter.classList.toggle("is-active", isNavigation);
      elements.paneObservationCenter.classList.toggle("is-active", isObservation);
      elements.paneCaseStoryRight.classList.toggle("is-active", isCaseStory);
      elements.paneScreeningRight.classList.toggle("is-active", isScreening);
      elements.paneNavigationRight.classList.toggle("is-active", isNavigation);
      elements.paneObservationRight.classList.toggle("is-active", isObservation);
      elements.viewDescription.textContent = isCaseStory
        ? "按真实 Observation 顺序串联 Trace、Navigation、Chief Review 与最终报告。"
        : isScreening
        ? "Patch 级全局筛查、分数筛选与人工审计。"
        : isNavigation
          ? "Step 级路径回放、FOV 范围与临床意图审计。"
          : "Junior 与 Chief 的双角色临床推演、分支状态与最终报告审计。";
      if (isCaseStory) {{
        safeRun("renderCaseStory", renderCaseStory);
      }}
      if (isNavigation) {{
        safeRun("renderNavigation", renderNavigation);
      }}
      if (isObservation) {{
        safeRun("renderObservation", renderObservation);
      }}
    }}

    function applyNavigationTransform() {{
      if (!elements.navCanvasStage) return;
      elements.navCanvasStage.style.transform = 'translate(calc(-50% + ' + state.navigation.panX + 'px), calc(-50% + ' + state.navigation.panY + 'px)) scale(' + state.navigation.zoom + ')';
    }}

    function focusNavigationStep(step) {{
      if (!step || !elements.navCanvasScroll) return;
      state.navigation.zoom = step.m === "20x" || step.m === "10x" ? 1.8 : 1.35;
      const stageWidth = (DASHBOARD.navigation && DASHBOARD.navigation.canvas_width) || elements.navCanvasStage.offsetWidth || 1;
      const stageHeight = (DASHBOARD.navigation && DASHBOARD.navigation.canvas_height) || elements.navCanvasStage.offsetHeight || 1;
      const targetX = (Number(step.center_x_percent) / 100) * stageWidth;
      const targetY = (Number(step.center_y_percent) / 100) * stageHeight;
      state.navigation.panX = (stageWidth / 2 - targetX) * state.navigation.zoom;
      state.navigation.panY = (stageHeight / 2 - targetY) * state.navigation.zoom;
      applyNavigationTransform();
    }}

    function setSelectedNavigationStep(stepId, options = {{ focus: true }}) {{
      const step = stepById(stepId);
      if (!step) return;
      state.navigation.selectedStepId = step.step_id;
      renderNavigationStepList();
      renderTrajectory();
      renderNavigationInspector();
      updateTimelineStatus();
      if (options.focus !== false) {{
        focusNavigationStep(step);
      }}
    }}

    function updateTimelineStatus() {{
      const steps = navigationSteps();
      const index = currentNavigationStepIndex();
      if (elements.timelineRange) {{
        elements.timelineRange.max = String(Math.max(0, steps.length - 1));
        elements.timelineRange.value = String(index);
      }}
      if (elements.timelineStatus) {{
        elements.timelineStatus.textContent = "Step " + (steps.length ? index + 1 : 0) + " / " + steps.length;
      }}
      if (elements.navigationTotalSteps) {{
        elements.navigationTotalSteps.textContent = String(steps.length);
      }}
      if (elements.navigationCurrentStepPill) {{
        const current = currentNavigationStep();
        elements.navigationCurrentStepPill.textContent = current ? current.step_id : "-";
      }}
      if (elements.navigationCurrentSpeedPill) {{
        elements.navigationCurrentSpeedPill.textContent = String(state.navigation.speed) + "x";
      }}
    }}

    function selectNavigationAdjacentStep(delta) {{
      const steps = navigationSteps();
      if (!steps.length) return;
      const currentIndex = currentNavigationStepIndex();
      const nextIndex = Math.max(0, Math.min(steps.length - 1, currentIndex + delta));
      setSelectedNavigationStep(steps[nextIndex].step_id);
    }}

    function playNavigation() {{
      const steps = navigationSteps();
      if (!steps.length) return;
      stopNavigationPlayback();
      state.navigation.isPlaying = true;
      elements.timelinePlay.textContent = "暂停";
      const tickMs = Math.max(260, Math.round(900 / state.navigation.speed));
      state.navigation.playTimer = window.setInterval(() => {{
        const currentIndex = currentNavigationStepIndex();
        if (currentIndex >= steps.length - 1) {{
          stopNavigationPlayback();
          return;
        }}
        setSelectedNavigationStep(steps[currentIndex + 1].step_id, {{ focus: true }});
      }}, tickMs);
    }}

    function toggleNavigationPlayback() {{
      if (state.navigation.isPlaying) {{
        stopNavigationPlayback();
      }} else {{
        playNavigation();
      }}
    }}

    function renderNavigationClusterList() {{
      const rows = navigationClusters();
      if (!rows.length) {{
        elements.navigationClusterList.innerHTML = '<div class="empty-state">没有可显示的 Trace Clusters。</div>';
        return;
      }}
      elements.navigationClusterList.innerHTML = rows.map((cluster) => {{
        const classes = ["nav-cluster-card"];
        if (cluster.missed_high_priority) classes.push("is-missed");
        if (cluster.planned_in_path) classes.push("is-planned");
        const statuses = [];
        statuses.push(cluster.planned_in_path ? "已规划进路径" : "未进入路径");
        statuses.push(cluster.observed ? "已关联观察" : "未关联观察");
        if (cluster.missed_high_priority) statuses.push("高优先级疑似漏掉");
        return `<div class="${{classes.join(" ")}}"><div class="nav-cluster-head"><strong>${{cluster.cluster_id || "-"}}</strong><span class="nav-cluster-badge">P${{cluster.cluster_priority}}</span></div><div class="tiny">${{cluster.cluster_label || "-"}} · ${{cluster.workflow_branch || "-"}}</div><div class="nav-status-row">${{statuses.map((text) => `<span class="nav-status-chip">${{text}}</span>`).join("")}}</div></div>`;
      }}).join("");
    }}

    function renderNavigationStepList() {{
      const rows = navigationSteps();
      if (!rows.length) {{
        elements.navigationStepList.innerHTML = '<div class="empty-state">没有可回放的 navigation steps。</div>';
        return;
      }}
      elements.navigationStepList.innerHTML = rows.map((step, index) => {{
        const active = step.step_id === state.navigation.selectedStepId ? " is-active" : "";
        const label = step.cluster_label ? (LABEL_SHORT[step.cluster_label] || step.cluster_label) : "Stop";
        const observed = step.is_stop ? "stop" : (step.is_observed ? "已观察" : "计划未观察");
        return `<button type="button" class="nav-step-item${{active}}" data-nav-step-id="${{step.step_id}}"><div class="nav-step-head"><strong>[Step ${{index + 1}}] ${{step.m}} | ${{step.action_display}}</strong><span class="nav-step-badge">P${{step.cluster_priority || 0}}</span></div><div class="tiny">${{step.cluster_id ? step.cluster_id + " (" + label + ")" : "终点步骤"}} · ${{observed}}</div></button>`;
      }}).join("");
      elements.navigationStepList.querySelectorAll("[data-nav-step-id]").forEach((node) => {{
        node.addEventListener("click", () => {{
          setSelectedNavigationStep(node.getAttribute("data-nav-step-id"));
        }});
      }});
    }}

    function renderNavigationValidation() {{
      const validation = navigationValidation();
      const violations = validation.violations || [];
      elements.navigationValidationStatus.textContent = validation.status === "ok" ? "正常" : "警告";
      elements.navigationValidationSummary.textContent = violations.length ? ("共 " + violations.length + " 项异常") : "没有检测到 navigation contract 异常";
      if (!violations.length) {{
        elements.navigationViolationList.innerHTML = '<div class="empty-state">当前路径未发现 contract 异常。</div>';
        return;
      }}
      elements.navigationViolationList.innerHTML = violations.map((item) => `<div class="violation-item"><strong>${{item.step_id || item.type}}</strong><div class="tiny">${{item.message || ""}}</div></div>`).join("");
    }}

    function renderTrajectory() {{
      const steps = navigationSteps();
      const lines = (DASHBOARD.navigation && DASHBOARD.navigation.lines) || [];
      elements.trajectoryLineGroup.innerHTML = lines.map((line, index) => {{
        const step = steps[index + 1] || {{}};
        const stopClass = step.is_stop ? " is-stop" : "";
        return `<line class="trajectory-segment${{stopClass}}" x1="${{line.x1}}" y1="${{line.y1}}" x2="${{line.x2}}" y2="${{line.y2}}" marker-end="url(#trajectory-arrow)"></line>`;
      }}).join("");
      elements.trajectoryStepLayer.innerHTML = steps.map((step, index) => {{
        const active = step.step_id === state.navigation.selectedStepId ? " is-active" : "";
        const stopClass = step.is_stop ? " is-stop" : "";
        const size = Math.max(16, Number(step.node_size || 16));
        return `<button type="button" class="trajectory-step-node${{active}}${{stopClass}}" data-nav-node-id="${{step.step_id}}" style="left:${{step.center_x_percent}}%;top:${{step.center_y_percent}}%;width:${{size}}px;height:${{size}}px;border-color:${{step.stroke}};background:${{step.fill}};">${{index + 1}}</button>`;
      }}).join("");
      elements.trajectoryStepLayer.querySelectorAll("[data-nav-node-id]").forEach((node) => {{
        node.addEventListener("mouseenter", () => {{
          state.navigation.hoverStepId = node.getAttribute("data-nav-node-id");
          updateNavigationFov();
        }});
        node.addEventListener("mouseleave", () => {{
          state.navigation.hoverStepId = null;
          updateNavigationFov();
        }});
        node.addEventListener("click", () => {{
          setSelectedNavigationStep(node.getAttribute("data-nav-node-id"));
        }});
      }});
      updateNavigationFov();
    }}

    function updateNavigationFov() {{
      const step = stepById(state.navigation.hoverStepId) || currentNavigationStep();
      if (!step) {{
        elements.trajectoryFovBox.classList.remove("is-visible");
        return;
      }}
      elements.trajectoryFovBox.classList.add("is-visible");
      elements.trajectoryFovBox.style.left = step.fov_left_percent + "%";
      elements.trajectoryFovBox.style.top = step.fov_top_percent + "%";
      elements.trajectoryFovBox.style.width = step.fov_width_percent + "%";
      elements.trajectoryFovBox.style.height = step.fov_height_percent + "%";
      elements.hoverReadout.textContent = `${{step.step_id}} · ${{step.m}} · ${{step.action_display}} · ${{step.cluster_id || "stop"}}`;
    }}

    function navigationImageSelection(step) {{
      if (!step) {{
        return {{
          primarySrc: missingImageDataUrl("未选中步骤", "请选择左侧 Step 或时间轴节点"),
          primaryLabel: "主视图",
          secondarySrc: missingImageDataUrl("未选中步骤", "请选择左侧 Step 或时间轴节点"),
          secondaryLabel: "辅助视图",
          missingPrimary: false,
          missingSecondary: false,
        }};
      }}
      if (step.m === "20x" || step.m === "10x") {{
        const primarySrc = step.detail_image || step.local_image || step.overview_image || "";
        const secondarySrc = step.local_image || step.overview_image || step.detail_image || "";
        const magLabel = step.m || "detail";
        return {{
          primarySrc: primarySrc || missingImageDataUrl("主视图缺失", step.step_id + " 计划存在但未产出 observe crop"),
          primaryLabel: magLabel + " / detail",
          secondarySrc: secondarySrc || missingImageDataUrl("辅助视图缺失", step.step_id + " 未产出 5x/overview 图像"),
          secondaryLabel: "5x / local",
          missingPrimary: !primarySrc,
          missingSecondary: !secondarySrc,
        }};
      }}
      const primarySrc = step.local_image || step.overview_image || step.detail_image || "";
      const secondarySrc = step.overview_image || step.detail_image || step.local_image || "";
      return {{
        primarySrc: primarySrc || missingImageDataUrl("主视图缺失", step.step_id + " 计划存在但未产出 observe crop"),
        primaryLabel: "5x / local",
        secondarySrc: secondarySrc || missingImageDataUrl("辅助视图缺失", step.step_id + " 未产出 overview/detail 图像"),
        secondaryLabel: "overview",
        missingPrimary: !primarySrc,
        missingSecondary: !secondarySrc,
      }};
    }}

    function renderNavigationInspector() {{
      const step = currentNavigationStep();
      const images = navigationImageSelection(step);
      elements.navPrimaryImage.src = images.primarySrc || "";
      elements.navSecondaryImage.src = images.secondarySrc || "";
      elements.navPrimaryLabel.textContent = images.primaryLabel + (images.missingPrimary ? " · 缺失" : "");
      elements.navSecondaryLabel.textContent = images.secondaryLabel + (images.missingSecondary ? " · 缺失" : "");
      if (!step) {{
        elements.navStepId.textContent = "-";
        elements.navStepMag.textContent = "-";
        elements.navStepAction.textContent = "-";
        elements.navStepCluster.textContent = "-";
        elements.navStepPriority.textContent = "-";
        elements.navStepCoords.textContent = "-";
        elements.navReviewGoal.textContent = "-";
        elements.navNeedToSee.textContent = "-";
        elements.navStageGate.textContent = "-";
        elements.navStageNote.textContent = "-";
        return;
      }}
      elements.navStepId.textContent = step.step_id;
      elements.navStepMag.textContent = step.m;
      elements.navStepAction.textContent = step.action_display;
      elements.navStepCluster.textContent = step.cluster_id ? (step.cluster_id + " / " + (LABEL_SHORT[step.cluster_label] || step.cluster_label || "-")) : "终点";
      elements.navStepPriority.textContent = String(step.cluster_priority || 0);
      elements.navStepCoords.textContent = Math.round(step.x) + ", " + Math.round(step.y);
      elements.navReviewGoal.textContent = step.review_goal || "-";
      elements.navNeedToSee.textContent = step.need_to_see || "-";
      elements.navStageGate.textContent = step.stage_gate || "-";
      if (images.missingPrimary || images.missingSecondary) {{
        elements.navStageNote.textContent = "该 Step 属于 navigation 计划路线，但没有对应实际 Observe crop。";
      }} else {{
        elements.navStageNote.textContent = step.is_stop ? "该步骤为终点 stop 节点" : "审查当前阶段与目标是否匹配";
      }}
    }}

    function renderNavigation() {{
      if (!state.navigation.selectedStepId && navigationSteps().length) {{
        state.navigation.selectedStepId = navigationSteps()[0].step_id;
      }}
      renderNavigationClusterList();
      renderNavigationStepList();
      renderNavigationValidation();
      renderTrajectory();
      renderNavigationInspector();
      updateTimelineStatus();
      applyNavigationTransform();
    }}

    function observationCards() {{
      return (DASHBOARD.observation && DASHBOARD.observation.cards) || [];
    }}

    function observationValidation() {{
      return (DASHBOARD.observation && DASHBOARD.observation.contract_validation) || {{ status: "ok", violations: [] }};
    }}

    function observationCardByStep(stepId) {{
      return observationCards().find((card) => card.step_id === stepId) || null;
    }}

    function currentObservationCard() {{
      return observationCardByStep((DASHBOARD.observation && DASHBOARD.observation.selected_step_id) || state.navigation.selectedStepId || "");
    }}

    function setSelectedObservationStep(stepId) {{
      const card = observationCardByStep(stepId);
      if (!card) return;
      DASHBOARD.observation.selected_step_id = card.step_id;
      renderObservationBranchGrid();
      renderObservationMemory();
      renderObservationCards();
      renderObservationReport();
      if (elements.observationCurrentStepPill) {{
        elements.observationCurrentStepPill.textContent = card.step_id;
      }}
    }}

    function renderObservationBranchGrid() {{
      const card = currentObservationCard();
      const branchState = (card && card.resolved_branch_state) || (DASHBOARD.observation && DASHBOARD.observation.branch_state) || {{}};
      const branchOrder = [
        ["serrated", "锯齿状"],
        ["abnormal_crypt", "异常隐窝"],
        ["conventional", "传统腺瘤"],
        ["dysplasia", "异型增生"],
      ];
      elements.observationBranchGrid.innerHTML = branchOrder.map(([key, label]) => {{
        const stateValue = branchState[key] || "unresolved";
        const display = BRANCH_STATE_DISPLAY[stateValue] || BRANCH_STATE_DISPLAY.unresolved;
        return `<div class="obs-branch-card ${{display.class_name}}"><div class="obs-branch-head"><strong>${{label}}</strong><span class="obs-branch-state">${{display.icon}} ${{display.label}}</span></div><div class="tiny">${{card ? ("当前步：" + card.step_id) : "尚未选择步骤"}}</div></div>`;
      }}).join("");
    }}

    function renderObservationMemory() {{
      const card = currentObservationCard();
      const validation = observationValidation();
      const unresolved = (card && card.unresolved_questions) || [];
      const sufficient = (card && card.sufficient_evidence) || [];
      elements.observationMemoryList.innerHTML = `
        <div class="obs-memory-card"><strong>Observation 记忆数</strong><div class="tiny">${{card ? card.memory_observation_count : 0}}</div></div>
        <div class="obs-memory-card"><strong>Chief Review 数</strong><div class="tiny">${{card ? card.memory_review_count : 0}}</div></div>
        <div class="obs-memory-card"><strong>Unresolved Questions</strong><div class="tiny">${{unresolved.length}}</div></div>
        <div class="obs-memory-card"><strong>Sufficient Evidence</strong><div class="tiny">${{sufficient.length}}</div></div>
        <div class="obs-memory-card"><strong>Contract 状态</strong><div class="tiny">${{validation.status === "ok" ? "正常" : "警告"}}</div></div>
      `;
    }}

    function renderObservationValidation() {{
      const validation = observationValidation();
      const violations = validation.violations || [];
      elements.observationValidationStatus.textContent = validation.status === "ok" ? "正常" : "警告";
      elements.observationValidationSummary.textContent = violations.length ? ("共 " + violations.length + " 项异常") : "Observation / Reasoning Contract 通过";
      if (!violations.length) {{
        elements.observationViolationList.innerHTML = '<div class="empty-state">当前 observation 视图未发现 contract 异常。</div>';
        return;
      }}
      elements.observationViolationList.innerHTML = violations.map((item) => `<div class="violation-item"><strong>${{item.step_id || item.type}}</strong><div class="tiny">${{item.message || ""}}</div></div>`).join("");
    }}

    function renderObservationCards() {{
      const cards = observationCards();
      if (!cards.length) {{
        elements.observationCardList.innerHTML = '<div class="empty-state">没有可显示的 Observation / Reasoning 记录。</div>';
        return;
      }}
      const selectedId = (DASHBOARD.observation && DASHBOARD.observation.selected_step_id) || cards[0].step_id;
      elements.observationCardList.innerHTML = cards.map((card, index) => {{
        const active = card.step_id === selectedId ? " is-active" : "";
        const decisionBanner = card.is_early_stop
          ? `<div class="obs-banner early-stop">🟢 EARLY STOP TRIGGERED</div>`
          : `<div class="obs-banner continue">🟡 CONTINUE SCANNING</div>`;
        const findingChips = []
          .concat((card.level_1_findings || []).map((item) => `<span class="obs-finding-chip level1">L1 · ${{item}}</span>`))
          .concat((card.level_2_findings || []).map((item) => `<span class="obs-finding-chip level2">L2 · ${{item}}</span>`))
          .concat((card.level_3_findings || []).map((item) => `<span class="obs-finding-chip level3">L3 · ${{item}}</span>`))
          .join("");
        const nextTargetHtml = card.next_visual_target
          ? `<div class="obs-next-target"><strong>下一视觉目标</strong><div class="tiny">cluster: ${{card.next_visual_target.target_cluster_id || "-"}}</div><div class="tiny">branch: ${{card.next_visual_target.target_branch || "-"}}</div><div class="tiny">target: ${{card.next_visual_target.target_region_semantic || "-"}}</div><div class="tiny">${{(card.next_visual_target.target_morphology_prompt || []).join(" / ")}}</div></div>`
          : "";
        const evidenceHtml = card.sufficient_evidence && card.sufficient_evidence.length
          ? `<div class="obs-evidence-box"><strong>Sufficient Evidence</strong><ul>${{card.sufficient_evidence.map((item) => `<li>${{item}}</li>`).join("")}}</ul></div>`
          : "";
        const unresolvedHtml = card.unresolved_questions && card.unresolved_questions.length
          ? `<div class="obs-evidence-box"><strong>Unresolved Questions</strong><ul>${{card.unresolved_questions.map((item) => `<li>${{item}}</li>`).join("")}}</ul></div>`
          : "";
        const chiefThinking = card.chief_thinking
          ? `<details class="obs-chief-thinking"><summary>Chief Thinking Process</summary><pre>${{card.chief_thinking}}</pre></details>`
          : "";
        return `<article class="obs-card ${{card.decision_class}}${{active}}" data-observation-step="${{card.step_id}}">
          <div class="obs-card-header">
            <div>
              <strong>${{card.step_id}}</strong>
              <div class="tiny">${{card.cluster_id || "-"}} · ${{card.review_goal || "-"}} · ${{card.stage_gate || "-"}}</div>
            </div>
            <div class="obs-card-badges">
              <span class="obs-badge">Junior ${{card.observation_confidence.toFixed(2)}}</span>
              <span class="obs-badge">Chief ${{card.chief_confidence.toFixed(2)}}</span>
            </div>
          </div>
          ${{decisionBanner}}
          <div class="obs-card-body">
            <div class="obs-image-grid">
              <div class="obs-image-frame"><img class="obs-image" src="${{card.overview_image || card.local_image || card.detail_image || ""}}" alt="overview" /><div class="obs-image-label">低倍概览</div></div>
              <div class="obs-image-frame"><img class="obs-image" src="${{card.detail_image || card.local_image || card.overview_image || ""}}" alt="detail" /><div class="obs-image-label">高倍局部</div></div>
            </div>
            <div class="obs-dialogue">
              <div class="obs-bubble junior">
                <strong>Junior Screener</strong>
                <p class="reason-text">${{card.observation || "-"}}</p>
                <p class="reason-text">${{card.reasoning || "-"}}</p>
                <div class="obs-findings">${{findingChips || '<span class="tiny">没有结构化 findings</span>'}}</div>
              </div>
              <div class="obs-bubble chief">
                <strong>Chief Pathologist</strong>
                <p class="reason-text">Decision: ${{card.decision}}</p>
                <p class="reason-text">${{card.continue_reason || card.branch_correction_reason || "Chief 未提供额外说明。"}}</p>
                ${{chiefThinking}}
                ${{nextTargetHtml}}
                ${{evidenceHtml}}
                ${{unresolvedHtml}}
              </div>
            </div>
          </div>
        </article>`;
      }}).join("");
      elements.observationCardList.querySelectorAll("[data-observation-step]").forEach((node) => {{
        node.addEventListener("click", () => {{
          setSelectedObservationStep(node.getAttribute("data-observation-step"));
        }});
      }});
    }}

    function renderObservationReport() {{
      const observation = DASHBOARD.observation || {{}};
      const unlocked = !!observation.unlocked;
      elements.observationTotalSteps.textContent = String(observationCards().length);
      elements.observationCurrentStepPill.textContent = (observation.selected_step_id || "-");
      elements.observationFinalStatePill.textContent = unlocked ? "已解锁" : "锁定";
      if (!unlocked) {{
        elements.observationReportRoot.innerHTML = '<div class="obs-report-locked">等待 Chief Pathologist 达成 Early Stop 共识...</div>';
        return;
      }}
      const tree = observation.hierarchical_prediction || {{}};
      const treeRows = Object.keys(tree).length
        ? Object.entries(tree).map(([key, value]) => `<div class="obs-tree-node"><strong>${{key}}</strong><div class="tiny">${{typeof value === "string" ? value : JSON.stringify(value)}}</div></div>`).join("")
        : '<div class="empty-state">没有分层预测树。</div>';
      const accordions = (observation.report_sections || []).map((section) => `<details><summary>${{section.key}}</summary><div class="tiny" style="margin-top:8px;">${{(section.items || []).length ? (section.items || []).map((item) => JSON.stringify(item)).join("<br/>") : "空"}}</div></details>`).join("");
      const recommendations = (observation.integrated_recommendations || []).map((item) => `<li>${{item}}</li>`).join("");
      elements.observationReportRoot.innerHTML = `
        <div class="obs-report-card">
          <p class="panel-kicker">Hierarchical Prediction</p>
          <div class="obs-report-tree">${{treeRows}}</div>
        </div>
        <div class="obs-report-card obs-report-accordion">
          <p class="panel-kicker">Checklist Accordion</p>
          ${{accordions || '<div class="empty-state">没有 checklist。</div>'}}
        </div>
        <div class="obs-report-card">
          <p class="panel-kicker">Integrated Report</p>
          <p class="reason-text">${{observation.integrated_summary || "暂无最终报告摘要。"}}</p>
          ${{recommendations ? `<ul>${{recommendations}}</ul>` : ""}}
        </div>
      `;
    }}

    function renderObservation() {{
      if (!(DASHBOARD.observation && DASHBOARD.observation.selected_step_id) && observationCards().length) {{
        DASHBOARD.observation.selected_step_id = observationCards()[0].step_id;
      }}
      renderObservationBranchGrid();
      renderObservationMemory();
      renderObservationValidation();
      renderObservationCards();
      renderObservationReport();
    }}

    function layoutStorageKey() {{
      return "agent_dashboard_layout_v1_" + ({case_id_json} || "case") + "_" + ({slide_id_json} || "slide");
    }}

    function loadLayoutState() {{
      try {{
        const raw = window.localStorage.getItem(layoutStorageKey());
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {{
          if (typeof parsed.leftWidth === "number") state.layout.leftWidth = parsed.leftWidth;
          if (typeof parsed.rightWidth === "number") state.layout.rightWidth = parsed.rightWidth;
          if (typeof parsed.leftCollapsed === "boolean") state.layout.leftCollapsed = parsed.leftCollapsed;
          if (typeof parsed.rightCollapsed === "boolean") state.layout.rightCollapsed = parsed.rightCollapsed;
        }}
      }} catch (error) {{
      }}
    }}

    function saveLayoutState() {{
      try {{
        window.localStorage.setItem(layoutStorageKey(), JSON.stringify(state.layout));
      }} catch (error) {{
      }}
    }}

    function applyLayout() {{
      const minLeft = 280;
      const maxLeft = 520;
      const minRight = 360;
      const maxRight = 620;
      state.layout.leftWidth = Math.max(minLeft, Math.min(maxLeft, state.layout.leftWidth));
      state.layout.rightWidth = Math.max(minRight, Math.min(maxRight, state.layout.rightWidth));
      elements.dashboardShell.style.setProperty("--sidebar-left-width", state.layout.leftCollapsed ? "0px" : state.layout.leftWidth + "px");
      elements.dashboardShell.style.setProperty("--sidebar-right-width", state.layout.rightCollapsed ? "0px" : state.layout.rightWidth + "px");
      elements.dashboardShell.classList.toggle("is-left-collapsed", state.layout.leftCollapsed);
      elements.dashboardShell.classList.toggle("is-right-collapsed", state.layout.rightCollapsed);
      elements.toggleLeftHeader.textContent = state.layout.leftCollapsed ? "展开左栏" : "收起左栏";
      elements.toggleRightHeader.textContent = state.layout.rightCollapsed ? "展开右栏" : "收起右栏";
      elements.toggleLeftRail.textContent = state.layout.leftCollapsed ? "展" : "收";
      elements.toggleRightRail.textContent = state.layout.rightCollapsed ? "展" : "收";
      saveLayoutState();
    }}

    function toggleLeftColumn() {{
      state.layout.leftCollapsed = !state.layout.leftCollapsed;
      applyLayout();
    }}

    function toggleRightColumn() {{
      state.layout.rightCollapsed = !state.layout.rightCollapsed;
      applyLayout();
    }}

    function startResize(side, clientX) {{
      if (window.innerWidth <= 1180) return;
      state.resize.active = true;
      state.resize.side = side;
      state.resize.startX = clientX;
      state.resize.startLeftWidth = state.layout.leftWidth;
      state.resize.startRightWidth = state.layout.rightWidth;
      if (side === "left") elements.leftResizer.classList.add("is-dragging");
      if (side === "right") elements.rightResizer.classList.add("is-dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }}

    function updateResize(clientX) {{
      if (!state.resize.active) return;
      const delta = clientX - state.resize.startX;
      if (state.resize.side === "left") {{
        state.layout.leftCollapsed = false;
        state.layout.leftWidth = state.resize.startLeftWidth + delta;
      }} else if (state.resize.side === "right") {{
        state.layout.rightCollapsed = false;
        state.layout.rightWidth = state.resize.startRightWidth - delta;
      }}
      applyLayout();
    }}

    function stopResize() {{
      if (!state.resize.active) return;
      state.resize.active = false;
      state.resize.side = null;
      elements.leftResizer.classList.remove("is-dragging");
      elements.rightResizer.classList.remove("is-dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }}

    function visiblePatchKeys() {{
      const keys = [];
      for (const patch of state.patchMap.values()) {{
        if (!state.labelFilters.has(patch.label)) continue;
        if (state.scoreFilter !== "all" && patch.score_bucket !== state.scoreFilter) continue;
        if (state.extraFilters.has("require_high_mag") && !patch.require_high_magnification) continue;
        if (state.extraFilters.has("risk_disagreement") && patch.agreement_status !== "risk_disagreement") continue;
        if (state.extraFilters.has("priority_high") && Number(patch.diagnostic_priority) < 3) continue;
        keys.push(patch.patch_key);
      }}
      return keys;
    }}

    function highlightReasoning(text) {{
      let html = String(text || "");
      REASON_TOKENS.forEach((token) => {{
        const pattern = new RegExp("(" + token.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&") + ")", "gi");
        html = html.replace(pattern, '<span class="highlight-token">$1</span>');
      }});
      return html;
    }}

    function setViewMode(mode) {{
      state.viewMode = mode;
      elements.modeDefault.classList.toggle("is-active", mode === "default_score_mode");
      elements.modeEntropy.classList.toggle("is-active", mode === "entropy_disagreement_mode");
      renderCanvas();
    }}

    function setAuditMode(mode) {{
      state.auditMode = mode;
      elements.auditStandard.classList.toggle("is-active", mode === "standard");
      elements.auditTurbo.classList.toggle("is-active", mode === "turbo_audit");
    }}

    function selectPatch(key, scrollIntoView = false) {{
      if (!key || !patchByKey(key)) return;
      state.selectedPatchKey = key;
      state.pendingAudit = null;
      elements.auditComment.value = "";
      renderCanvas();
      renderInspector();
      if (scrollIntoView) {{
        const button = document.querySelector('.patch-cell[data-patch-key="' + key + '"]');
        if (button) button.scrollIntoView({{ block: "nearest", inline: "nearest" }});
      }}
    }}

    function ensureSelectedPatch() {{
      if (state.selectedPatchKey && patchByKey(state.selectedPatchKey)) return;
      const keys = visiblePatchKeys();
      if (keys.length) {{
        state.selectedPatchKey = keys[0];
      }} else {{
        const all = [...state.patchMap.keys()];
        state.selectedPatchKey = all.length ? all[0] : null;
      }}
    }}

    function nextVisiblePatch(step) {{
      const keys = visiblePatchKeys();
      if (!keys.length) return null;
      const current = state.selectedPatchKey;
      const currentIndex = Math.max(0, keys.indexOf(current));
      const nextIndex = (currentIndex + step + keys.length) % keys.length;
      return keys[nextIndex];
    }}

    function applyPanZoom() {{
      elements.canvasStage.style.transform = 'translate(calc(-50% + ' + state.panX + 'px), calc(-50% + ' + state.panY + 'px)) scale(' + state.zoom + ')';
    }}

    function zoomBy(delta) {{
      state.zoom = Math.max(0.6, Math.min(3.2, round(state.zoom + delta, 2)));
      applyPanZoom();
    }}

    function resetZoom() {{
      state.zoom = 1;
      state.panX = 0;
      state.panY = 0;
      applyPanZoom();
    }}

    function pendingOrPatch() {{
      const patch = patchByKey(state.selectedPatchKey);
      if (!patch) return null;
      if (!state.pendingAudit || state.pendingAudit.patchKey !== state.selectedPatchKey) {{
        return patch;
      }}
      return {{
        ...patch,
        label: state.pendingAudit.label,
        diagnostic_priority: state.pendingAudit.priority,
        require_high_magnification: state.pendingAudit.requireHighMag
      }};
    }}

    function beginPendingAudit(label) {{
      const patch = patchByKey(state.selectedPatchKey);
      if (!patch) return;
      const priority = defaultPriorityFor(label, patch.diagnostic_priority);
      state.pendingAudit = {{
        patchKey: patch.patch_key,
        label,
        priority,
        requireHighMag: patch.require_high_magnification,
        comment: elements.auditComment.value.trim()
      }};
      renderInspector();
      if (state.auditMode === "turbo_audit") {{
        saveAudit(true);
      }}
    }}

    function cancelPendingAudit() {{
      state.pendingAudit = null;
      elements.auditComment.value = "";
      renderInspector();
    }}

    function saveAudit(autoAdvance = false) {{
      const patch = patchByKey(state.selectedPatchKey);
      if (!patch) return;
      const current = pendingOrPatch();
      const comment = elements.auditComment.value.trim();
      const labelChanged = current.label !== patch.label;
      const priorityChanged = Number(current.diagnostic_priority) !== Number(patch.diagnostic_priority);
      const highMagChanged = Boolean(current.require_high_magnification) !== Boolean(patch.require_high_magnification);
      if (!labelChanged && !priorityChanged && !highMagChanged && !comment) {{
        if (autoAdvance) {{
          const next = nextVisiblePatch(1);
          if (next) selectPatch(next, true);
        }}
        return;
      }}
      state.history.push({{
        patchKey: patch.patch_key,
        previous: cloneValue(patch),
        previousAuditLog: [...state.auditLog]
      }});
      patch.label = current.label;
      patch.label_short = LABEL_SHORT[current.label] || current.label;
      patch.diagnostic_priority = Number(current.diagnostic_priority);
      patch.require_high_magnification = Boolean(current.require_high_magnification);
      patch.audited = true;
      patch.fill = (DASHBOARD.label_styles[current.label] && DASHBOARD.label_styles[current.label].fill) || patch.fill;
      patch.stroke = (DASHBOARD.label_styles[current.label] && DASHBOARD.label_styles[current.label].stroke) || patch.stroke;
      patch.text = (DASHBOARD.label_styles[current.label] && DASHBOARD.label_styles[current.label].text) || patch.text;
      patch.high_mag_url = patch.high_mag_url || "";
      state.auditLog.unshift({{
        audit_id: "audit_" + String(Date.now()),
        patch_id: patch.patch_id,
        original_region_semantic: state.history[state.history.length - 1].previous.label,
        corrected_region_semantic: patch.label,
        original_diagnostic_priority: state.history[state.history.length - 1].previous.diagnostic_priority,
        corrected_diagnostic_priority: patch.diagnostic_priority,
        original_require_high_magnification: state.history[state.history.length - 1].previous.require_high_magnification,
        corrected_require_high_magnification: patch.require_high_magnification,
        operator: "local_reviewer",
        comment,
        created_at: new Date().toISOString()
      }});
      state.pendingAudit = null;
      elements.auditComment.value = "";
      renderCanvas();
      renderStats();
      renderInspector();
      renderAuditLog();
      if (autoAdvance) {{
        const next = nextVisiblePatch(1);
        if (next) selectPatch(next, true);
      }}
    }}

    function undoAudit() {{
      const item = state.history.pop();
      if (!item) return;
      state.patchMap.set(item.patchKey, cloneValue(item.previous));
      state.auditLog = [...item.previousAuditLog];
      renderCanvas();
      renderStats();
      renderInspector();
      renderAuditLog();
    }}

    function renderCanvas() {{
      ensureSelectedPatch();
      const visible = new Set(visiblePatchKeys());
      document.querySelectorAll(".patch-cell").forEach((node) => {{
        const key = node.getAttribute("data-patch-key");
        const patch = patchByKey(key);
        if (!patch) return;
        const isVisible = visible.has(key);
        const isSelected = key === state.selectedPatchKey;
        const isMuted = state.viewMode === "entropy_disagreement_mode" && !patch.is_uncertain_focus;
        const isUncertain = state.viewMode === "entropy_disagreement_mode" && patch.is_uncertain_focus;
        node.classList.toggle("is-hidden", !isVisible);
        node.classList.toggle("is-selected", isSelected);
        node.classList.toggle("is-audited", !!patch.audited);
        node.classList.toggle("is-uncertain", isUncertain);
        node.classList.toggle("is-muted", isVisible && isMuted);
        node.classList.toggle("is-dimmed", !isVisible);
        if (state.viewMode === "entropy_disagreement_mode") {{
          node.style.opacity = isVisible ? (patch.is_uncertain_focus ? "1" : "0.1") : "0.08";
        }} else {{
          node.style.opacity = isVisible ? "1" : "0.16";
        }}
        const fill = state.viewMode === "entropy_disagreement_mode" && patch.is_uncertain_focus
          ? DASHBOARD.entropy_fill_by_label[patch.label] || patch.fill
          : patch.fill;
        node.style.background = fill;
        node.style.borderColor = patch.stroke;
      }});
    }}

    function renderStats() {{
      const visible = visiblePatchKeys();
      const visibleSet = new Set(visible);
      let highPriority = 0;
      let audited = 0;
      let disagreements = 0;
      let uncertain = 0;
      const counts = Object.fromEntries(LABELS.map((label) => [label, 0]));
      for (const patch of state.patchMap.values()) {{
        if (!visibleSet.has(patch.patch_key)) continue;
        counts[patch.label] = (counts[patch.label] || 0) + 1;
        if (Number(patch.diagnostic_priority) >= 3) highPriority += 1;
        if (patch.audited) audited += 1;
        if (patch.agreement_status === "risk_disagreement") disagreements += 1;
        if (patch.is_uncertain_focus) uncertain += 1;
      }}
      const summary = {{
        total: DASHBOARD.stats.total_patches,
        filtered: visible.length,
        priority: highPriority,
        audited,
        disagreement: disagreements,
        uncertainty: uncertain,
        missing: DASHBOARD.stats.missing_patches,
        orphan: DASHBOARD.stats.orphan_patches
      }};
      elements.statsGrid.innerHTML = `
        <div class="stat-card"><span class="stat-label">当前总数</span><span class="stat-value">${{summary.total}}</span></div>
        <div class="stat-card"><span class="stat-label">过滤后</span><span class="stat-value">${{summary.filtered}}</span></div>
        <div class="stat-card"><span class="stat-label">优先级 3+</span><span class="stat-value">${{summary.priority}}</span></div>
        <div class="stat-card"><span class="stat-label">已审计</span><span class="stat-value">${{summary.audited}}</span></div>
        <div class="stat-card"><span class="stat-label">高风险分歧</span><span class="stat-value">${{summary.disagreement}}</span></div>
        <div class="stat-card"><span class="stat-label">灰区焦点</span><span class="stat-value">${{summary.uncertainty}}</span></div>
        <div class="stat-card"><span class="stat-label">缺失</span><span class="stat-value">${{summary.missing}}</span></div>
        <div class="stat-card"><span class="stat-label">孤儿资产</span><span class="stat-value">${{summary.orphan}}</span></div>
      `;
    }}

    function renderInspector() {{
      ensureSelectedPatch();
      const patch = pendingOrPatch();
      if (!patch) {{
        elements.detailFusion.textContent = "尚未选中 Patch。";
        return;
      }}
      const live = patchByKey(state.selectedPatchKey);
      elements.inspectorImage.src = live.high_mag_url || "";
      elements.inspectorImageBadge.textContent = patch.patch_text + " · 20x 高倍图";
      elements.detailPatchId.textContent = patch.patch_text;
      elements.detailLocation.textContent = "第 " + patch.row + " 行，第 " + patch.col + " 列";
      elements.detailClusterId.textContent = live.cluster_id || "-";
      elements.detailScoreOrigin.textContent = displayScoreOrigin(live.score_origin);
      elements.detailScore.textContent = round(live.score, 2).toFixed(2);
      elements.detailUncertainty.textContent = live.uncertainty_score == null ? "-" : round(live.uncertainty_score, 2).toFixed(2);
      elements.detailSemantic.textContent = displaySemantic(patch.label);
      elements.detailPriority.textContent = String(patch.diagnostic_priority);
      elements.detailHighMag.textContent = patch.require_high_magnification ? "是" : "否";
      elements.detailAgreement.textContent = displayAgreement(live.agreement_status);
      elements.detailConch.textContent = live.conch_region_semantic || "-";
      elements.detailPatho.textContent = live.pathoreasoner_r1_region_semantic || "-";
      elements.detailFusion.innerHTML = highlightReasoning(live.fusion_reasoning || "");
      elements.auditHighMag.checked = Boolean(patch.require_high_magnification);
      document.querySelectorAll(".audit-class").forEach((node) => {{
        node.classList.toggle("is-selected", node.getAttribute("data-label") === patch.label);
      }});
    }}

    function renderAuditLog() {{
      if (!state.auditLog.length) {{
        elements.auditLogList.innerHTML = '<div class="empty-state">暂时没有审计记录。</div>';
        return;
      }}
      elements.auditLogList.innerHTML = state.auditLog.slice(0, 8).map((item) => {{
        const patchId = Array.isArray(item.patch_id) ? '[' + item.patch_id.join(',') + ']' : String(item.patch_id);
        const comment = item.comment ? `<div class="tiny">${{item.comment}}</div>` : '';
        return `<div class="audit-item"><strong>${{patchId}} · ${{item.corrected_region_semantic}}</strong><div class="tiny">${{item.original_region_semantic}} → ${{item.corrected_region_semantic}}</div>${{comment}}</div>`;
      }}).join("");
    }}

    function syncFilterChips() {{
      document.querySelectorAll("#label-filter-row .chip").forEach((node) => {{
        const label = node.getAttribute("data-label-filter");
        node.classList.toggle("is-active", state.labelFilters.has(label));
      }});
      document.querySelectorAll("#score-filter-row .chip").forEach((node) => {{
        node.classList.toggle("is-active", node.getAttribute("data-score-filter") === state.scoreFilter);
      }});
      document.querySelectorAll("#extra-filter-row .chip").forEach((node) => {{
        const key = node.getAttribute("data-extra-filter");
        node.classList.toggle("is-active", state.extraFilters.has(key));
      }});
    }}

    function attachEvents() {{
      elements.patchButtons.forEach((node) => {{
        node.addEventListener("mouseenter", () => {{
          const patch = patchByKey(node.getAttribute("data-patch-key"));
          if (!patch) return;
          elements.hoverReadout.textContent = `${{patch.patch_text}} · ${{LABEL_SHORT[patch.label] || patch.label}} · 分数 ${{round(patch.score, 2).toFixed(2)}} · 优先级 ${{patch.diagnostic_priority}}`;
        }});
        node.addEventListener("mouseleave", () => {{
          elements.hoverReadout.textContent = "悬停或点击 Patch 以查看详情。";
        }});
        node.addEventListener("click", () => selectPatch(node.getAttribute("data-patch-key")));
      }});
      document.querySelectorAll("#label-filter-row .chip").forEach((node) => {{
        node.addEventListener("click", () => {{
          const label = node.getAttribute("data-label-filter");
          if (state.labelFilters.has(label)) {{
            if (state.labelFilters.size > 1) state.labelFilters.delete(label);
          }} else {{
            state.labelFilters.add(label);
          }}
          syncFilterChips();
          renderCanvas();
          renderStats();
        }});
      }});
      document.querySelectorAll("#score-filter-row .chip").forEach((node) => {{
        node.addEventListener("click", () => {{
          state.scoreFilter = node.getAttribute("data-score-filter");
          syncFilterChips();
          renderCanvas();
          renderStats();
        }});
      }});
      document.querySelectorAll("#extra-filter-row .chip").forEach((node) => {{
        node.addEventListener("click", () => {{
          const key = node.getAttribute("data-extra-filter");
          if (state.extraFilters.has(key)) state.extraFilters.delete(key);
          else state.extraFilters.add(key);
          syncFilterChips();
          renderCanvas();
          renderStats();
        }});
      }});
      elements.viewTabCaseStory.addEventListener("click", () => setActiveView("case_story"));
      elements.viewTabScreening.addEventListener("click", () => setActiveView("screening"));
      elements.viewTabNavigation.addEventListener("click", () => setActiveView("navigation"));
      elements.viewTabObservation.addEventListener("click", () => setActiveView("observation"));
      elements.modeDefault.addEventListener("click", () => setViewMode("default_score_mode"));
      elements.modeEntropy.addEventListener("click", () => setViewMode("entropy_disagreement_mode"));
      elements.auditStandard.addEventListener("click", () => setAuditMode("standard"));
      elements.auditTurbo.addEventListener("click", () => setAuditMode("turbo_audit"));
      document.querySelectorAll(".audit-class").forEach((node) => {{
        node.addEventListener("click", () => beginPendingAudit(node.getAttribute("data-label")));
      }});
      elements.auditHighMag.addEventListener("change", () => {{
        const patch = patchByKey(state.selectedPatchKey);
        if (!patch) return;
        const fallbackLabel = (state.pendingAudit && state.pendingAudit.label) || patch.label;
        state.pendingAudit = {{
          patchKey: patch.patch_key,
          label: fallbackLabel,
          priority: defaultPriorityFor(fallbackLabel, patch.diagnostic_priority),
          requireHighMag: elements.auditHighMag.checked,
          comment: elements.auditComment.value.trim()
        }};
        renderInspector();
      }});
      elements.saveAuditButton.addEventListener("click", () => saveAudit(false));
      elements.cancelAuditButton.addEventListener("click", cancelPendingAudit);
      elements.undoAuditButton.addEventListener("click", undoAudit);
      elements.zoomIn.addEventListener("click", () => zoomBy(0.2));
      elements.zoomOut.addEventListener("click", () => zoomBy(-0.2));
      elements.zoomReset.addEventListener("click", resetZoom);
      elements.navZoomIn.addEventListener("click", () => {{
        state.navigation.zoom = Math.max(0.6, Math.min(3.2, round(state.navigation.zoom + 0.2, 2)));
        applyNavigationTransform();
      }});
      elements.navZoomOut.addEventListener("click", () => {{
        state.navigation.zoom = Math.max(0.6, Math.min(3.2, round(state.navigation.zoom - 0.2, 2)));
        applyNavigationTransform();
      }});
      elements.navZoomReset.addEventListener("click", () => {{
        state.navigation.zoom = 1;
        state.navigation.panX = 0;
        state.navigation.panY = 0;
        applyNavigationTransform();
      }});
      elements.timelinePrev.addEventListener("click", () => {{
        stopNavigationPlayback();
        selectNavigationAdjacentStep(-1);
      }});
      elements.timelineNext.addEventListener("click", () => {{
        stopNavigationPlayback();
        selectNavigationAdjacentStep(1);
      }});
      elements.timelinePlay.addEventListener("click", () => toggleNavigationPlayback());
      elements.timelineRange.addEventListener("input", () => {{
        stopNavigationPlayback();
        const steps = navigationSteps();
        const index = Math.max(0, Math.min(steps.length - 1, Number(elements.timelineRange.value)));
        if (steps[index]) setSelectedNavigationStep(steps[index].step_id, {{ focus: true }});
      }});
      elements.timelineSpeed.addEventListener("change", () => {{
        state.navigation.speed = Number(elements.timelineSpeed.value) || 1;
        elements.navigationCurrentSpeedPill.textContent = String(state.navigation.speed) + "x";
        if (state.navigation.isPlaying) {{
          playNavigation();
        }}
      }});
      elements.toggleLeftHeader.addEventListener("click", toggleLeftColumn);
      elements.toggleRightHeader.addEventListener("click", toggleRightColumn);
      elements.toggleLeftRail.addEventListener("click", toggleLeftColumn);
      elements.toggleRightRail.addEventListener("click", toggleRightColumn);
      elements.leftResizer.addEventListener("mousedown", (event) => {{
        if (event.target === elements.toggleLeftRail) return;
        event.preventDefault();
        startResize("left", event.clientX);
      }});
      elements.rightResizer.addEventListener("mousedown", (event) => {{
        if (event.target === elements.toggleRightRail) return;
        event.preventDefault();
        startResize("right", event.clientX);
      }});

      elements.canvasStage.addEventListener("mousedown", (event) => {{
        if (safeClosest(event.target, ".patch-cell")) return;
        state.drag.active = true;
        state.drag.startX = event.clientX;
        state.drag.startY = event.clientY;
        state.drag.originX = state.panX;
        state.drag.originY = state.panY;
        elements.canvasStage.classList.add("is-dragging");
      }});
      elements.navCanvasStage.addEventListener("mousedown", (event) => {{
        if (safeClosest(event.target, ".trajectory-step-node")) return;
        state.navigation.drag.active = true;
        state.navigation.drag.startX = event.clientX;
        state.navigation.drag.startY = event.clientY;
        state.navigation.drag.originX = state.navigation.panX;
        state.navigation.drag.originY = state.navigation.panY;
        elements.navCanvasStage.classList.add("is-dragging");
      }});
      window.addEventListener("mousemove", (event) => {{
        updateResize(event.clientX);
        if (state.drag.active) {{
          state.panX = state.drag.originX + (event.clientX - state.drag.startX);
          state.panY = state.drag.originY + (event.clientY - state.drag.startY);
          applyPanZoom();
        }}
        if (state.navigation.drag.active) {{
          state.navigation.panX = state.navigation.drag.originX + (event.clientX - state.navigation.drag.startX);
          state.navigation.panY = state.navigation.drag.originY + (event.clientY - state.navigation.drag.startY);
          applyNavigationTransform();
        }}
      }});
      window.addEventListener("mouseup", () => {{
        stopResize();
        state.drag.active = false;
        elements.canvasStage.classList.remove("is-dragging");
        state.navigation.drag.active = false;
        elements.navCanvasStage.classList.remove("is-dragging");
      }});
      elements.canvasScroll.addEventListener("wheel", (event) => {{
        event.preventDefault();
        zoomBy(event.deltaY < 0 ? 0.1 : -0.1);
      }}, {{ passive: false }});
      elements.navCanvasScroll.addEventListener("wheel", (event) => {{
        event.preventDefault();
        state.navigation.zoom = Math.max(0.6, Math.min(3.2, round(state.navigation.zoom + (event.deltaY < 0 ? 0.1 : -0.1), 2)));
        applyNavigationTransform();
      }}, {{ passive: false }});

      window.addEventListener("keydown", (event) => {{
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {{
          event.preventDefault();
          if (state.activeView === "screening") {{
            undoAudit();
          }}
          return;
        }}
        if (event.target && ["TEXTAREA", "INPUT"].includes(event.target.tagName)) {{
          if (event.key === "Escape") {{
            event.preventDefault();
            cancelPendingAudit();
            event.target.blur();
          }}
          return;
        }}
        if (KEY_TO_LABEL[event.key]) {{
          if (state.activeView !== "screening") return;
          event.preventDefault();
          beginPendingAudit(KEY_TO_LABEL[event.key]);
          return;
        }}
        if (event.key === "[") {{
          event.preventDefault();
          if (state.activeView === "screening") {{
            const prev = nextVisiblePatch(-1);
            if (prev) selectPatch(prev, true);
          }} else if (state.activeView === "navigation") {{
            selectNavigationAdjacentStep(-1);
          }}
          return;
        }}
        if (event.key === "]") {{
          event.preventDefault();
          if (state.activeView === "screening") {{
            const next = nextVisiblePatch(1);
            if (next) selectPatch(next, true);
          }} else if (state.activeView === "navigation") {{
            selectNavigationAdjacentStep(1);
          }}
          return;
        }}
        if (event.key.toLowerCase() === "h") {{
          if (state.activeView !== "screening") return;
          event.preventDefault();
          elements.auditHighMag.checked = !elements.auditHighMag.checked;
          elements.auditHighMag.dispatchEvent(new Event("change"));
          return;
        }}
        if (event.key === "Enter") {{
          event.preventDefault();
          if (state.activeView === "screening") {{
            saveAudit(false);
          }} else if (state.activeView === "navigation") {{
            toggleNavigationPlayback();
          }}
          return;
        }}
        if (event.key === "Escape") {{
          event.preventDefault();
          if (state.activeView === "screening") {{
            cancelPendingAudit();
          }} else if (state.activeView === "navigation") {{
            stopNavigationPlayback();
          }}
        }}
      }});
    }}

    ensureSelectedPatch();
    loadLayoutState();
    safeRun("applyLayout", applyLayout);
    safeRun("syncFilterChips", syncFilterChips);
    safeRun("attachEvents", attachEvents);
    safeRun("renderCanvas", renderCanvas);
    safeRun("renderStats", renderStats);
    safeRun("renderInspector", renderInspector);
    safeRun("renderAuditLog", renderAuditLog);
    safeRun("renderCaseStory", renderCaseStory);
    safeRun("renderNavigation", renderNavigation);
    safeRun("renderObservation", renderObservation);
    safeRun("applyPanZoom", applyPanZoom);
    safeRun("setActiveView(case_story)", () => setActiveView("case_story"));
  </script>
</body>
</html>
""".format(
        title=escape(context["title"]),
        data_source_label=escape(context["data_source_label"]),
        metadata_cards=context["metadata_cards"],
        rubric_cards=context["rubric_cards"],
        label_filter_buttons=context["label_filter_buttons"],
        stats_cards=context["stats_cards"],
        validation_status=escape(context["validation_status"]),
        validation_summary=escape(context["validation_summary"]),
        violation_items=context["violation_items"],
        case_id=escape(context["case_id"]),
        slide_id=escape(context["slide_id"]),
        selected_patch_count=escape(str(context["selected_patch_count"])),
        rendering_threshold=escape(context["rendering_threshold"]),
        rendering_base=escape(context["rendering_base"]),
        canvas_width=escape(str(context["canvas_width"])),
        canvas_height=escape(str(context["canvas_height"])),
        overview_image_url=escape(context["overview_image_url"]),
        patch_cells=context["patch_cells"],
        missing_cells=context["missing_cells"],
        orphan_open="open" if context["orphan_count"] else "",
        orphan_count=escape(str(context["orphan_count"])),
        orphan_cards=context["orphan_cards"],
        audit_class_buttons=context["audit_class_buttons"],
        audit_items=context["audit_items"],
        dashboard_json=_script_json_text(context["dashboard_state"]),
        labels_json=json.dumps(LABEL_ORDER, ensure_ascii=False),
        label_short_json=json.dumps(LABEL_SHORT, ensure_ascii=False),
        agreement_display_json=json.dumps(AGREEMENT_DISPLAY, ensure_ascii=False),
        score_origin_display_json=json.dumps(SCORE_ORIGIN_DISPLAY, ensure_ascii=False),
        branch_state_display_json=json.dumps(BRANCH_STATE_DISPLAY, ensure_ascii=False),
        case_id_json=json.dumps(context["case_id"], ensure_ascii=False),
        slide_id_json=json.dumps(context["slide_id"], ensure_ascii=False),
    )


def _metadata_cards_html(payload):
    grid_meta = payload.get("grid_metadata", {})
    rows = [
        ("病例 ID", payload.get("case_id")),
        ("切片 ID", payload.get("slide_id")),
        ("网格大小", "{0} x {1}".format(grid_meta.get("grid_rows", "-"), grid_meta.get("grid_cols", "-"))),
        ("Patch 尺寸", grid_meta.get("patch_size")),
        ("Overview 倍率", grid_meta.get("overview_magnification")),
        ("已选 Patch", grid_meta.get("selected_patch_count") or len(payload.get("selected_patch_ids", []))),
        ("生成时间", grid_meta.get("generated_at")),
        ("模型版本", grid_meta.get("model_version")),
    ]
    optional_rows = [
        ("运行 ID", grid_meta.get("run_id")),
        ("实验标签", grid_meta.get("experiment_tag")),
        ("输入模式", grid_meta.get("input_mode")),
        ("后端", grid_meta.get("backend")),
    ]
    for label, value in optional_rows:
        if value not in (None, "", []):
            rows.append((label, value))
    cards = []
    for label, value in rows:
        cards.append(
            '<div class="meta-card"><span class="meta-label">{label}</span><span class="meta-value">{value}</span></div>'.format(
                label=escape(str(label)),
                value=escape(str(value)),
            )
        )
    return "".join(cards)


def _rubric_cards_html(trace_rubric):
    labels = trace_rubric.get("labels") if isinstance(trace_rubric, dict) else None
    if not isinstance(labels, list):
        labels = _default_trace_rubric_summary()
    items = []
    for item in labels:
        label = str(item.get("region_semantic", "unknown"))
        items.append(
            '<div class="legend-item"><span class="legend-swatch" style="background:{fill};border-color:{stroke};"></span>'
            '<div class="legend-copy"><strong>{name}</strong><span>优先级 {priority} · 高倍 {highmag}</span></div></div>'.format(
                fill=escape(str(item.get("color_fill", TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["fill"]))),
                stroke=escape(str(item.get("color_stroke", TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["stroke"]))),
                name=escape(str(item.get("display_name", LABEL_SHORT.get(label, label)))),
                priority=escape(str(item.get("default_priority", "-"))),
                highmag=escape("是" if item.get("require_high_magnification", False) else "否"),
            )
        )
    return "".join(items)


def _label_filter_buttons_html():
    rows = []
    for label in LABEL_ORDER:
        style = TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])
        rows.append(
            '<button class="chip is-active" data-label-filter="{label}"><span class="chip-color-dot" style="background:{stroke};"></span>{text}</button>'.format(
                label=escape(label),
                stroke=escape(style["stroke"]),
                text=escape(LABEL_SHORT.get(label, label)),
            )
        )
    return "".join(rows)


def _stats_cards_html(stats):
    rows = [
        ("当前总数", stats["total_patches"]),
        ("过滤后", stats["filtered_patches"]),
        ("优先级 3+", stats["high_priority_patches"]),
        ("已审计", stats["audited_patches"]),
        ("高风险分歧", stats["disagreement_patches"]),
        ("灰区焦点", stats["uncertain_focus_patches"]),
        ("缺失", stats["missing_patches"]),
        ("孤儿资产", stats["orphan_patches"]),
    ]
    return "".join(
        '<div class="stat-card"><span class="stat-label">{label}</span><span class="stat-value">{value}</span></div>'.format(
            label=escape(str(label)),
            value=escape(str(value)),
        )
        for label, value in rows
    )


def _violation_items_html(contract_validation):
    violations = contract_validation.get("violations", [])
    if not violations:
        return '<div class="empty-state">未检测到契约异常。</div>'
    rows = []
    for item in violations:
        vtype = str(item.get("type", "unknown"))
        patch = item.get("patch_id")
        patch_text = "[{0},{1}]".format(*patch) if isinstance(patch, list) and len(patch) == 2 else str(patch)
        rows.append(
            '<div class="violation-item"><strong>{label}</strong><div class="tiny">{patch}</div><div class="tiny">{message}</div></div>'.format(
                label=escape(ERROR_LABELS.get(vtype, vtype)),
                patch=escape(patch_text),
                message=escape(str(item.get("message", ""))),
            )
        )
    return "".join(rows)


def _patch_cells_html(serialized_patches):
    rows = []
    for patch in serialized_patches:
        rows.append(
            '<button type="button" class="patch-cell" data-patch-key="{key}" title="{title}" '
            'style="left:{left}%;top:{top}%;width:{width}%;height:{height}%;background:{fill};border-color:{stroke};">'
            '<span class="patch-chip">{patch_text}</span><span class="patch-corner">{short}</span></button>'.format(
                key=escape(patch["patch_key"]),
                title=escape("{0} | {1} | 分数 {2}".format(patch["patch_text"], patch["label"], round(patch["score"], 2))),
                left=patch["left"],
                top=patch["top"],
                width=patch["width"],
                height=patch["height"],
                fill=escape(patch["fill"]),
                stroke=escape(patch["stroke"]),
                patch_text=escape(patch["patch_text"]),
                short=escape(patch["label_short"]),
            )
        )
    return "".join(rows)


def _missing_cells_html(serialized_missing):
    rows = []
    for item in serialized_missing:
        rows.append(
            '<div class="missing-cell" title="缺失 Patch {patch_text}" style="left:{left}%;top:{top}%;width:{width}%;height:{height}%;"></div>'.format(
                patch_text=escape(item["patch_text"]),
                left=item["left"],
                top=item["top"],
                width=item["width"],
                height=item["height"],
            )
        )
    return "".join(rows)


def _audit_class_buttons_html():
    rows = []
    for label in [
        "background_artifact_stroma",
        "normal_mucosa",
        "inflammatory_polyp_like",
        "conventional_adenoma_like",
        "ssl_suspicious_mucosa",
    ]:
        rows.append(
            '<button class="audit-class" data-label="{label}" style="background:{bg};border-color:{stroke};">{text}</button>'.format(
                label=escape(label),
                bg=escape(_gradient_for_label(label)),
                stroke=escape(TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["stroke"]),
                text=escape(LABEL_SHORT.get(label, label)),
            )
        )
    return "".join(rows)


def _audit_items_html(audit_log):
    if not audit_log:
        return '<div class="empty-state">暂时没有审计记录。</div>'
    rows = []
    for item in audit_log[:8]:
        patch = item.get("patch_id")
        patch_text = "[{0},{1}]".format(*patch) if isinstance(patch, list) and len(patch) == 2 else str(patch)
        comment = str(item.get("comment", "")).strip()
        rows.append(
            '<div class="audit-item"><strong>{patch} · {semantic}</strong><div class="tiny">{before} → {after}</div>{comment_html}</div>'.format(
                patch=escape(patch_text),
                semantic=escape(str(item.get("corrected_region_semantic", ""))),
                before=escape(str(item.get("original_region_semantic", ""))),
                after=escape(str(item.get("corrected_region_semantic", ""))),
                comment_html='<div class="tiny">{0}</div>'.format(escape(comment)) if comment else "",
            )
        )
    return "".join(rows)


def _orphan_cards_html(orphan_assets):
    if not orphan_assets:
        return '<div class="empty-state">没有孤儿资产。</div>'
    rows = []
    for item in orphan_assets:
        rows.append(
            '<div class="orphan-card"><strong>{patch}</strong><div class="tiny">{reason}</div><div class="tiny">{semantic} · 分数 {score}</div></div>'.format(
                patch=escape(str(item["patch_text"])),
                reason=escape(ERROR_LABELS.get(item["reason"], item["reason"])),
                semantic=escape(str(item["region_semantic"])),
                score=escape(str(round(item["score"], 2))),
            )
        )
    return "".join(rows)


def export_dashboard(payload, output_dir, source_root=None):
    output_dir = ensure_dir(output_dir)
    payload = dict(payload)
    overview, high_mag_assets = _resolve_asset_urls(payload, output_dir, source_root=source_root)
    payload["overview_image"] = overview
    payload["high_mag_assets"] = high_mag_assets
    payload["observe_assets_by_step"] = _resolve_navigation_observe_assets(payload.get("observe_assets_by_step", []), output_dir, source_root=source_root)

    normalized = _normalize_contract_validation(payload)
    selected_patch_ids = normalized["selected_patch_ids"]
    payload["selected_patch_ids"] = selected_patch_ids
    payload["contract_validation"] = normalized["contract_validation"]
    payload["orphan_assets"] = normalized["orphan_assets"]

    grid_meta = payload.get("grid_metadata", {})
    canvas_width, canvas_height = _grid_canvas_size(grid_meta)
    grid_lookup = _grid_cell_lookup(grid_meta)
    audit_log = list(payload.get("audit_log", [])) if isinstance(payload.get("audit_log"), list) else []
    audited_keys = {_patch_key(item.get("patch_id")) for item in audit_log if isinstance(item, dict)}
    assets_by_ref = {str(item.get("asset_id", "")): item for item in high_mag_assets if isinstance(item, dict)}

    serialized_patches = []
    for patch in normalized["valid_patch_assignments"]:
        row_col = _normalize_patch_id(patch.get("patch_id") or [patch.get("row"), patch.get("col")])
        if row_col is None:
            continue
        cell = grid_lookup.get(row_col)
        if not cell:
            continue
        serialized_patches.append(_serialize_patch(patch, cell, canvas_width, canvas_height, assets_by_ref, audited_keys))

    serialized_missing = []
    for item in normalized["missing_patch_ids"]:
        row_col = _normalize_patch_id(item)
        if row_col is None:
            continue
        cell = grid_lookup.get(row_col)
        if not cell:
            continue
        serialized_missing.append(_serialize_missing_patch(row_col, cell, canvas_width, canvas_height))

    serialized_orphans = _serialize_orphan_assets(normalized["orphan_assets"])
    stats = _compute_stats(serialized_patches, len(serialized_missing), len(serialized_orphans), audit_log)
    navigation_view = _build_navigation_view_model(payload, canvas_width, canvas_height)
    observation_view = _build_observation_view_model(payload)
    case_story_view = _build_case_story_view_model(payload)

    rendering_base = "Canvas" if len(selected_patch_ids) >= 2000 else "DOM 叠层"
    validation_counts = normalized["contract_validation"].get("counts", {})
    validation_summary = ", ".join("{0}: {1}".format(ERROR_LABELS.get(key, key), value) for key, value in sorted(validation_counts.items())) or "未发现异常。"
    observed_priorities = {label: [] for label in LABEL_ORDER}
    for patch in serialized_patches:
        label = patch.get("label")
        if label in observed_priorities:
            observed_priorities[label].append(_safe_int(patch.get("diagnostic_priority"), 0))
    default_priorities = {}
    for label in LABEL_ORDER:
        rubric_default = _safe_int(TRACE_LABEL_RUBRIC.get(label, {}).get("default_priority"), 0)
        observed = [value for value in observed_priorities.get(label, []) if value is not None]
        default_priorities[label] = max([rubric_default] + observed) if observed else rubric_default
    label_styles = {
        label: {
            "fill": TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["fill"],
            "stroke": TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["stroke"],
            "text": TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["text"],
        }
        for label in LABEL_ORDER
    }
    entropy_fill_by_label = {
        label: _rgba_with_alpha(TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])["fill"], 1.0)
        for label in LABEL_ORDER
    }

    dashboard_state = {
        "patches": serialized_patches,
        "stats": stats,
        "audit_log": audit_log,
        "default_priorities": default_priorities,
        "label_styles": label_styles,
        "entropy_fill_by_label": entropy_fill_by_label,
        "initial_selected_patch_key": serialized_patches[0]["patch_key"] if serialized_patches else None,
        "navigation": navigation_view,
        "observation": observation_view,
        "case_story": case_story_view,
        "playback_state": payload.get("playback_state", {"current_step_index": 0, "is_playing": False, "speed": 1.0}),
    }
    data_source_label = "{0} / case={1}".format(
        str(payload.get("source_run_dir") or payload.get("grid_metadata", {}).get("run_id", "") or "-"),
        str(payload.get("case_id", "") or "-"),
    )

    html = _html_for_dashboard(
        {
            "title": "Agent 实验可视化看板：{0}".format(payload.get("case_id", "unknown_case")),
            "metadata_cards": _metadata_cards_html(payload),
            "rubric_cards": _rubric_cards_html(payload.get("trace_rubric", {})),
            "label_filter_buttons": _label_filter_buttons_html(),
            "stats_cards": _stats_cards_html(stats),
            "validation_status": VALIDATION_STATUS_DISPLAY.get(str(normalized["contract_validation"].get("status", "ok")), str(normalized["contract_validation"].get("status", "ok"))),
            "validation_summary": validation_summary,
            "violation_items": _violation_items_html(normalized["contract_validation"]),
            "case_id": str(payload.get("case_id", "")),
            "slide_id": str(payload.get("slide_id", "")),
            "selected_patch_count": len(selected_patch_ids),
            "rendering_threshold": "Patch 数量 >= 2000 时切换到 Canvas",
            "rendering_base": rendering_base,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "overview_image_url": overview["resolved_image_url"],
            "patch_cells": _patch_cells_html(serialized_patches),
            "missing_cells": _missing_cells_html(serialized_missing),
            "orphan_count": len(serialized_orphans),
            "orphan_cards": _orphan_cards_html(serialized_orphans),
            "audit_class_buttons": _audit_class_buttons_html(),
            "audit_items": _audit_items_html(audit_log),
            "dashboard_state": dashboard_state,
            "data_source_label": data_source_label,
        }
    )

    write_json(Path(output_dir) / "dashboard_payload.json", payload)
    write_text(Path(output_dir) / "index.html", html)
    return Path(output_dir)


def export_dashboard_from_json(payload_json, output_dir):
    payload_json = Path(payload_json)
    return export_dashboard(read_json(payload_json), output_dir, source_root=payload_json.parent)


def _case_dirs_in_harness_run(run_dir):
    run_dir = Path(run_dir)
    case_dirs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "trace").exists() or (child / "navigation").exists() or (child / "observe").exists() or (child / "case_result.json").exists():
            case_dirs.append(child)
    return case_dirs


def _read_case_result_status(case_dir):
    result_path = Path(case_dir) / "case_result.json"
    if not result_path.exists():
        return "unknown", ""
    try:
        result = read_json(result_path)
    except Exception as exc:
        return "unreadable", str(exc)
    status = str(result.get("status") or result.get("final_status") or result.get("state") or "ok")
    detail = str(result.get("error") or result.get("message") or result.get("diagnosis") or "")
    return status, detail


def _read_batch_case_summary(case_dir):
    case_dir = Path(case_dir)
    case_result = read_json(case_dir / "case_result.json") if (case_dir / "case_result.json").exists() else {}
    report = read_json(case_dir / "observe" / "pathological_report.json") if (case_dir / "observe" / "pathological_report.json").exists() else {}
    observations = read_json(case_dir / "observe" / "observation_records.json") if (case_dir / "observe" / "observation_records.json").exists() else {}
    reasoning = read_json(case_dir / "observe" / "reasoning_state.json") if (case_dir / "observe" / "reasoning_state.json").exists() else {}
    prediction = report.get("hierarchical_prediction") or case_result.get("hierarchical_prediction") or {}
    observation_count = len(observations.get("observations", [])) if isinstance(observations, dict) else 0
    chief_dir = case_dir / "observe" / "chief_reviews"
    chief_count = len(list(chief_dir.glob("*_chief_response.json"))) if chief_dir.exists() else 0
    parse_error_count = 0
    if chief_dir.exists():
        for path in chief_dir.glob("*_chief_parse_error.txt"):
            if _read_text_if_exists(path).strip():
                parse_error_count += 1
    report_info = _report_summary_from_sources(report, case_result)
    case_id = str(case_result.get("case_id") or case_dir.name)
    ground_truth = _ground_truth_label_for_case(case_id)
    comparison = _compare_final_report_with_ground_truth(ground_truth, report, case_result)
    return {
        "prediction": prediction if isinstance(prediction, dict) else {},
        "observation_count": observation_count,
        "chief_count": chief_count,
        "parse_error_count": parse_error_count,
        "report_ready": report_info["report_ready"],
        "stop_reason": str(reasoning.get("stop_reason", "") if isinstance(reasoning, dict) else ""),
        "ground_truth_label": ground_truth,
        "final_report_comparison": comparison,
    }


def _batch_index_html(run_dir, exported_cases):
    run_dir = Path(run_dir)
    cards = []
    for item in exported_cases:
        case_id = item["case_id"]
        status_class = "ok" if str(item["status"]).lower() in {"ok", "success", "completed", "complete"} else "warn"
        summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
        prediction = summary.get("prediction", {}) if isinstance(summary.get("prediction"), dict) else {}
        prediction_text = ", ".join("{0}={1}".format(key, value) for key, value in prediction.items()) or "no prediction"
        report_ready = "report_ready" if summary.get("report_ready") else "report_missing"
        ground_truth = summary.get("ground_truth_label", {}) if isinstance(summary.get("ground_truth_label"), dict) else {}
        comparison = summary.get("final_report_comparison", {}) if isinstance(summary.get("final_report_comparison"), dict) else {}
        truth_text = "Label: {0}".format(ground_truth.get("summary", "missing"))
        label_status = "Label {0}".format(comparison.get("status_display", "缺少真实标签"))
        cards.append(
            """
      <a class="case-card {status_class}" href="{href}/">
        <div class="case-top">
          <span class="case-id">{case_id}</span>
          <span class="status">{status}</span>
        </div>
        <div class="case-detail">{prediction_text}</div>
        <div class="case-truth">{truth_text}</div>
        <div class="case-metrics">
          <span>Obs {observation_count}</span>
          <span>Chief {chief_count}</span>
          <span>Parse {parse_error_count}</span>
          <span>{report_ready}</span>
          <span>{label_status}</span>
        </div>
      </a>
            """.format(
                status_class=escape(status_class),
                href=escape(case_id),
                case_id=escape(case_id),
                status=escape(str(item["status"])),
                prediction_text=escape(prediction_text),
                observation_count=escape(str(summary.get("observation_count", 0))),
                chief_count=escape(str(summary.get("chief_count", 0))),
                parse_error_count=escape(str(summary.get("parse_error_count", 0))),
                report_ready=escape(report_ready),
                truth_text=escape(truth_text),
                label_status=escape(label_status),
            )
        )
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent 实验批量看板</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18201d;
      --muted: #64706b;
      --line: #d9e0dc;
      --paper: #f7f8f5;
      --panel: #ffffff;
      --accent: #256f6a;
      --warn: #a84b37;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(37, 111, 106, 0.10), transparent 34%),
        linear-gradient(215deg, rgba(168, 75, 55, 0.08), transparent 30%),
        var(--paper);
    }}
    header {{
      padding: 28px 34px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      backdrop-filter: blur(10px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .source {{
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    main {{
      padding: 24px 34px 40px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
    }}
    .case-card {{
      display: block;
      min-height: 118px;
      padding: 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--accent);
      border-radius: 8px;
      background: var(--panel);
      color: inherit;
      text-decoration: none;
      box-shadow: 0 8px 22px rgba(24, 32, 29, 0.06);
      transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    }}
    .case-card:hover {{
      transform: translateY(-2px);
      border-color: rgba(37, 111, 106, 0.42);
      box-shadow: 0 12px 28px rgba(24, 32, 29, 0.10);
    }}
    .case-card.warn {{ border-left-color: var(--warn); }}
    .case-top {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .case-id {{
      font-weight: 800;
      font-size: 17px;
      overflow-wrap: anywhere;
    }}
    .status {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      color: var(--muted);
      background: #fbfcfa;
      white-space: nowrap;
    }}
    .case-detail {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      overflow-wrap: anywhere;
    }}
    .case-truth {{
      margin-top: 8px;
      color: #3f2b27;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .case-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin-top: 12px;
    }}
    .case-metrics span {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      background: #fbfcfa;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Agent 实验批量看板</h1>
    <div class="source">当前数据源：{source}</div>
  </header>
  <main>
    <div class="grid">
{cards}
    </div>
  </main>
</body>
</html>
""".format(source=escape(str(run_dir)), cards="".join(cards))


def export_dashboard_batch_from_harness_run(run_dir, output_dir):
    run_dir = Path(run_dir).resolve()
    output_dir = ensure_dir(output_dir)
    case_dirs = _case_dirs_in_harness_run(run_dir)
    if not case_dirs:
        raise ValueError("No case output directories found under: {0}".format(run_dir))

    exported_cases = []
    for case_dir in case_dirs:
        case_output_dir = output_dir / case_dir.name
        status, detail = _read_case_result_status(case_dir)
        summary = _read_batch_case_summary(case_dir)
        export_status = "exported"
        export_error = ""
        try:
            export_dashboard(build_dashboard_payload_from_harness_case(case_dir), case_output_dir, source_root=case_dir)
        except FileNotFoundError as exc:
            export_status = "skipped_missing_artifact"
            export_error = str(exc)
        exported_cases.append(
            {
                "case_id": case_dir.name,
                "status": status,
                "detail": detail,
                "summary": summary,
                "export_status": export_status,
                "export_error": export_error,
            }
        )

    write_json(
        output_dir / "dashboard_batch_manifest.json",
        {
            "source_run_dir": str(run_dir),
            "case_count": len(exported_cases),
            "cases": exported_cases,
        },
    )
    write_text(output_dir / "index.html", _batch_index_html(run_dir, exported_cases))
    return output_dir
