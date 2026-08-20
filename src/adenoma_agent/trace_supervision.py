import json
import re
import shutil
from html import escape
from pathlib import Path

from adenoma_agent.utils import ensure_dir, read_json, write_json


TRACE_LABEL_RUBRIC = {
    "epithelial_neoplasia_suspicious": {
        "default_priority": 4,
        "default_high_mag": True,
        "positive_cues": [
            "epithelial",
            "neoplasia",
            "neoplastic",
            "tum",
            "adenoma",
            "serrated",
            "lesion",
            "reviewable atypical epithelium",
        ],
        "observation_points": [
            "resolve serrated versus conventional morphology",
            "assess epithelial architecture at higher magnification",
            "keep serrated and conventional branches open",
        ],
    },
    "mucus_rich_or_pale_context": {
        "default_priority": 4,
        "default_high_mag": True,
        "positive_cues": [
            "mucus",
            "mucin",
            "pale",
            "mucus-rich",
            "mucosa requiring morphology resolution",
        ],
        "observation_points": [
            "determine whether mucus-rich mucosa reflects serrated architecture",
            "check epithelial architecture before branch assignment",
        ],
    },
    "uncertain_reviewable_mucosa": {
        "default_priority": 3,
        "default_high_mag": True,
        "positive_cues": [
            "uncertain",
            "reviewable",
            "low confidence",
            "ambiguous mucosa",
        ],
        "observation_points": [
            "resolve ambiguous reviewable mucosa",
            "avoid early serrated versus conventional branch lock",
        ],
    },
    "inflammatory_or_stromal_context": {
        "default_priority": 2,
        "default_high_mag": False,
        "positive_cues": [
            "inflammatory",
            "lymphocyte",
            "stroma",
            "reactive",
            "stromal",
        ],
        "observation_points": [
            "confirm inflammatory or stromal context",
            "exclude hidden epithelial lesion if sampled",
        ],
    },
    "reviewable_normal_mucosa": {
        "default_priority": 1,
        "default_high_mag": False,
        "positive_cues": [
            "normal",
            "reviewable normal",
            "benign mucosa",
            "norm",
        ],
        "observation_points": [
            "confirm benign architecture if sampled",
            "low-priority reviewable mucosa",
        ],
    },
    "background_or_artifact": {
        "default_priority": 0,
        "default_high_mag": False,
        "positive_cues": [
            "background",
            "artifact",
            "adipose",
            "debris",
            "muscle",
            "discard",
        ],
        "observation_points": [
            "coverage-preserving discard group",
            "low-value background or artifact",
        ],
    },
    "serrated": {
        "default_priority": 4,
        "default_high_mag": True,
        "positive_cues": [
            "serrated",
            "ssl",
            "sessile serrated",
            "mucus",
            "mucous",
            "pale",
            "flat",
            "broad",
            "crypt branching",
            "basal dilatation",
            "horizontal growth",
            "serration to base",
            "abnormal maturation",
            "lesion edge",
        ],
        "observation_points": [
            "basal crypt dilatation",
            "crypt branching",
            "horizontal crypt growth",
            "serration to crypt base",
            "mucus cap",
            "abnormal maturation",
        ],
    },
    "conventional": {
        "default_priority": 3,
        "default_high_mag": True,
        "positive_cues": [
            "conventional",
            "adenoma",
            "adenomatous",
            "tubular",
            "tubulovillous",
            "villous",
            "crowded gland",
            "gland crowding",
            "dysplasia",
            "hyperchromatic",
        ],
        "observation_points": [
            "tubular or tubulovillous architecture",
            "adenomatous gland crowding",
            "conventional dysplasia branch review",
        ],
    },
    "normal": {
        "default_priority": 1,
        "default_high_mag": False,
        "positive_cues": [
            "normal",
            "benign",
            "non-lesional",
            "uniform crypt",
            "reviewable mucosa",
            "low-priority mucosa",
        ],
        "observation_points": [
            "confirm benign architecture if sampled",
            "low-priority non-lesional mucosa",
        ],
    },
    "background": {
        "default_priority": 0,
        "default_high_mag": False,
        "positive_cues": [
            "background",
            "artifact",
            "stroma",
            "muscle",
            "adipose",
            "black",
            "masked",
            "discard",
            "low-value",
            "low tissue",
        ],
        "observation_points": [
            "coverage-preserving discard group",
            "low-value background or artifact",
        ],
    },
}

FIXED_DIAGNOSTIC_PRIORITY = {
    "background_or_artifact": 0,
    "reviewable_normal_mucosa": 1,
    "inflammatory_or_stromal_context": 2,
    "uncertain_reviewable_mucosa": 3,
    "mucus_rich_or_pale_context": 4,
    "epithelial_neoplasia_suspicious": 4,
    "background": 0,
    "normal": 1,
    "conventional": 3,
    "serrated": 4,
}

LESION_TRACE_LABELS = {
    "serrated",
    "conventional",
    "epithelial_neoplasia_suspicious",
    "mucus_rich_or_pale_context",
    "uncertain_reviewable_mucosa",
}

TRACE_LABEL_COLORS = {
    "epithelial_neoplasia_suspicious": {"fill": "rgba(220, 38, 38, 0.26)", "stroke": "#dc2626", "text": "#7f1d1d"},
    "mucus_rich_or_pale_context": {"fill": "rgba(14, 165, 233, 0.24)", "stroke": "#0ea5e9", "text": "#075985"},
    "uncertain_reviewable_mucosa": {"fill": "rgba(168, 85, 247, 0.22)", "stroke": "#a855f7", "text": "#581c87"},
    "inflammatory_or_stromal_context": {"fill": "rgba(20, 184, 166, 0.22)", "stroke": "#14b8a6", "text": "#134e4a"},
    "reviewable_normal_mucosa": {"fill": "rgba(16, 185, 129, 0.24)", "stroke": "#10b981", "text": "#064e3b"},
    "background_or_artifact": {"fill": "rgba(107, 114, 128, 0.24)", "stroke": "#6b7280", "text": "#374151"},
    "serrated": {"fill": "rgba(220, 53, 69, 0.28)", "stroke": "#dc3545", "text": "#7f1d1d"},
    "conventional": {"fill": "rgba(245, 158, 11, 0.28)", "stroke": "#f59e0b", "text": "#78350f"},
    "normal": {"fill": "rgba(16, 185, 129, 0.24)", "stroke": "#10b981", "text": "#064e3b"},
    "background": {"fill": "rgba(107, 114, 128, 0.24)", "stroke": "#6b7280", "text": "#374151"},
    "unknown": {"fill": "rgba(148, 163, 184, 0.24)", "stroke": "#64748b", "text": "#334155"},
}

GLOBAL_SCREENING_SINGLE_MODEL_STATUS = "single_model_trace"
GLOBAL_SCREENING_CONCH_ONLY_STATUS = "conch_only_trace"
GLOBAL_SCREENING_DUAL_MODEL_STATUSES = {"strong_agreement", "risk_disagreement", "low_value_disagreement"}
GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN = "consensus_fusion"
GLOBAL_SCREENING_LEGACY_SCORE_ORIGIN = "normalized_diagnostic_priority"
GLOBAL_SCREENING_LABEL_RANK = {
    "background_or_artifact": 0,
    "reviewable_normal_mucosa": 1,
    "inflammatory_or_stromal_context": 2,
    "uncertain_reviewable_mucosa": 3,
    "mucus_rich_or_pale_context": 4,
    "epithelial_neoplasia_suspicious": 4,
    "background": 0,
    "normal": 1,
    "conventional": 3,
    "serrated": 4,
}
GLOBAL_SCREENING_LESION_LABELS = {"serrated", "conventional", "epithelial_neoplasia_suspicious", "mucus_rich_or_pale_context", "uncertain_reviewable_mucosa"}
GLOBAL_SCREENING_PLACEHOLDER_LABELS = {"", "not_available_in_this_run", "n/a", "na", "none"}


def _normalize_global_screening_label(label):
    label = str(label or "").strip()
    legacy_map = {
        "ssl_suspicious_mucosa": "serrated",
        "ssl_high_priority_mucosa": "serrated",
        "ssl_like_mucosa": "serrated",
        "hp_like_mucosa": "serrated",
        "tsa_like_mucosa": "serrated",
        "unclassified_serrated_like_mucosa": "serrated",
        "conventional_adenoma_like": "conventional",
        "tubular_adenoma_like": "conventional",
        "tubulovillous_adenoma_like": "conventional",
        "inflammatory_polyp_like": "normal",
        "normal_mucosa": "normal",
        "background_artifact_stroma": "background",
        "ADI": "background_or_artifact",
        "BACK": "background_or_artifact",
        "DEB": "background_or_artifact",
        "MUS": "background_or_artifact",
        "NORM": "reviewable_normal_mucosa",
        "LYM": "inflammatory_or_stromal_context",
        "STR": "inflammatory_or_stromal_context",
        "MUC": "mucus_rich_or_pale_context",
        "TUM": "epithelial_neoplasia_suspicious",
    }
    label = legacy_map.get(label, legacy_map.get(label.upper(), label))
    return label if label in TRACE_LABEL_RUBRIC else ""


def _is_global_screening_lesion_label(label):
    return str(label or "") in GLOBAL_SCREENING_LESION_LABELS


def _looks_like_dual_model_global_screening_patch(patch):
    if not isinstance(patch, dict):
        return False
    agreement_status = str(patch.get("agreement_status", "")).strip()
    score_origin = str(patch.get("score_origin", "")).strip()
    if agreement_status == GLOBAL_SCREENING_CONCH_ONLY_STATUS or score_origin == "conch_crc100k_neutral_mapping":
        return False
    conch_label = _normalize_global_screening_label(patch.get("conch_region_semantic"))
    patho_label = _normalize_global_screening_label(patch.get("pathoreasoner_r1_region_semantic"))
    if score_origin == GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN:
        return True
    if agreement_status in GLOBAL_SCREENING_DUAL_MODEL_STATUSES:
        return True
    if conch_label and patho_label:
        return True
    if conch_label or patho_label:
        return agreement_status != GLOBAL_SCREENING_SINGLE_MODEL_STATUS
    return False


def classify_global_screening_agreement(conch_label, patho_label):
    conch_label = _normalize_global_screening_label(conch_label)
    patho_label = _normalize_global_screening_label(patho_label)
    if not conch_label or not patho_label:
        return GLOBAL_SCREENING_SINGLE_MODEL_STATUS
    if conch_label == patho_label:
        return "strong_agreement"
    if _is_global_screening_lesion_label(conch_label) or _is_global_screening_lesion_label(patho_label):
        return "risk_disagreement"
    return "low_value_disagreement"


def fuse_global_screening_region_semantic(conch_label, patho_label):
    conch_label = _normalize_global_screening_label(conch_label)
    patho_label = _normalize_global_screening_label(patho_label)
    candidates = [label for label in (conch_label, patho_label) if label]
    if not candidates:
        return "background"
    return max(candidates, key=lambda label: GLOBAL_SCREENING_LABEL_RANK.get(label, -1))


def expected_global_screening_priority(region_semantic, agreement_status):
    region_semantic = _normalize_global_screening_label(region_semantic)
    agreement_status = str(agreement_status or "").strip()
    if region_semantic in {"epithelial_neoplasia_suspicious", "mucus_rich_or_pale_context"}:
        return 4
    if region_semantic == "uncertain_reviewable_mucosa":
        return 3
    if region_semantic == "inflammatory_or_stromal_context":
        return 2
    if region_semantic == "reviewable_normal_mucosa":
        return 1
    if region_semantic == "background_or_artifact":
        return 0
    if region_semantic == "serrated":
        return 4
    if region_semantic == "conventional":
        return 4 if agreement_status == "risk_disagreement" else 3
    if region_semantic == "normal":
        return 1
    return 0


def expected_global_screening_require_high_magnification(region_semantic, agreement_status):
    region_semantic = _normalize_global_screening_label(region_semantic)
    agreement_status = str(agreement_status or "").strip()
    if region_semantic in {"epithelial_neoplasia_suspicious", "mucus_rich_or_pale_context", "uncertain_reviewable_mucosa"}:
        return True
    if region_semantic in {"serrated", "conventional"}:
        return agreement_status in {"strong_agreement", "risk_disagreement"}
    return False


def build_global_screening_fusion_reasoning(conch_label, patho_label, fused_label, agreement_status, diagnostic_priority):
    conch_label = _normalize_global_screening_label(conch_label) or str(conch_label or "").strip() or "unknown"
    patho_label = _normalize_global_screening_label(patho_label) or str(patho_label or "").strip() or "unknown"
    fused_label = _normalize_global_screening_label(fused_label) or "background"
    agreement_status = str(agreement_status or "").strip() or GLOBAL_SCREENING_SINGLE_MODEL_STATUS
    if agreement_status == "strong_agreement":
        return "CONCH and PathoReasoner-R1 both selected {0}; consensus keeps {0} with priority {1}.".format(
            fused_label,
            int(diagnostic_priority),
        )
    if agreement_status == "risk_disagreement":
        return "CONCH={0} and PathoReasoner-R1={1} disagree on a lesion-relevant patch; conservative fusion keeps {2} and escalates priority to {3}.".format(
            conch_label,
            patho_label,
            fused_label,
            int(diagnostic_priority),
        )
    if agreement_status == "low_value_disagreement":
        return "CONCH={0} and PathoReasoner-R1={1} disagree only in low-value territory; conservative fusion keeps {2} at priority {3}.".format(
            conch_label,
            patho_label,
            fused_label,
            int(diagnostic_priority),
        )
    return "Single-model trace output was retained for {0} with priority {1}.".format(fused_label, int(diagnostic_priority))


def derive_global_screening_fusion(conch_label, patho_label):
    conch_label = _normalize_global_screening_label(conch_label)
    patho_label = _normalize_global_screening_label(patho_label)
    if not conch_label or not patho_label:
        return None
    agreement_status = classify_global_screening_agreement(conch_label, patho_label)
    fused_label = fuse_global_screening_region_semantic(conch_label, patho_label)
    diagnostic_priority = expected_global_screening_priority(fused_label, agreement_status)
    require_high_magnification = expected_global_screening_require_high_magnification(fused_label, agreement_status)
    return {
        "conch_region_semantic": conch_label,
        "pathoreasoner_r1_region_semantic": patho_label,
        "agreement_status": agreement_status,
        "region_semantic": fused_label,
        "diagnostic_priority": diagnostic_priority,
        "require_high_magnification": require_high_magnification,
        "score_origin": GLOBAL_SCREENING_CONSENSUS_SCORE_ORIGIN,
        "fusion_reasoning": build_global_screening_fusion_reasoning(
            conch_label,
            patho_label,
            fused_label,
            agreement_status,
            diagnostic_priority,
        ),
    }


def _lower_text(value):
    return str(value or "").strip().lower()


def extract_first_json_object(text):
    text = str(text or "")
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_patch_assignment_response(text):
    parsed = None
    parse_failure = False
    try:
        parsed = json.loads(text)
    except Exception:
        blob = extract_first_json_object(text)
        if blob:
            try:
                parsed = json.loads(blob)
            except Exception:
                parse_failure = True
        else:
            parse_failure = True
    if isinstance(parsed, dict) and isinstance(parsed.get("patches"), list):
        return {"payload": parsed, "parse_failure": False, "parse_error": None}
    return {
        "payload": {"patches": []},
        "parse_failure": True,
        "parse_error": "missing_patches_array" if not parse_failure else "invalid_json",
    }


def build_image_only_teacher_prompt(teacher_request):
    selected_patch_ids = teacher_request.get("selected_patch_ids", [])
    rubric = teacher_request.get("trace_rubric", TRACE_LABEL_RUBRIC)
    vocabulary = teacher_request.get("agent_vocabulary")
    grid_meta_summary = teacher_request.get("grid_metadata_summary", {})
    no_report_block = teacher_request.get(
        "no_report_prompt_block",
        (
            "No pathology report is available. This is image-only morphology supervision. "
            "Produce screening-level only trace labels; no final diagnosis."
        ),
    )
    selected_vocab = json.dumps(selected_patch_ids, ensure_ascii=False, separators=(",", ":"))
    rubric_json = json.dumps(rubric, ensure_ascii=False, indent=2)
    grid_meta_json = json.dumps(grid_meta_summary, ensure_ascii=False, indent=2)
    vocabulary_json = json.dumps(vocabulary, ensure_ascii=False, indent=2) if vocabulary else None
    lines = [
        "You are generating adenoma trace supervision from a grid thumbnail.",
        no_report_block,
        "Use cautious screening language: possible, suspicious, review-worthy, or check.",
        "The black background is intentional masking to make real tissue stand out; do not treat black masked areas as lesion evidence.",
        "",
        "Hard structure contract:",
        "- Return exactly one JSON object and no prose.",
        '- The primary schema is {"patches":[...]}',
        "- Every selected patch must appear exactly once.",
        "- No duplicate patch_id values.",
        "- No patch_id outside the exact allowed vocabulary.",
        "- Low-value tissue/background still receives one assignment, usually background.",
        "",
        "Exact allowed patch vocabulary:",
        selected_vocab,
        "",
        "Grid metadata summary:",
        grid_meta_json,
        "",
        "Adenoma trace rubric:",
        rubric_json,
    ]
    if vocabulary_json:
        lines.extend(
            [
                "",
                "Adenoma agent vocabulary:",
                vocabulary_json,
                "- Keep output text pathology-native and consistent with the vocabulary.",
                "- Prefer pathology workflow terminology over generic visual-caption phrasing.",
            ]
        )
    lines.extend(
        [
            "",
            "Semantic constraints:",
            "- Do not call a patch serrated only because it is at an edge or upper-left location.",
            "- Serrated routing requires visual morphology cues such as serrated architecture, mucus cap, pale flat mucosa, basal crypt abnormality, or crypt distortion.",
            "- Do not make final diagnosis claims. Describe image-only morphology and review priority.",
            "",
            "Required output fields per patch:",
            "- patch_id",
            "- region_semantic",
            "- name",
            "- description",
            "- require_high_magnification",
            "- severity_reasoning",
            "- diagnostic_priority",
            "- observation_points",
            "",
            "Minimal example:",
            json.dumps(
                {
                    "patches": [
                        {
                            "patch_id": [0, 0],
                            "region_semantic": "background",
                            "name": "Masked or low-value background",
                            "description": "Black masked or low-value area kept for coverage, not lesion evidence.",
                            "require_high_magnification": False,
                            "severity_reasoning": "Low diagnostic priority but must be assigned exactly once.",
                            "diagnostic_priority": 0,
                            "observation_points": ["coverage-preserving discard group"],
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Now return the complete JSON object for all selected patches.",
        ]
    )
    return "\n".join(lines)


def _normalize_patch_id(item):
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return None
    try:
        return (int(item[0]), int(item[1]))
    except Exception:
        return None


def _patch_map(payload):
    mapping = {}
    for patch in payload.get("patches", []) if isinstance(payload, dict) else []:
        if not isinstance(patch, dict):
            continue
        row_col = _normalize_patch_id(patch.get("patch_id"))
        if row_col is not None:
            mapping[row_col] = patch
    return mapping


def _grid_cell_lookup(grid_meta):
    lookup = {}
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict):
            continue
        row_col = _normalize_patch_id(cell.get("patch_id") or [cell.get("row_id"), cell.get("col_id")])
        if row_col is not None:
            lookup[row_col] = cell
    return lookup


def _grid_canvas_size(grid_meta):
    width, height = grid_meta.get("cropped_thumbnail_size", [0, 0])[:2]
    if width and height:
        return int(width), int(height)
    max_x = 0
    max_y = 0
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict):
            continue
        max_x = max(max_x, int(cell.get("thumbnail_top_left_x", 0)) + int(cell.get("thumbnail_width", 0)))
        max_y = max(max_y, int(cell.get("thumbnail_top_left_y", 0)) + int(cell.get("thumbnail_height", 0)))
    return max_x or 1, max_y or 1


def _trace_label_style(label):
    return TRACE_LABEL_COLORS.get(label, TRACE_LABEL_COLORS["unknown"])


def _human_label(label):
    return str(label or "unknown")


def _overlay_html(pane_id, image_name, grid_meta, payload, diff_patch_ids=None, selected_patch_id=None):
    width, height = _grid_canvas_size(grid_meta)
    patch_lookup = _patch_map(payload)
    cell_lookup = _grid_cell_lookup(grid_meta)
    cells_html = []
    for row_col in selected_patch_ids_from_grid(grid_meta):
        cell = cell_lookup.get(row_col)
        if not cell:
            continue
        patch = patch_lookup.get(row_col, {})
        label = str(patch.get("region_semantic", "unknown"))
        style = _trace_label_style(label)
        left = 100.0 * float(cell.get("thumbnail_top_left_x", 0)) / float(width)
        top = 100.0 * float(cell.get("thumbnail_top_left_y", 0)) / float(height)
        cell_width = 100.0 * float(cell.get("thumbnail_width", 0)) / float(width)
        cell_height = 100.0 * float(cell.get("thumbnail_height", 0)) / float(height)
        patch_id_text = "[{0},{1}]".format(int(row_col[0]), int(row_col[1]))
        title = "{0} | {1}".format(patch_id_text, _human_label(label))
        classes = ["patch-overlay"]
        if diff_patch_ids and row_col in diff_patch_ids:
            classes.append("is-diff")
        if selected_patch_id and row_col == selected_patch_id:
            classes.append("is-selected")
        cells_html.append(
            '<button type="button" class="{classes}" data-pane="{pane}" data-patch-id="{patch_id}" title="{title}" '
            'style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;'
            'background:{fill};border-color:{stroke};color:{text};">'
            '<span class="patch-chip">{patch_text}</span>'
            "</button>".format(
                classes=" ".join(classes),
                pane=escape(str(pane_id)),
                patch_id=escape("{0}_{1}".format(int(row_col[0]), int(row_col[1]))),
                title=escape(title),
                left=left,
                top=top,
                width=cell_width,
                height=cell_height,
                fill=style["fill"],
                stroke=style["stroke"],
                text=style["text"],
                patch_text=escape(patch_id_text),
            )
        )
    return (
        '<div class="review-pane">'
        '<div class="review-canvas">'
        '<img src="{image_name}" alt="{alt}" class="review-image" />'
        '<div class="overlay-layer">{cells}</div>'
        "</div>"
        "</div>"
    ).format(
        image_name=escape(image_name),
        alt=escape("grid thumbnail {0}".format(pane_id)),
        cells="".join(cells_html),
    )


def _candidate_patch_table(candidate_results, selected_index):
    candidate_payloads = [item.get("payload", {"patches": []}) for item in candidate_results]
    mappings = [_patch_map(payload) for payload in candidate_payloads]
    all_ids = sorted(set().union(*[set(mapping) for mapping in mappings])) if mappings else []
    rows = []
    for row_col in all_ids:
        row_cells = ['<td class="patch-col">{0}</td>'.format(escape("[{0},{1}]".format(*row_col)))]
        labels = []
        for index, mapping in enumerate(mappings):
            patch = mapping.get(row_col, {})
            label = str(patch.get("region_semantic", "missing"))
            labels.append(label)
            row_cells.append('<td>{0}</td>'.format(escape(label)))
        classes = []
        if len(set(labels)) > 1:
            classes.append("row-diff")
        if selected_index is not None and selected_index < len(mappings):
            selected_patch = mappings[selected_index].get(row_col, {})
            row_cells.append('<td>{0}</td>'.format(escape(str(selected_patch.get("name", "")))))
        rows.append('<tr class="{classes}" data-patch-id="{patch_id}">{cells}</tr>'.format(
            classes=" ".join(classes),
            patch_id=escape("{0}_{1}".format(*row_col)),
            cells="".join(row_cells),
        ))
    header = ["<th>Patch</th>"]
    for index in range(len(candidate_results)):
        header.append("<th>Candidate {0}</th>".format(index))
    header.append("<th>Selected name</th>")
    return '<table class="patch-table"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'.format(
        header="".join(header),
        rows="".join(rows),
    )


def _warning_list_html(items):
    if not items:
        return "<p class=\"muted\">None.</p>"
    return "<ul>{0}</ul>".format(
        "".join("<li>{0}</li>".format(escape(str(item))) for item in items)
    )


def _build_visual_review_html(case_id, image_name, grid_meta, auto_review, review_target):
    selection = auto_review.get("selection", {})
    candidates = auto_review.get("candidates", [])
    selected_index = selection.get("selected_index")
    agreement = auto_review.get("candidate_agreement", {})
    diff_patch_ids = {
        _normalize_patch_id(item.get("patch_id"))
        for item in agreement.get("disagreements", [])
        if _normalize_patch_id(item.get("patch_id")) is not None
    }
    selected_patch_id = None
    selected_candidate = auto_review.get("selected_candidate") or {}
    selected_score = selected_candidate.get("score", {})
    slide_warnings = [item.get("warning") for item in (selected_score.get("slide_label_consistency") or {}).get("warnings", [])]
    review_reason = selection.get("review_reason") or selection.get("reason", "")
    panes = []
    for index, candidate in enumerate(candidates[:2]):
        panes.append(
            '<section class="pane-block"><h3>Candidate {idx}</h3>{pane}<div class="pane-meta"><p><strong>Total:</strong> {score}</p><p><strong>Parse failure:</strong> {parse}</p></div></section>'.format(
                idx=index,
                pane=_overlay_html(
                    "candidate_{0}".format(index),
                    image_name,
                    grid_meta,
                    candidate.get("payload", {"patches": []}),
                    diff_patch_ids=diff_patch_ids,
                    selected_patch_id=selected_patch_id,
                ),
                score=escape(str((candidate.get("score") or {}).get("total_score"))),
                parse=escape(str(candidate.get("parse_failure"))),
            )
        )
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --surface: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #d1d5db;
      --accent: #111827;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    .header, .panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }}
    .header h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta-grid, .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .meta-item, .stat {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: #fafafa; }}
    .meta-label, .stat-label {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .section {{ margin-top: 18px; }}
    .section h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .panes {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .pane-block {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }}
    .pane-block h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .review-canvas {{ position: relative; width: 100%; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #0f172a; }}
    .review-image {{ display: block; width: 100%; height: auto; }}
    .overlay-layer {{ position: absolute; inset: 0; }}
    .patch-overlay {{ position: absolute; border: 2px solid; border-radius: 6px; padding: 0; cursor: pointer; display: flex; align-items: flex-start; justify-content: flex-start; }}
    .patch-overlay.is-diff {{ box-shadow: 0 0 0 2px rgba(185,28,28,0.3); }}
    .patch-overlay.active {{ outline: 3px solid var(--accent); z-index: 4; }}
    .patch-chip {{ display: inline-block; font-size: 11px; font-weight: 700; background: rgba(255,255,255,0.86); border-bottom-right-radius: 6px; padding: 2px 6px; }}
    .muted {{ color: var(--muted); }}
    .patch-table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
    .patch-table th, .patch-table td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    .patch-table th {{ background: #f3f4f6; font-size: 12px; text-transform: uppercase; color: var(--muted); }}
    .patch-table tr.row-diff {{ background: rgba(185, 28, 28, 0.06); }}
    .patch-table tr.active {{ outline: 2px solid var(--accent); }}
    .note-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .danger {{ color: var(--danger); font-weight: 700; }}
    ul {{ margin: 8px 0 0 18px; padding: 0; }}
    p {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="header">
      <h1>{title}</h1>
      <div class="meta-grid">
        <div class="meta-item"><span class="meta-label">Review status</span><strong>{review_status}</strong></div>
        <div class="meta-item"><span class="meta-label">Review reason</span><strong>{review_reason}</strong></div>
        <div class="meta-item"><span class="meta-label">Selected candidate</span><strong>{selected_index}</strong></div>
        <div class="meta-item"><span class="meta-label">Agreement</span><strong>{agreement}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>Candidates</h2>
      <div class="panes">{panes}</div>
    </section>
    <section class="section">
      <h2>Selected candidate summary</h2>
      <div class="stats-grid">
        <div class="stat"><span class="stat-label">Total score</span><strong>{total_score}</strong></div>
        <div class="stat"><span class="stat-label">Coverage OK</span><strong>{coverage_ok}</strong></div>
        <div class="stat"><span class="stat-label">Missing patches</span><strong>{missing_count}</strong></div>
        <div class="stat"><span class="stat-label">Slide-level label</span><strong>{slide_label}</strong></div>
      </div>
    </section>
    <section class="section">
      <h2>Patch comparison</h2>
      {patch_table}
    </section>
    <section class="section">
      <h2>Warnings</h2>
      <div class="note-grid">
        <div class="panel">
          <h3>Slide-label risks</h3>
          {slide_warnings}
        </div>
        <div class="panel">
          <h3>Review instructions</h3>
          {instructions}
        </div>
      </div>
    </section>
  </div>
  <script>
    const rows = document.querySelectorAll("[data-patch-id]");
    rows.forEach((row) => {{
      row.addEventListener("click", () => {{
        const patchId = row.getAttribute("data-patch-id");
        document.querySelectorAll(".active").forEach((node) => node.classList.remove("active"));
        document.querySelectorAll('[data-patch-id="' + patchId + '"]').forEach((node) => node.classList.add("active"));
      }});
    }});
  </script>
</body>
</html>
"""
    return html.format(
        title=escape("Global Screening Review: {0}".format(case_id)),
        review_status=escape(str(review_target.get("review_status", "needs_review"))),
        review_reason=escape(str(review_reason or "manual_review_requested")),
        selected_index=escape(str(selected_index)),
        agreement=escape(str(agreement.get("agreement_rate"))),
        panes="".join(panes),
        total_score=escape(str(selected_score.get("total_score"))),
        coverage_ok=escape(str((selected_score.get("structure") or {}).get("coverage_ok"))),
        missing_count=escape(str(len((selected_score.get("structure") or {}).get("missing_patch_ids") or []))),
        slide_label=escape(str((selected_score.get("slide_label_consistency") or {}).get("slide_label"))),
        patch_table=_candidate_patch_table(candidates[:2], selected_index),
        slide_warnings=_warning_list_html(slide_warnings),
        instructions=_warning_list_html(review_target.get("instructions", [])),
    )


def selected_patch_ids_from_grid(grid_meta):
    ids = []
    for cell in grid_meta.get("grid_cells", []):
        if not isinstance(cell, dict) or not cell.get("is_selected"):
            continue
        row_col = (int(cell["row_id"]), int(cell["col_id"]))
        if row_col not in ids:
            ids.append(row_col)
    return ids


def validate_patch_assignments(payload, grid_meta):
    patches = payload.get("patches", []) if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        patches = []
    selected = selected_patch_ids_from_grid(grid_meta)
    selected_lookup = set(selected)
    covered = set()
    missing = []
    duplicate = []
    unexpected = []
    ignored = []
    records = []
    for index, patch in enumerate(patches):
        record = {
            "assignment_index": index,
            "patch": patch if isinstance(patch, dict) else {},
            "normalized_id": None,
            "issues": [],
        }
        if not isinstance(patch, dict):
            record["issues"].append("assignment_not_object")
            ignored.append(patch)
            records.append(record)
            continue
        row_col = _normalize_patch_id(patch.get("patch_id"))
        if row_col is None:
            record["issues"].append("invalid_patch_id")
            ignored.append(patch.get("patch_id"))
            records.append(record)
            continue
        if row_col not in selected_lookup:
            record["issues"].append("unexpected_patch_id")
            if list(row_col) not in unexpected:
                unexpected.append(list(row_col))
            records.append(record)
            continue
        if row_col in covered:
            record["issues"].append("duplicate_patch_id")
            if list(row_col) not in duplicate:
                duplicate.append(list(row_col))
            records.append(record)
            continue
        covered.add(row_col)
        record["normalized_id"] = row_col
        records.append(record)
    for row_col in selected:
        if row_col not in covered:
            missing.append([int(row_col[0]), int(row_col[1])])
    return {
        "coverage_ok": not (missing or duplicate or unexpected or ignored),
        "missing_patch_ids": missing,
        "duplicate_patch_ids": duplicate,
        "unexpected_patch_ids": unexpected,
        "ignored_patch_ids": ignored,
        "selected_patch_count": len(selected),
        "covered_patch_count": len(covered),
        "assignment_count": len(patches),
        "records": records,
    }


def _haystack_for_patch(patch):
    values = [
        patch.get("name", ""),
        patch.get("description", ""),
        patch.get("severity_reasoning", ""),
    ]
    obs = patch.get("observation_points", [])
    if isinstance(obs, list):
        values.extend(obs)
    else:
        values.append(obs)
    return " ".join(str(value).lower() for value in values)


def _label_known(label):
    return label in TRACE_LABEL_RUBRIC


def _normalize_slide_label_context(slide_label_context):
    slide_label_context = slide_label_context or {}
    return {
        "label": slide_label_context.get("label"),
        "serrated_target": slide_label_context.get("serrated_target"),
        "abnormal_crypt_target": slide_label_context.get("abnormal_crypt_target"),
        "dysplasia_proxy_target": slide_label_context.get("dysplasia_proxy_target"),
        "metadata": slide_label_context.get("metadata", {}) if isinstance(slide_label_context.get("metadata", {}), dict) else {},
    }


def score_patch_field_consistency(payload):
    patches = payload.get("patches", []) if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        patches = []
    warnings = []
    per_patch = []
    score = 100
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            continue
        label = str(patch.get("region_semantic", ""))
        patch_warnings = []
        dual_model_mode = _looks_like_dual_model_global_screening_patch(patch)
        priority_value = patch.get("diagnostic_priority", 0)
        try:
            priority = int(priority_value)
        except Exception:
            priority = None
            patch_warnings.append("invalid_priority")
        high_mag = bool(patch.get("require_high_magnification", False))
        if dual_model_mode:
            conch_label = _normalize_global_screening_label(patch.get("conch_region_semantic"))
            patho_label = _normalize_global_screening_label(patch.get("pathoreasoner_r1_region_semantic"))
            if not conch_label or not patho_label:
                patch_warnings.append("missing_dual_model_fields")
            else:
                expected = derive_global_screening_fusion(conch_label, patho_label)
                if expected is None:
                    patch_warnings.append("missing_dual_model_fields")
                else:
                    actual_agreement = str(patch.get("agreement_status", "")).strip()
                    if actual_agreement != expected["agreement_status"]:
                        patch_warnings.append("invalid_agreement_status")
                    if not str(patch.get("fusion_reasoning", "")).strip():
                        patch_warnings.append("missing_fusion_reasoning")
                    if label != expected["region_semantic"]:
                        patch_warnings.append("invalid_fused_region_semantic")
                    if priority is not None and priority != expected["diagnostic_priority"]:
                        patch_warnings.append("invalid_priority")
                    if high_mag != expected["require_high_magnification"]:
                        patch_warnings.append("invalid_highmag")
                    actual_score_origin = str(patch.get("score_origin", "")).strip()
                    if actual_score_origin != expected["score_origin"]:
                        patch_warnings.append("invalid_score_origin")
        else:
            expected_priority = FIXED_DIAGNOSTIC_PRIORITY.get(label)
            if priority is not None and expected_priority is not None and priority != expected_priority:
                patch_warnings.append("invalid_priority")
            if label == "background" and ((priority is not None and priority != 0) or high_mag):
                patch_warnings.append("background_priority_highmag_conflict")
            if label == "serrated" and ((priority is not None and priority != 4) or not high_mag):
                patch_warnings.append("serrated_priority_highmag_conflict")
            if label == "normal" and priority is not None and priority != 1:
                patch_warnings.append("normal_priority_too_high")
        score -= 4 * len(patch_warnings)
        warnings.extend(
            {
                "assignment_index": index,
                "patch_id": patch.get("patch_id"),
                "warning": warning,
                "region_semantic": label,
            }
            for warning in patch_warnings
        )
        per_patch.append({"assignment_index": index, "patch_id": patch.get("patch_id"), "warnings": patch_warnings})
    return {
        "field_consistency_score": max(0, min(100, score)),
        "warnings": warnings,
        "per_patch": per_patch,
    }


def score_patch_semantics(payload):
    patches = payload.get("patches", []) if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        patches = []
    warnings = []
    per_patch = []
    label_counts = {}
    score = 100
    for index, patch in enumerate(patches):
        if not isinstance(patch, dict):
            continue
        label = str(patch.get("region_semantic", ""))
        label_counts[label] = label_counts.get(label, 0) + 1
        patch_warnings = []
        dual_model_mode = _looks_like_dual_model_global_screening_patch(patch)
        if label not in TRACE_LABEL_RUBRIC:
            patch_warnings.append("unknown_region_semantic")
        rubric = TRACE_LABEL_RUBRIC.get(label, {})
        haystack = _haystack_for_patch(patch)
        cues = rubric.get("positive_cues", [])
        if cues and not any(cue in haystack for cue in cues):
            patch_warnings.append("label_evidence_weak")
        observation_points = patch.get("observation_points", [])
        if not isinstance(observation_points, list) or not [item for item in observation_points if str(item).strip()]:
            patch_warnings.append("empty_observation_points")
        if dual_model_mode:
            conch_label = _normalize_global_screening_label(patch.get("conch_region_semantic"))
            patho_label = _normalize_global_screening_label(patch.get("pathoreasoner_r1_region_semantic"))
            if not conch_label or not patho_label:
                patch_warnings.append("missing_dual_model_fields")
            else:
                expected = derive_global_screening_fusion(conch_label, patho_label)
                if expected is None:
                    patch_warnings.append("missing_dual_model_fields")
                else:
                    if str(patch.get("agreement_status", "")).strip() != expected["agreement_status"]:
                        patch_warnings.append("invalid_agreement_status")
                    if not str(patch.get("fusion_reasoning", "")).strip():
                        patch_warnings.append("missing_fusion_reasoning")
                    if label != expected["region_semantic"]:
                        patch_warnings.append("invalid_fused_region_semantic")
        priority = patch.get("diagnostic_priority", 0)
        try:
            priority = int(priority)
        except Exception:
            priority = 0
            patch_warnings.append("invalid_priority")
        high_mag = bool(patch.get("require_high_magnification", False))
        if label == "background" and (priority != 0 or high_mag):
            patch_warnings.append("background_priority_highmag_conflict")
        if label == "serrated" and (priority < 3 or not high_mag):
            patch_warnings.append("serrated_priority_highmag_conflict")
        if label == "normal" and priority > 2:
            patch_warnings.append("normal_priority_too_high")
        if label == "serrated" and "edge" in haystack and not any(
            cue in haystack for cue in ("serrated", "ssl", "mucus", "mucous", "crypt", "pale")
        ):
            patch_warnings.append("serrated_edge_only_risk")
        score -= 4 * len(patch_warnings)
        warnings.extend(
            {
                "assignment_index": index,
                "patch_id": patch.get("patch_id"),
                "warning": warning,
                "region_semantic": label,
            }
            for warning in patch_warnings
        )
        per_patch.append({"assignment_index": index, "patch_id": patch.get("patch_id"), "warnings": patch_warnings})
    return {
        "semantic_score": max(0, min(100, score)),
        "warnings": warnings,
        "per_patch": per_patch,
        "label_counts": label_counts,
    }


def connected_components_for_patch_ids(patch_ids, diagonal=False):
    remaining = set((int(row), int(col)) for row, col in patch_ids)
    components = []
    if diagonal:
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            row, col = stack.pop()
            for dr, dc in deltas:
                nxt = (row + dr, col + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    component.append(nxt)
        components.append(sorted(component))
    components.sort(key=lambda comp: (comp[0][0], comp[0][1], len(comp)))
    return components


def score_cluster_granularity(clusters):
    warnings = []
    over_merge_count = 0
    for cluster in clusters:
        label = cluster.get("l") or cluster.get("metadata", {}).get("region_semantic")
        patch_ids = [_normalize_patch_id(item) for item in cluster.get("patch_ids_ordered", [])]
        patch_ids = [item for item in patch_ids if item is not None]
        if label in LESION_TRACE_LABELS and len(patch_ids) > 1:
            components = connected_components_for_patch_ids(patch_ids)
            if len(components) > 1:
                over_merge_count += 1
                warnings.append(
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "warning": "lesion_cluster_has_multiple_spatial_components",
                        "component_count": len(components),
                        "components": [[list(item) for item in comp] for comp in components],
                    }
                )
    score = max(0, 100 - 10 * over_merge_count)
    return {"aggregation_score": score, "warnings": warnings, "over_merge_count": over_merge_count}


def score_trace_case(teacher_payload, grid_meta, clusters=None, pathreasoner_payload=None, conch_payload=None):
    structure = validate_patch_assignments(teacher_payload, grid_meta)
    semantics = score_patch_semantics(teacher_payload)
    field_consistency = score_patch_field_consistency(teacher_payload)
    aggregation = score_cluster_granularity(clusters or [])
    warnings = []
    warnings.extend(semantics["warnings"])
    warnings.extend(field_consistency["warnings"])
    warnings.extend(aggregation["warnings"])
    if not structure["coverage_ok"]:
        warnings.append({"warning": "coverage_failure", "details": structure})
    disagreement = None
    dual_source_disagreement = None
    if conch_payload and pathreasoner_payload:
        dual_source_disagreement = compare_patch_labels(conch_payload, pathreasoner_payload)
        if dual_source_disagreement["disagreement_rate"] > 0.25:
            warnings.append({"warning": "conch_pathreasoner_high_disagreement", "details": dual_source_disagreement})
    if pathreasoner_payload:
        disagreement = compare_patch_labels(teacher_payload, pathreasoner_payload)
        if disagreement["disagreement_rate"] > 0.25:
            warnings.append({"warning": "teacher_pathreasoner_high_disagreement", "details": disagreement})
    total = int(
        round(
            0.35 * (100 if structure["coverage_ok"] else 0)
            + 0.30 * semantics["semantic_score"]
            + 0.15 * field_consistency["field_consistency_score"]
            + 0.20 * aggregation["aggregation_score"]
        )
    )
    return {
        "total_score": total,
        "structure": structure,
        "semantics": semantics,
        "field_consistency": field_consistency,
        "aggregation": aggregation,
        "teacher_pathreasoner_disagreement": disagreement,
        "conch_pathreasoner_disagreement": dual_source_disagreement,
        "review_recommended": total < 85
        or any(item.get("region_semantic") == "serrated" for item in warnings)
        or any(item.get("warning") in {"conch_pathreasoner_high_disagreement", "teacher_pathreasoner_high_disagreement"} for item in warnings),
        "warnings": warnings,
    }


def candidate_label_agreement(candidate_payloads):
    if len(candidate_payloads) < 2:
        return {"candidate_count": len(candidate_payloads), "agreement_rate": 1.0, "disagreements": []}

    def label_map(payload):
        mapping = {}
        for patch in payload.get("patches", []) if isinstance(payload, dict) else []:
            if not isinstance(patch, dict):
                continue
            row_col = _normalize_patch_id(patch.get("patch_id"))
            if row_col is not None:
                mapping[row_col] = str(patch.get("region_semantic", ""))
        return mapping

    maps = [label_map(payload) for payload in candidate_payloads]
    all_ids = sorted(set().union(*[set(item) for item in maps]))
    disagreements = []
    agree_count = 0
    for row_col in all_ids:
        labels = [mapping.get(row_col) for mapping in maps]
        if len(set(labels)) == 1:
            agree_count += 1
        else:
            disagreements.append({"patch_id": list(row_col), "labels": labels})
    return {
        "candidate_count": len(candidate_payloads),
        "compared_patch_count": len(all_ids),
        "agreement_rate": float(agree_count) / float(max(1, len(all_ids))),
        "disagreements": disagreements,
    }


def score_candidate_agreement(candidate_payloads):
    return candidate_label_agreement(candidate_payloads)


def score_slide_label_consistency(payload, slide_label_context):
    context = _normalize_slide_label_context(slide_label_context)
    patches = payload.get("patches", []) if isinstance(payload, dict) else []
    if not isinstance(patches, list):
        patches = []
    labels = [
        str(patch.get("region_semantic", ""))
        for patch in patches
        if isinstance(patch, dict) and str(patch.get("region_semantic", ""))
    ]
    normalized_labels = [_normalize_global_screening_label(label) for label in labels]
    has_ssl = "serrated" in normalized_labels
    has_conventional = "conventional" in normalized_labels
    lesion_labels = [label for label in labels if label in LESION_TRACE_LABELS]
    warnings = []
    raw_label = _lower_text(context.get("label"))
    serrated_target = context.get("serrated_target")
    if serrated_target == 1 and not has_ssl:
        warnings.append(
            {
                "warning": "ssl_absent_but_slide_positive_risk",
                "slide_label": context.get("label"),
            }
        )
    if "adenoma" in raw_label and not (has_ssl or has_conventional):
        warnings.append(
            {
                "warning": "lesion_absent_but_slide_positive_risk",
                "slide_label": context.get("label"),
            }
        )
    if "conventional" in raw_label or "tubular" in raw_label or "tubulovillous" in raw_label or "villous" in raw_label:
        if not has_conventional:
            warnings.append(
                {
                    "warning": "conventional_absent_but_slide_positive_risk",
                    "slide_label": context.get("label"),
                }
            )
    if "hyperplastic" in raw_label and lesion_labels and not has_ssl:
        warnings.append(
            {
                "warning": "slide_label_direction_mismatch",
                "slide_label": context.get("label"),
                "predicted_labels": sorted(set(lesion_labels)),
            }
        )
    score = max(0, 100 - 15 * len(warnings))
    return {
        "slide_label": context.get("label"),
        "serrated_target": serrated_target,
        "warnings": warnings,
        "slide_label_score": score,
    }


def score_trace_auto_review(candidate_results, grid_meta, slide_label_context=None, min_auto_pass_score=88, min_agreement=0.85):
    candidate_payloads = [item.get("payload", {"patches": []}) for item in candidate_results]
    agreement = score_candidate_agreement(candidate_payloads)
    candidate_details = []
    for item in candidate_results:
        payload = item.get("payload", {"patches": []})
        base_score = item.get("score") or score_trace_case(payload, grid_meta)
        slide_label_consistency = score_slide_label_consistency(payload, slide_label_context)
        warnings = list(base_score.get("warnings", [])) + list(slide_label_consistency.get("warnings", []))
        candidate_details.append(
            {
                **item,
                "score": {
                    **base_score,
                    "slide_label_consistency": slide_label_consistency,
                    "warnings": warnings,
                },
            }
        )
    selection = select_best_candidate(candidate_details, min_auto_pass_score=min_auto_pass_score, min_agreement=min_agreement)
    selected_index = selection.get("selected_index")
    selected = candidate_details[int(selected_index)] if selected_index is not None and 0 <= int(selected_index) < len(candidate_details) else None
    selected_score = (selected or {}).get("score", {})
    reasons = []
    if selected and not selected_score.get("structure", {}).get("coverage_ok", False):
        reasons.append("selected_coverage_fail")
    if selected and selected.get("parse_failure"):
        reasons.append("selected_parse_failure")
    if agreement.get("agreement_rate", 1.0) < min_agreement:
        reasons.append("candidate_disagreement")
    if any((item.get("parse_failure") for item in candidate_details)):
        reasons.append("candidate_parse_failure")
    slide_warnings = (selected_score.get("slide_label_consistency") or {}).get("warnings", [])
    if slide_warnings:
        reasons.append("slide_label_consistency_risk")
    if selected_score.get("review_recommended"):
        reasons.append("score_review_recommended")
    review_status = selection.get("review_status", "needs_review")
    if reasons:
        review_status = "needs_review"
    return {
        "candidates": candidate_details,
        "candidate_agreement": agreement,
        "selection": {
            **selection,
            "review_status": review_status,
            "review_reason": ",".join(sorted(set(reasons))) if reasons else selection.get("reason", ""),
        },
        "selected_candidate": selected,
    }


def build_candidate_diff_view(candidate_payloads):
    def patch_map(payload):
        mapping = {}
        for patch in payload.get("patches", []) if isinstance(payload, dict) else []:
            if not isinstance(patch, dict):
                continue
            row_col = _normalize_patch_id(patch.get("patch_id"))
            if row_col is not None:
                mapping[row_col] = patch
        return mapping

    maps = [patch_map(payload) for payload in candidate_payloads]
    all_ids = sorted(set().union(*[set(mapping) for mapping in maps])) if maps else []
    rows = []
    for row_col in all_ids:
        labels = []
        entries = []
        for mapping in maps:
            patch = mapping.get(row_col, {})
            labels.append(patch.get("region_semantic"))
            entries.append(
                {
                    "region_semantic": patch.get("region_semantic"),
                    "name": patch.get("name"),
                    "diagnostic_priority": patch.get("diagnostic_priority"),
                    "require_high_magnification": patch.get("require_high_magnification"),
                    "agreement_status": patch.get("agreement_status"),
                    "conch_region_semantic": patch.get("conch_region_semantic"),
                    "pathoreasoner_r1_region_semantic": patch.get("pathoreasoner_r1_region_semantic"),
                    "fusion_reasoning": patch.get("fusion_reasoning"),
                    "score_origin": patch.get("score_origin"),
                    "description": patch.get("description"),
                    "observation_points": patch.get("observation_points", []),
                }
            )
        if len(set(labels)) > 1:
            rows.append({"patch_id": list(row_col), "candidate_entries": entries})
    return {"rows": rows, "disagreement_count": len(rows)}


def select_best_candidate(candidate_results, min_auto_pass_score=88, min_agreement=0.85):
    if not candidate_results:
        return {
            "selected_index": None,
            "review_status": "needs_review",
            "reason": "no_candidates",
            "agreement": candidate_label_agreement([]),
        }
    payloads = [item.get("payload", {"patches": []}) for item in candidate_results]
    agreement = candidate_label_agreement(payloads)
    ranked = sorted(
        enumerate(candidate_results),
        key=lambda item: (
            int(item[1].get("score", {}).get("total_score", 0)),
            1 if item[1].get("score", {}).get("structure", {}).get("coverage_ok") else 0,
            -int(item[0]),
        ),
        reverse=True,
    )
    selected_index, selected = ranked[0]
    selected_score = int(selected.get("score", {}).get("total_score", 0))
    selected_payload = selected.get("payload", {"patches": []})
    has_ssl = any(
        isinstance(patch, dict) and _normalize_global_screening_label(patch.get("region_semantic")) == "serrated"
        for patch in selected_payload.get("patches", [])
    )
    review_status = "auto_pass"
    reasons = []
    if selected_score < min_auto_pass_score:
        review_status = "needs_review"
        reasons.append("low_auto_score")
    if agreement.get("agreement_rate", 1.0) < min_agreement:
        review_status = "needs_review"
        reasons.append("candidate_disagreement")
    if has_ssl:
        review_status = "needs_review"
        reasons.append("ssl_candidate")
    if selected.get("parse_failure"):
        review_status = "needs_review"
        reasons.append("parse_failure")
    return {
        "selected_index": int(selected_index),
        "review_status": review_status,
        "reason": ",".join(reasons) if reasons else "high_confidence_auto_pass",
        "agreement": agreement,
        "selected_score": selected_score,
    }


def compare_patch_labels(a_payload, b_payload):
    def label_map(payload):
        mapping = {}
        for patch in payload.get("patches", []) if isinstance(payload, dict) else []:
            if not isinstance(patch, dict):
                continue
            row_col = _normalize_patch_id(patch.get("patch_id"))
            if row_col is not None:
                mapping[row_col] = patch.get("region_semantic")
        return mapping

    a = label_map(a_payload)
    b = label_map(b_payload)
    ids = sorted(set(a) | set(b))
    disagreements = []
    for row_col in ids:
        if a.get(row_col) != b.get(row_col):
            disagreements.append({"patch_id": list(row_col), "teacher": a.get(row_col), "pathreasoner": b.get(row_col)})
    return {
        "compared_patch_count": len(ids),
        "disagreement_count": len(disagreements),
        "disagreement_rate": float(len(disagreements)) / float(max(1, len(ids))),
        "disagreements": disagreements,
    }


def build_supervision_record(case_id, grid_thumbnail_path, grid_metadata_path, target_payload, source=None, target_groups=None, score=None):
    grid_meta = read_json(grid_metadata_path)
    return {
        "case_id": case_id,
        "input": {
            "grid_thumbnail_path": str(grid_thumbnail_path),
            "grid_metadata_path": str(grid_metadata_path),
            "selected_patch_ids": [list(item) for item in selected_patch_ids_from_grid(grid_meta)],
            "task_prompt": "Assign every selected patch exactly once using adenoma trace patch assignments.",
        },
        "target": target_payload,
        "target_groups": target_groups or [],
        "provenance": source or {},
        "auto_score": score or score_trace_case(target_payload, grid_meta),
    }


def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def export_review_package(
    case_id,
    grid_thumbnail_path,
    grid_metadata_path,
    teacher_payload,
    output_dir,
    pathreasoner_payload=None,
    conch_payload=None,
    score=None,
):
    output_dir = ensure_dir(output_dir)
    grid_thumbnail_path = Path(grid_thumbnail_path)
    grid_metadata_path = Path(grid_metadata_path)
    copied_image = output_dir / grid_thumbnail_path.name
    copied_meta = output_dir / grid_metadata_path.name
    if copied_image.resolve() != grid_thumbnail_path.resolve():
        shutil.copy2(str(grid_thumbnail_path), str(copied_image))
    if copied_meta.resolve() != grid_metadata_path.resolve():
        shutil.copy2(str(grid_metadata_path), str(copied_meta))
    grid_meta = read_json(grid_metadata_path)
    score = score or score_trace_case(teacher_payload, grid_meta, pathreasoner_payload=pathreasoner_payload)
    write_json(output_dir / "teacher_patch_assignments.json", teacher_payload)
    write_json(output_dir / "auto_score.json", score)
    if conch_payload:
        write_json(output_dir / "conch_patch_assignments.json", conch_payload)
    if pathreasoner_payload:
        write_json(output_dir / "pathreasoner_patch_assignments.json", pathreasoner_payload)
    review_target = {
        "case_id": case_id,
        "review_status": "needs_review" if score.get("review_recommended") else "auto_pass",
        "instructions": [
            "Edit only target.patches and optional target_groups.",
            "Prioritize SSL/conventional/inflammatory boundaries and merge/split decisions.",
            "Keep every selected patch exactly once.",
        ],
        "target": teacher_payload,
        "auto_score": score,
    }
    write_json(output_dir / "review_target.json", review_target)
    return output_dir


def export_candidate_review_package(
    case_id,
    grid_thumbnail_path,
    grid_metadata_path,
    candidate_results,
    output_dir,
    selection=None,
    pathreasoner_payload=None,
    conch_payload=None,
    auto_review=None,
):
    output_dir = ensure_dir(output_dir)
    grid_thumbnail_path = Path(grid_thumbnail_path)
    grid_metadata_path = Path(grid_metadata_path)
    copied_image = output_dir / grid_thumbnail_path.name
    copied_meta = output_dir / grid_metadata_path.name
    if copied_image.resolve() != grid_thumbnail_path.resolve():
        shutil.copy2(str(grid_thumbnail_path), str(copied_image))
    if copied_meta.resolve() != grid_metadata_path.resolve():
        shutil.copy2(str(grid_metadata_path), str(copied_meta))
    grid_meta = read_json(grid_metadata_path)
    auto_review = auto_review or {}
    selection = selection or auto_review.get("selection") or select_best_candidate(candidate_results)
    selected_index = selection.get("selected_index")
    selected_payload = {"patches": []}
    selected_score = None
    if selected_index is not None and 0 <= int(selected_index) < len(candidate_results):
        selected_pool = auto_review.get("candidates") or candidate_results
        selected_payload = selected_pool[int(selected_index)].get("payload", {"patches": []})
        selected_score = selected_pool[int(selected_index)].get("score")
    write_json(
        output_dir / "gemini_candidates.json",
        {
            "candidates": auto_review.get("candidates") or candidate_results,
            "selection": selection,
            "auto_review": auto_review,
        },
    )
    if pathreasoner_payload:
        write_json(output_dir / "pathreasoner_patch_assignments.json", pathreasoner_payload)
    if conch_payload:
        write_json(output_dir / "conch_patch_assignments.json", conch_payload)
    candidate_payloads = [item.get("payload", {"patches": []}) for item in (auto_review.get("candidates") or candidate_results)]
    diff_view = build_candidate_diff_view(candidate_payloads)
    diff_view["agreement"] = auto_review.get("candidate_agreement", selection.get("agreement", {}))
    write_json(output_dir / "candidate_diff.json", diff_view)
    review_target = {
        "case_id": case_id,
        "teacher_mode": "image_only_screening_trace",
        "report_available": False,
        "review_status": selection.get("review_status", "needs_review"),
        "human_reviewed": False,
        "selected_candidate_index": selected_index,
        "selection_reason": selection.get("review_reason") or selection.get("reason", ""),
        "instructions": [
            "This is image-only screening trace supervision, not final diagnosis.",
            "Edit target.patches and optional target_groups only.",
            "Keep every selected patch exactly once.",
            "Prioritize SSL/conventional/inflammatory boundaries and cluster merge/split decisions.",
        ],
        "target": selected_payload,
        "target_groups": [],
        "auto_score": selected_score,
        "candidate_agreement": auto_review.get("candidate_agreement", selection.get("agreement", {})),
        "auto_review": auto_review,
    }
    write_json(output_dir / "review_target.json", review_target)
    html = _build_visual_review_html(case_id, copied_image.name, grid_meta, auto_review, review_target)
    with open(output_dir / "index.html", "w", encoding="utf-8") as handle:
        handle.write(html)
    return output_dir


def _read_assignment_payload(path):
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("target"), dict):
        return payload["target"]
    if isinstance(payload, dict) and isinstance(payload.get("patches"), list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("clusters"), list):
        patches = []
        for cluster in payload.get("clusters", []):
            for patch_id in cluster.get("patch_ids_ordered", []):
                patches.append(
                    {
                        "patch_id": patch_id,
                        "region_semantic": cluster.get("l"),
                        "name": cluster.get("metadata", {}).get("group_name", cluster.get("l", "")),
                        "description": cluster.get("desc", ""),
                        "require_high_magnification": bool(cluster.get("d", False)),
                        "severity_reasoning": cluster.get("metadata", {}).get("severity_reasoning", ""),
                        "diagnostic_priority": int(cluster.get("s", 0)),
                        "observation_points": list(cluster.get("evidence", [])),
                    }
                )
        return {"patches": patches}
    raise ValueError("Unsupported assignment payload: {0}".format(path))


def read_assignment_payload(path):
    return _read_assignment_payload(path)
