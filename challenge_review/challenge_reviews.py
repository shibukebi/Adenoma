from __future__ import annotations

import json

from .challenge_taxonomy import (
    CHALLENGE_DISPOSITIONS,
    DIAGNOSTIC_ROLES,
    DIFFICULTY_MODIFIERS,
    EVIDENCE_BY_CHALLENGE,
    EVIDENCE_STRENGTHS,
    GENERAL_DIRECTIONS,
    GENERAL_EVIDENCE,
    HGD_DIRECTIONS,
    HGD_STATUSES,
    LABEL_ACTIONS,
    LESION_DIAGNOSES,
    NO_ROI_REASONS,
    PAIR_DIRECTIONS,
    PRIMARY_CHALLENGES,
)
from .db import utc_now


LEGACY_CONFIDENCE = {"low": 2, "medium": 3, "high": 4}
HGD_LABELS = {"SSLD": "SSL", "TSAD": "TSA", "TAD": "TA", "TVAD": "TVA"}
HGD_CLASS_LABELS = {"SSL": "SSLD", "TSA": "TSAD", "TA": "TAD", "TVA": "TVAD"}


def final_class_label(lesion: str, hgd_status: str) -> str | None:
    if lesion in HGD_CLASS_LABELS and hgd_status == "Present":
        return HGD_CLASS_LABELS[lesion]
    if lesion in {"HP", "SSL", "TSA", "TA", "TVA", "IP", "USA"} and hgd_status == "Absent":
        return lesion
    return None


def split_legacy_label(label: str) -> tuple[str, str]:
    if label in HGD_LABELS:
        return HGD_LABELS[label], "Present"
    if label in {"HP", "SSL", "TSA", "TA", "TVA", "IP", "USA"}:
        return label, "Absent"
    return "Other", "Uncertain/conflicting"


def migrate_legacy_reviews(connection) -> int:
    rows = connection.execute(
        """
        SELECT r.*,s.original_label FROM reviews r JOIN slides s ON s.slide_id=r.slide_id
        WHERE s.wrong_configurations=14
        ORDER BY r.id
        """
    ).fetchall()
    migrated = 0
    for row in rows:
        if connection.execute(
            "SELECT 1 FROM challenge_case_reviews WHERE slide_id=? AND user_id=?",
            (row["slide_id"], row["user_id"]),
        ).fetchone():
            continue
        lesion, hgd = split_legacy_label(row["revised_label"])
        original_lesion, original_hgd = split_legacy_label(row["original_label"])
        label_action = (
            "confirm_original"
            if (lesion, hgd) == (original_lesion, original_hgd)
            else "correct_original"
        )
        now = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO challenge_case_reviews(
                slide_id,user_id,lesion_diagnosis,hgd_status,label_action,
                difficulty_note,expert_confidence,status,version,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,'draft',1,?,?)
            """,
            (
                row["slide_id"], row["user_id"], lesion, hgd, label_action,
                row["notes"], LEGACY_CONFIDENCE.get(row["diagnostic_confidence"], 3), now, now,
            ),
        )
        review_id = cursor.lastrowid
        snapshot = review_snapshot(connection, review_id)
        connection.execute(
            """
            INSERT INTO challenge_review_history(
                review_id,slide_id,user_id,action,version,snapshot_json,changed_by,changed_at
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (review_id, row["slide_id"], row["user_id"], "legacy_migration", 1,
             json.dumps(snapshot, ensure_ascii=False), row["user_id"], now),
        )
        migrated += 1
    return migrated


def serialize_review(connection, review_row, include_peer=False):
    if review_row is None:
        return None
    review = dict(review_row)
    review["modifiers"] = [
        row["modifier_code"]
        for row in connection.execute(
            "SELECT modifier_code FROM challenge_review_modifiers WHERE review_id=? ORDER BY modifier_code",
            (review["id"],),
        )
    ]
    review["rois"] = [serialize_roi(connection, row) for row in connection.execute(
        "SELECT * FROM challenge_rois WHERE review_id=? ORDER BY display_order", (review["id"],)
    )]
    review["locked"] = bool(review["locked_at"])
    if include_peer:
        review["peer_summary"] = peer_summary(connection, review["slide_id"])
    return review


def serialize_roi(connection, roi_row):
    roi = dict(roi_row)
    evidence = connection.execute(
        "SELECT evidence_code,evidence_other FROM challenge_roi_evidence WHERE roi_id=? ORDER BY evidence_code,evidence_other",
        (roi["id"],),
    ).fetchall()
    roi["evidence"] = [dict(row) for row in evidence]
    return roi


def review_snapshot(connection, review_id: int):
    row = connection.execute("SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()
    return serialize_review(connection, row) if row else None


def record_history(connection, review_id: int, action: str, changed_by: int):
    snapshot = review_snapshot(connection, review_id)
    connection.execute(
        """
        INSERT INTO challenge_review_history(
            review_id,slide_id,user_id,action,version,snapshot_json,changed_by,changed_at
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            review_id, snapshot["slide_id"], snapshot["user_id"], action, snapshot["version"],
            json.dumps(snapshot, ensure_ascii=False), changed_by, utc_now(),
        ),
    )


def peer_summary(connection, slide_id: str):
    slide = connection.execute(
        "SELECT original_label,consensus_wrong_class FROM slides WHERE slide_id=?", (slide_id,)
    ).fetchone()
    rows = connection.execute(
        """
        SELECT lesion_diagnosis,hgd_status,primary_challenge,challenge_disposition,
               (SELECT COUNT(*) FROM challenge_rois roi WHERE roi.review_id=r.id) AS roi_count
        FROM challenge_case_reviews r WHERE slide_id=? AND status='submitted'
        """,
        (slide_id,),
    ).fetchall()
    if not rows:
        return None

    def counts(key):
        values = {}
        for row in rows:
            value = row[key] or "-"
            values[value] = values.get(value, 0) + 1
        return [{"value": value, "votes": votes} for value, votes in sorted(values.items())]

    qualified = {}
    for row in rows:
        label = final_class_label(row["lesion_diagnosis"], row["hgd_status"])
        if (
            row["challenge_disposition"] in {"pending_label_adjudication", "exclude_label_error"}
            and label
            and slide
            and label == slide["consensus_wrong_class"]
            and label != slide["original_label"]
        ):
            qualified[label] = qualified.get(label, 0) + 1
    agreed_label, agreed_votes = max(qualified.items(), key=lambda item: item[1], default=(None, 0))
    if agreed_votes >= 2:
        resolution = "eligible_for_exclusion"
    elif agreed_votes == 1:
        resolution = "pending_adjudication"
    elif any(row["challenge_disposition"] == "retain_challenge" for row in rows):
        resolution = "retained_challenge"
    else:
        resolution = "unresolved"
    return {
        "review_count": len(rows),
        "lesion_votes": counts("lesion_diagnosis"),
        "hgd_votes": counts("hgd_status"),
        "challenge_votes": counts("primary_challenge"),
        "disposition_votes": counts("challenge_disposition"),
        "roi_count_distribution": counts("roi_count"),
        "resolution_status": resolution,
        "agreed_revised_label": agreed_label,
        "agreed_reviewer_count": agreed_votes,
    }


def validate_case_payload(payload, submitted=False):
    if payload.lesion_diagnosis not in LESION_DIAGNOSES:
        return "Unknown lesion diagnosis"
    if payload.hgd_status not in HGD_STATUSES:
        return "Unknown HGD status"
    if payload.label_action not in LABEL_ACTIONS:
        return "Unknown label action"
    if payload.challenge_disposition not in CHALLENGE_DISPOSITIONS:
        return "Unknown challenge disposition"
    if payload.challenge_disposition in {"pending_label_adjudication", "exclude_label_error"}:
        if payload.label_action != "correct_original":
            return "Label-error disposition requires correcting the original label"
    if payload.primary_challenge and payload.primary_challenge not in PRIMARY_CHALLENGES:
        return "Unknown primary challenge"
    if not set(payload.modifiers).issubset(DIFFICULTY_MODIFIERS):
        return "Unknown difficulty modifier"
    if payload.no_roi_reason and payload.no_roi_reason not in NO_ROI_REASONS:
        return "Unknown no-ROI reason"
    if submitted and not payload.primary_challenge:
        return "Primary challenge is required"
    if payload.primary_challenge == "Other differential" and submitted and not payload.primary_challenge_other.strip():
        return "Other differential description is required"
    if payload.no_localizable_evidence and submitted and not payload.no_roi_reason:
        return "No-ROI reason is required"
    if payload.no_roi_reason == "other" and submitted and not payload.no_roi_reason_other.strip():
        return "Other no-ROI reason is required"
    return None


def valid_evidence(primary_challenge: str, items) -> bool:
    allowed = set(GENERAL_EVIDENCE) | set(EVIDENCE_BY_CHALLENGE.get(primary_challenge, []))
    return bool(items) and all(
        item.evidence_code in allowed
        and (item.evidence_code != "other" or bool(item.evidence_other.strip()))
        for item in items
    )


def valid_direction(primary_challenge: str, value: str) -> bool:
    if primary_challenge in {"Focal HGD", "Borderline HGD"}:
        return value in HGD_DIRECTIONS
    return value in PAIR_DIRECTIONS.get(primary_challenge, GENERAL_DIRECTIONS)


def validate_roi_payload(primary_challenge: str, payload, submitted=False):
    if payload.width_level0 <= 0 or payload.height_level0 <= 0:
        return "ROI dimensions must be positive"
    if payload.x_level0 < 0 or payload.y_level0 < 0:
        return "ROI coordinates must be non-negative"
    if submitted or payload.diagnostic_role:
        if payload.diagnostic_role not in DIAGNOSTIC_ROLES:
            return "Unknown diagnostic role"
    if submitted or payload.differential_direction:
        if not valid_direction(primary_challenge, payload.differential_direction):
            return "Unknown differential direction"
    if submitted or payload.evidence_strength:
        if payload.evidence_strength not in EVIDENCE_STRENGTHS:
            return "Unknown evidence strength"
    if submitted and not valid_evidence(primary_challenge, payload.evidence):
        return "At least one valid evidence type is required"
    if payload.evidence and not valid_evidence(primary_challenge, payload.evidence):
        return "Unknown evidence type"
    return None
