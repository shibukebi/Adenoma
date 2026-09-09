from __future__ import annotations

import csv
from io import StringIO
import json

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from .challenge_reviews import (
    final_class_label,
    peer_summary,
    record_history,
    serialize_review,
    serialize_roi,
    validate_case_payload,
    validate_roi_payload,
)
from .challenge_taxonomy import taxonomy_payload
from .db import connect, transaction, utc_now


class CaseReviewPayload(BaseModel):
    version: int = Field(ge=0)
    lesion_diagnosis: str
    hgd_status: str
    label_action: str
    challenge_disposition: str = "retain_challenge"
    primary_challenge: str = ""
    primary_challenge_other: str = Field(default="", max_length=500)
    modifiers: list[str] = []
    difficulty_note: str = Field(default="", max_length=5000)
    expert_confidence: int = Field(ge=1, le=5)
    no_localizable_evidence: bool = False
    no_roi_reason: str = ""
    no_roi_reason_other: str = Field(default="", max_length=500)


class EvidenceItem(BaseModel):
    evidence_code: str
    evidence_other: str = Field(default="", max_length=500)


class RoiPayload(BaseModel):
    review_version: int = Field(ge=1)
    version: int = Field(default=0, ge=0)
    x_level0: float
    y_level0: float
    width_level0: float
    height_level0: float
    mpp_x: float | None = None
    mpp_y: float | None = None
    physical_width_um: float | None = None
    physical_height_um: float | None = None
    viewer_zoom: float | None = None
    diagnostic_role: str = ""
    differential_direction: str = ""
    evidence_strength: str = ""
    evidence: list[EvidenceItem] = []
    note: str = Field(default="", max_length=5000)


class VersionPayload(BaseModel):
    version: int = Field(ge=1)


class RoiReorderPayload(VersionPayload):
    ordered_roi_ids: list[int] = Field(min_length=1, max_length=3)


def _require_core_slide(connection, slide_id):
    slide = connection.execute("SELECT * FROM slides WHERE slide_id=?", (slide_id,)).fetchone()
    if not slide:
        raise HTTPException(status_code=404, detail="Slide not found")
    if slide["wrong_configurations"] != 14:
        raise HTTPException(status_code=403, detail="ROI review is limited to 14/14 challenge slides")
    return slide


def _owned_review(connection, review_id, user):
    row = connection.execute("SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Challenge review not found")
    if row["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Review belongs to another user")
    if row["locked_at"]:
        raise HTTPException(status_code=423, detail="Challenge review is locked")
    return row


def _check_version(actual, expected):
    if actual != expected:
        raise HTTPException(status_code=409, detail="Review was changed in another session; reload it")


def _replace_modifiers(connection, review_id, modifiers):
    connection.execute("DELETE FROM challenge_review_modifiers WHERE review_id=?", (review_id,))
    connection.executemany(
        "INSERT INTO challenge_review_modifiers(review_id,modifier_code) VALUES (?,?)",
        [(review_id, code) for code in sorted(set(modifiers))],
    )


def _replace_evidence(connection, roi_id, evidence):
    connection.execute("DELETE FROM challenge_roi_evidence WHERE roi_id=?", (roi_id,))
    connection.executemany(
        "INSERT INTO challenge_roi_evidence(roi_id,evidence_code,evidence_other) VALUES (?,?,?)",
        [(roi_id, item.evidence_code, item.evidence_other.strip()) for item in evidence],
    )


def _save_case(slide_id, payload, user, submitted):
    error = validate_case_payload(payload, submitted=submitted)
    if error:
        raise HTTPException(status_code=422, detail=error)
    now = utc_now()
    with transaction() as connection:
        slide = _require_core_slide(connection, slide_id)
        final_label = final_class_label(payload.lesion_diagnosis, payload.hgd_status)
        if payload.challenge_disposition in {"pending_label_adjudication", "exclude_label_error"}:
            if final_label != slide["consensus_wrong_class"] or final_label == slide["original_label"]:
                raise HTTPException(
                    status_code=422,
                    detail="Label-error disposition requires a final label matching model consensus and differing from the original label",
                )
        review = connection.execute(
            "SELECT * FROM challenge_case_reviews WHERE slide_id=? AND user_id=?",
            (slide_id, user["id"]),
        ).fetchone()
        if review and review["locked_at"]:
            raise HTTPException(status_code=423, detail="Challenge review is locked")
        if review:
            _check_version(review["version"], payload.version)
            review_id = review["id"]
            next_version = review["version"] + 1
        else:
            if payload.version != 0:
                raise HTTPException(status_code=409, detail="Challenge review does not exist; reload it")
            cursor = connection.execute(
                """
                INSERT INTO challenge_case_reviews(
                    slide_id,user_id,lesion_diagnosis,hgd_status,label_action,
                    challenge_disposition,expert_confidence,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (slide_id, user["id"], payload.lesion_diagnosis, payload.hgd_status,
                 payload.label_action, payload.challenge_disposition, payload.expert_confidence, now, now),
            )
            review_id = cursor.lastrowid
            next_version = 1
        roi_count = connection.execute(
            "SELECT COUNT(*) FROM challenge_rois WHERE review_id=?", (review_id,)
        ).fetchone()[0]
        if payload.no_localizable_evidence and roi_count:
            raise HTTPException(status_code=422, detail="Remove all ROIs before selecting no localizable evidence")
        if submitted:
            for roi in connection.execute("SELECT * FROM challenge_rois WHERE review_id=?", (review_id,)):
                roi_payload = RoiPayload(
                    review_version=next_version,
                    version=roi["version"],
                    **{key: roi[key] for key in (
                        "x_level0", "y_level0", "width_level0", "height_level0", "mpp_x", "mpp_y",
                        "physical_width_um", "physical_height_um", "viewer_zoom", "diagnostic_role",
                        "differential_direction", "evidence_strength", "note",
                    )},
                    evidence=[EvidenceItem(**dict(item)) for item in connection.execute(
                        "SELECT evidence_code,evidence_other FROM challenge_roi_evidence WHERE roi_id=?", (roi["id"],)
                    )],
                )
                roi_error = validate_roi_payload(payload.primary_challenge, roi_payload, submitted=True)
                if roi_error:
                    raise HTTPException(status_code=422, detail=f"ROI {roi['display_order']}: {roi_error}")
        connection.execute(
            """
            UPDATE challenge_case_reviews SET lesion_diagnosis=?,hgd_status=?,label_action=?,
                challenge_disposition=?,primary_challenge=?,primary_challenge_other=?,difficulty_note=?,expert_confidence=?,
                no_localizable_evidence=?,no_roi_reason=?,no_roi_reason_other=?,status=?,version=?,
                updated_at=?,submitted_at=CASE WHEN ?='submitted' THEN ? ELSE submitted_at END
            WHERE id=?
            """,
            (
                payload.lesion_diagnosis, payload.hgd_status, payload.label_action,
                payload.challenge_disposition, payload.primary_challenge, payload.primary_challenge_other.strip(), payload.difficulty_note.strip(),
                payload.expert_confidence, int(payload.no_localizable_evidence), payload.no_roi_reason,
                payload.no_roi_reason_other.strip(), "submitted" if submitted else (review["status"] if review else "draft"),
                next_version, now, "submitted" if submitted else "draft", now, review_id,
            ),
        )
        _replace_modifiers(connection, review_id, payload.modifiers)
        record_history(connection, review_id, "submit" if submitted else "draft_save", user["id"])
        saved = connection.execute("SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()
        return serialize_review(connection, saved, include_peer=submitted)


def create_challenge_router(current_user, admin_user):
    router = APIRouter()

    @router.get("/api/challenge-taxonomy")
    async def challenge_taxonomy(user=Depends(current_user)):
        return taxonomy_payload()

    @router.get("/api/slides/{slide_id}/challenge-review")
    async def get_challenge_review(slide_id: str, user=Depends(current_user)):
        connection = connect()
        try:
            slide = _require_core_slide(connection, slide_id)
            row = connection.execute(
                "SELECT * FROM challenge_case_reviews WHERE slide_id=? AND user_id=?",
                (slide_id, user["id"]),
            ).fetchone()
            review = serialize_review(connection, row, include_peer=bool(row and row["status"] == "submitted"))
            return {"slide_id": slide_id, "eligible": True, "review": review}
        finally:
            connection.close()

    @router.put("/api/slides/{slide_id}/challenge-review/draft")
    async def save_challenge_draft(slide_id: str, payload: CaseReviewPayload, user=Depends(current_user)):
        return _save_case(slide_id, payload, user, submitted=False)

    @router.post("/api/slides/{slide_id}/challenge-review/submit")
    async def submit_challenge_review(slide_id: str, payload: CaseReviewPayload, user=Depends(current_user)):
        return _save_case(slide_id, payload, user, submitted=True)

    @router.post("/api/challenge-reviews/{review_id}/rois", status_code=201)
    async def create_roi(review_id: int, payload: RoiPayload, user=Depends(current_user)):
        with transaction() as connection:
            review = _owned_review(connection, review_id, user)
            _check_version(review["version"], payload.review_version)
            if review["no_localizable_evidence"]:
                raise HTTPException(status_code=422, detail="Clear no localizable evidence before adding an ROI")
            count = connection.execute("SELECT COUNT(*) FROM challenge_rois WHERE review_id=?", (review_id,)).fetchone()[0]
            if count >= 3:
                raise HTTPException(status_code=422, detail="A challenge review can contain at most 3 ROIs")
            error = validate_roi_payload(review["primary_challenge"], payload, submitted=False)
            if error:
                raise HTTPException(status_code=422, detail=error)
            now = utc_now()
            cursor = connection.execute(
                """
                INSERT INTO challenge_rois(
                    review_id,slide_id,display_order,x_level0,y_level0,width_level0,height_level0,
                    mpp_x,mpp_y,physical_width_um,physical_height_um,viewer_zoom,diagnostic_role,
                    differential_direction,evidence_strength,note,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (review_id, review["slide_id"], count + 1, payload.x_level0, payload.y_level0,
                 payload.width_level0, payload.height_level0, payload.mpp_x, payload.mpp_y,
                 payload.physical_width_um, payload.physical_height_um, payload.viewer_zoom,
                 payload.diagnostic_role, payload.differential_direction, payload.evidence_strength,
                 payload.note.strip(), now, now),
            )
            roi_id = cursor.lastrowid
            _replace_evidence(connection, roi_id, payload.evidence)
            connection.execute(
                "UPDATE challenge_case_reviews SET version=version+1,updated_at=? WHERE id=?", (now, review_id)
            )
            record_history(connection, review_id, "roi_create", user["id"])
            roi = connection.execute("SELECT * FROM challenge_rois WHERE id=?", (roi_id,)).fetchone()
            new_version = connection.execute("SELECT version FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()[0]
            return {"review_version": new_version, "roi": serialize_roi(connection, roi)}

    @router.put("/api/challenge-rois/{roi_id}")
    async def update_roi(roi_id: int, payload: RoiPayload, user=Depends(current_user)):
        with transaction() as connection:
            roi = connection.execute("SELECT * FROM challenge_rois WHERE id=?", (roi_id,)).fetchone()
            if not roi:
                raise HTTPException(status_code=404, detail="ROI not found")
            review = _owned_review(connection, roi["review_id"], user)
            _check_version(review["version"], payload.review_version)
            _check_version(roi["version"], payload.version)
            error = validate_roi_payload(review["primary_challenge"], payload, submitted=False)
            if error:
                raise HTTPException(status_code=422, detail=error)
            now = utc_now()
            connection.execute(
                """
                UPDATE challenge_rois SET x_level0=?,y_level0=?,width_level0=?,height_level0=?,
                    mpp_x=?,mpp_y=?,physical_width_um=?,physical_height_um=?,viewer_zoom=?,
                    diagnostic_role=?,differential_direction=?,evidence_strength=?,note=?,
                    version=version+1,updated_at=? WHERE id=?
                """,
                (payload.x_level0, payload.y_level0, payload.width_level0, payload.height_level0,
                 payload.mpp_x, payload.mpp_y, payload.physical_width_um, payload.physical_height_um,
                 payload.viewer_zoom, payload.diagnostic_role, payload.differential_direction,
                 payload.evidence_strength, payload.note.strip(), now, roi_id),
            )
            _replace_evidence(connection, roi_id, payload.evidence)
            connection.execute("UPDATE challenge_case_reviews SET version=version+1,updated_at=? WHERE id=?", (now, review["id"]))
            record_history(connection, review["id"], "roi_update", user["id"])
            updated = connection.execute("SELECT * FROM challenge_rois WHERE id=?", (roi_id,)).fetchone()
            new_version = connection.execute("SELECT version FROM challenge_case_reviews WHERE id=?", (review["id"],)).fetchone()[0]
            return {"review_version": new_version, "roi": serialize_roi(connection, updated)}

    @router.delete("/api/challenge-rois/{roi_id}")
    async def delete_roi(roi_id: int, payload: VersionPayload = Body(...), user=Depends(current_user)):
        with transaction() as connection:
            roi = connection.execute("SELECT * FROM challenge_rois WHERE id=?", (roi_id,)).fetchone()
            if not roi:
                raise HTTPException(status_code=404, detail="ROI not found")
            review = _owned_review(connection, roi["review_id"], user)
            _check_version(review["version"], payload.version)
            connection.execute("DELETE FROM challenge_rois WHERE id=?", (roi_id,))
            remaining = connection.execute(
                "SELECT id FROM challenge_rois WHERE review_id=? ORDER BY display_order", (review["id"],)
            ).fetchall()
            for order, row in enumerate(remaining, 1):
                connection.execute("UPDATE challenge_rois SET display_order=? WHERE id=?", (order, row["id"]))
            connection.execute("UPDATE challenge_case_reviews SET version=version+1,updated_at=? WHERE id=?", (utc_now(), review["id"]))
            record_history(connection, review["id"], "roi_delete", user["id"])
            return {"review_version": review["version"] + 1}

    @router.put("/api/challenge-reviews/{review_id}/rois/order")
    async def reorder_rois(review_id: int, payload: RoiReorderPayload, user=Depends(current_user)):
        with transaction() as connection:
            review = _owned_review(connection, review_id, user)
            _check_version(review["version"], payload.version)
            rows = connection.execute(
                "SELECT id FROM challenge_rois WHERE review_id=? ORDER BY display_order", (review_id,)
            ).fetchall()
            existing_ids = [row["id"] for row in rows]
            if len(set(payload.ordered_roi_ids)) != len(payload.ordered_roi_ids) or set(payload.ordered_roi_ids) != set(existing_ids):
                raise HTTPException(status_code=422, detail="ROI order must contain every ROI exactly once")
            now = utc_now()
            connection.execute(
                "UPDATE challenge_rois SET display_order=-display_order WHERE review_id=?", (review_id,)
            )
            for order, roi_id in enumerate(payload.ordered_roi_ids, 1):
                connection.execute(
                    "UPDATE challenge_rois SET display_order=?,version=version+1,updated_at=? WHERE id=?",
                    (order, now, roi_id),
                )
            connection.execute(
                "UPDATE challenge_case_reviews SET version=version+1,updated_at=? WHERE id=?", (now, review_id)
            )
            record_history(connection, review_id, "roi_reorder", user["id"])
            updated = connection.execute(
                "SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)
            ).fetchone()
            return serialize_review(connection, updated, include_peer=True)

    def set_lock(review_id, payload, user, locked):
        with transaction() as connection:
            review = connection.execute("SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()
            if not review:
                raise HTTPException(status_code=404, detail="Challenge review not found")
            _check_version(review["version"], payload.version)
            if locked and review["status"] != "submitted":
                raise HTTPException(status_code=422, detail="Only submitted reviews can be locked")
            now = utc_now()
            connection.execute(
                "UPDATE challenge_case_reviews SET locked_by=?,locked_at=?,version=version+1,updated_at=? WHERE id=?",
                (user["id"] if locked else None, now if locked else None, now, review_id),
            )
            record_history(connection, review_id, "lock" if locked else "unlock", user["id"])
            updated = connection.execute("SELECT * FROM challenge_case_reviews WHERE id=?", (review_id,)).fetchone()
            return serialize_review(connection, updated, include_peer=True)

    @router.post("/api/admin/challenge-reviews/{review_id}/lock")
    async def lock_review(review_id: int, payload: VersionPayload, user=Depends(admin_user)):
        return set_lock(review_id, payload, user, True)

    @router.post("/api/admin/challenge-reviews/{review_id}/unlock")
    async def unlock_review(review_id: int, payload: VersionPayload, user=Depends(admin_user)):
        return set_lock(review_id, payload, user, False)

    def export_rows():
        connection = connect()
        try:
            return connection.execute(
                """
                SELECT r.id AS review_id,r.slide_id,s.original_label,u.username,u.display_name,
                       r.lesion_diagnosis,r.hgd_status,r.label_action,r.primary_challenge,
                       r.challenge_disposition,r.primary_challenge_other,r.difficulty_note,r.expert_confidence,
                       r.no_localizable_evidence,r.no_roi_reason,r.no_roi_reason_other,r.status,
                       r.version,r.locked_at,r.created_at,r.updated_at,r.submitted_at,
                       GROUP_CONCAT(m.modifier_code,';') AS modifiers,
                       (SELECT COUNT(*) FROM challenge_rois roi WHERE roi.review_id=r.id) AS roi_count
                FROM challenge_case_reviews r JOIN slides s ON s.slide_id=r.slide_id
                JOIN users u ON u.id=r.user_id
                LEFT JOIN challenge_review_modifiers m ON m.review_id=r.id
                GROUP BY r.id ORDER BY r.slide_id,u.username
                """
            ).fetchall()
        finally:
            connection.close()

    def csv_response(filename, rows, fallback):
        output = StringIO()
        headers = list(rows[0].keys()) if rows else fallback
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(tuple(row) for row in rows)
        return Response(output.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @router.get("/api/exports/challenge-case-reviews.csv")
    async def export_case_reviews(user=Depends(current_user)):
        return csv_response("challenge_case_reviews.csv", export_rows(), ["review_id", "slide_id"])

    @router.get("/api/exports/challenge-resolutions.csv")
    async def export_challenge_resolutions(user=Depends(current_user)):
        connection = connect()
        try:
            rows = connection.execute(
                """
                SELECT s.slide_id,s.original_label,s.consensus_wrong_class,s.consensus_wrong_count,
                       r.submitted_reviews,r.retain_votes,r.consensus_label_error_votes,r.resolution_status,
                       CASE WHEN r.resolution_status='eligible_for_exclusion' THEN s.consensus_wrong_class END AS suggested_revised_label
                FROM slides s JOIN challenge_slide_resolution r ON r.slide_id=s.slide_id
                WHERE s.wrong_configurations=14
                ORDER BY CASE r.resolution_status
                           WHEN 'eligible_for_exclusion' THEN 1 WHEN 'pending_adjudication' THEN 2
                           WHEN 'retained_challenge' THEN 3 WHEN 'unresolved' THEN 4 ELSE 5 END,
                         s.consensus_wrong_count DESC,s.slide_id
                """
            ).fetchall()
        finally:
            connection.close()
        return csv_response("challenge_resolutions.csv", rows, ["slide_id", "resolution_status"])

    def roi_export_rows():
        connection = connect()
        try:
            rows = connection.execute(
                """
                SELECT roi.*,r.user_id,r.lesion_diagnosis,r.hgd_status,r.primary_challenge,
                       u.username,GROUP_CONCAT(e.evidence_code || CASE WHEN e.evidence_other<>'' THEN ':'||e.evidence_other ELSE '' END,';') AS evidence_codes
                FROM challenge_rois roi JOIN challenge_case_reviews r ON r.id=roi.review_id
                JOIN users u ON u.id=r.user_id LEFT JOIN challenge_roi_evidence e ON e.roi_id=roi.id
                GROUP BY roi.id ORDER BY roi.slide_id,u.username,roi.display_order
                """
            ).fetchall()
            return rows
        finally:
            connection.close()

    @router.get("/api/exports/challenge-rois.csv")
    async def export_rois_csv(user=Depends(current_user)):
        return csv_response("challenge_rois.csv", roi_export_rows(), ["id", "review_id", "slide_id"])

    @router.get("/api/exports/challenge-rois.geojson")
    async def export_rois_geojson(user=Depends(current_user)):
        features = []
        for row in roi_export_rows():
            item = dict(row)
            x, y, width, height = (item[key] for key in ("x_level0", "y_level0", "width_level0", "height_level0"))
            geometry = {"type": "Polygon", "coordinates": [[[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]]]}
            for key in ("x_level0", "y_level0", "width_level0", "height_level0"):
                item.pop(key)
            features.append({"type": "Feature", "id": row["id"], "geometry": geometry, "properties": item})
        content = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False)
        return Response(content.encode("utf-8"), media_type="application/geo+json",
                        headers={"Content-Disposition": 'attachment; filename="challenge_rois.geojson"'})

    return router
