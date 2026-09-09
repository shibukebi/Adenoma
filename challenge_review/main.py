from collections import Counter
import asyncio
from contextlib import asynccontextmanager
import csv
from io import StringIO
import json
from pathlib import Path

from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import CLASS_ORDER, COOKIE_SECURE, STATIC_ROOT
from .db import connect, initialize_database, transaction, utc_now
from .security import (
    create_session,
    delete_session,
    get_session_user,
    hash_password,
    verify_password,
)
from .wsi import (
    get_dzi,
    get_metadata,
    get_thumbnail,
    get_tile,
    warm_manager,
    wsi_service,
)
from .challenge_api import create_challenge_router


SESSION_COOKIE = "challenge_review_session"
QUALITY_FLAGS = {
    "poor_quality",
    "tissue_fold",
    "staining_issue",
    "incomplete_tissue",
    "label_dispute",
}
SORT_SQL = {
    "consensus_desc": "s.consensus_wrong_count DESC, s.wrong_configurations DESC, s.mean_wrong_confidence DESC, s.slide_id",
    "consensus_asc": "s.consensus_wrong_count ASC, s.wrong_configurations ASC, s.slide_id",
    "wrong_desc": "s.wrong_configurations DESC, s.consensus_wrong_count DESC, s.slide_id",
    "confidence_desc": "s.mean_wrong_confidence DESC, s.consensus_wrong_count DESC, s.slide_id",
    "rank": "s.hardness_rank ASC, s.slide_id",
    "slide_id": "s.slide_id COLLATE NOCASE ASC",
}


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=512)


class ReviewRequest(BaseModel):
    revised_label: str
    status: str
    diagnostic_confidence: str
    quality_flags: list[str] = []
    notes: str = Field(default="", max_length=5000)


class UserRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=10, max_length=512)
    role: str = "reviewer"


class PasswordRequest(BaseModel):
    password: str = Field(min_length=10, max_length=512)


@asynccontextmanager
async def lifespan(_app):
    initialize_database()
    wsi_service.start()
    yield
    await asyncio.to_thread(wsi_service.close)


app = FastAPI(title="Challenge Set WSI Review", docs_url=None, redoc_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


async def current_user(challenge_review_session: str | None = Cookie(default=None)):
    user = get_session_user(challenge_review_session)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


async def admin_user(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator required")
    return user


app.include_router(create_challenge_router(current_user, admin_user))


def slide_row_or_404(slide_id):
    connection = connect()
    try:
        row = connection.execute("SELECT * FROM slides WHERE slide_id = ?", (slide_id,)).fetchone()
    finally:
        connection.close()
    if not row:
        raise HTTPException(status_code=404, detail="Slide not found")
    return dict(row)


def require_wsi(slide_id):
    slide = slide_row_or_404(slide_id)
    if not slide["wsi_available"] or not slide["wsi_path"]:
        raise HTTPException(status_code=404, detail="Original WSI is unavailable")
    return slide


def is_priority_hp_slide(slide):
    return slide["source"] == "hp" and (
        slide["wrong_configurations"] == 14 or slide["hardness_rank"] <= 200
    )


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/config")
async def public_config():
    return {"classes": CLASS_ORDER, "quality_flags": sorted(QUALITY_FLAGS)}


@app.post("/api/auth/login")
async def login(payload: LoginRequest, response: Response):
    with transaction() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE AND active = 1",
            (payload.username.strip(),),
        ).fetchone()
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token, expires = create_session(row["id"])
    response.set_cookie(
        SESSION_COOKIE,
        token,
        expires=expires,
        httponly=True,
        samesite="strict",
        secure=COOKIE_SECURE,
        path="/",
    )
    return {"id": row["id"], "username": row["username"], "display_name": row["display_name"], "role": row["role"]}


@app.post("/api/auth/logout")
async def logout(response: Response, challenge_review_session: str | None = Cookie(default=None)):
    delete_session(challenge_review_session)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user=Depends(current_user)):
    return user


def build_slide_filter(
    user_id,
    search,
    core_only,
    original_label,
    consensus_label,
    source,
    cv_fold,
    review_status,
    resolution_status,
    consistency_min,
    consistency_max,
    wsi_available,
):
    clauses = ["1 = 1"]
    parameters = []
    if search:
        clauses.append("s.slide_id LIKE ?")
        parameters.append(f"%{search.strip()}%")
    if core_only:
        clauses.append("s.wrong_configurations = 14")
    if original_label:
        clauses.append("s.original_label = ?")
        parameters.append(original_label)
    if consensus_label:
        clauses.append("s.consensus_wrong_class = ?")
        parameters.append(consensus_label)
    if source:
        clauses.append("s.source = ?")
        parameters.append(source)
    if cv_fold:
        clauses.append("s.cv_fold = ?")
        parameters.append(cv_fold)
    if review_status == "unreviewed":
        clauses.append("r.id IS NULL")
    elif review_status in {"completed", "questionable"}:
        clauses.append("r.status = ?")
        parameters.append(review_status)
    if resolution_status:
        clauses.append("COALESCE(csr.resolution_status, 'unreviewed') = ?")
        parameters.append(resolution_status)
    if consistency_min is not None:
        clauses.append("s.consensus_wrong_count >= ?")
        parameters.append(consistency_min)
    if consistency_max is not None:
        clauses.append("s.consensus_wrong_count <= ?")
        parameters.append(consistency_max)
    if wsi_available is not None:
        clauses.append("s.wsi_available = ?")
        parameters.append(int(wsi_available))
    return " AND ".join(clauses), parameters


@app.get("/api/slides")
async def list_slides(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200),
    search: str | None = None,
    core_only: bool = False,
    original_label: str | None = None,
    consensus_label: str | None = None,
    source: str | None = None,
    cv_fold: int | None = Query(None, ge=1, le=5),
    review_status: str | None = None,
    resolution_status: str | None = None,
    consistency_min: int | None = Query(None, ge=0, le=14),
    consistency_max: int | None = Query(None, ge=0, le=14),
    wsi_available: bool | None = None,
    sort: str = "consensus_desc",
    user=Depends(current_user),
):
    where, parameters = build_slide_filter(
        user["id"], search, core_only, original_label, consensus_label, source, cv_fold,
        review_status, resolution_status, consistency_min, consistency_max, wsi_available,
    )
    order_by = SORT_SQL.get(sort, SORT_SQL["consensus_desc"])
    base = """FROM slides s
              LEFT JOIN reviews r ON r.slide_id = s.slide_id AND r.user_id = ?
              LEFT JOIN challenge_case_reviews cr ON cr.slide_id=s.slide_id AND cr.user_id=?
              LEFT JOIN challenge_slide_resolution csr ON csr.slide_id=s.slide_id"""
    query_parameters = [user["id"], user["id"], *parameters]
    connection = connect()
    try:
        total = connection.execute(f"SELECT COUNT(*) {base} WHERE {where}", query_parameters).fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT s.slide_id, s.original_label, s.source, s.cv_fold, s.wsi_available,
                   s.wrong_configurations, s.consensus_wrong_class, s.consensus_wrong_count,
                   s.mean_wrong_confidence, s.hardness_rank,
                   COALESCE(r.status, 'unreviewed') AS review_status, r.revised_label,
                   COALESCE(cr.status, 'unreviewed') AS challenge_review_status,
                   COALESCE(cr.challenge_disposition, '') AS my_challenge_disposition,
                   COALESCE(csr.resolution_status, 'unreviewed') AS resolution_status
            {base} WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?
            """,
            [*query_parameters, page_size, (page - 1) * page_size],
        ).fetchall()
    finally:
        connection.close()
    return {
        "items": [dict(row) for row in rows],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


@app.get("/api/slides/{slide_id}")
async def slide_detail(slide_id: str, user=Depends(current_user)):
    connection = connect()
    try:
        slide = connection.execute("SELECT * FROM slides WHERE slide_id = ?", (slide_id,)).fetchone()
        if not slide:
            raise HTTPException(status_code=404, detail="Slide not found")
        predictions = connection.execute(
            """
            SELECT model, feature, configuration, predicted_label, confidence,
                   second_predicted_label, second_confidence, is_correct
            FROM predictions WHERE slide_id = ?
            ORDER BY CASE model WHEN 'CLAM-SB' THEN 1 WHEN 'TransMIL' THEN 2 WHEN 'DSMIL' THEN 3 ELSE 4 END,
                     CASE feature WHEN '2p5x' THEN 1 WHEN '5x' THEN 2 WHEN '10x' THEN 3 WHEN '20x' THEN 4 ELSE 5 END
            """,
            (slide_id,),
        ).fetchall()
        review = connection.execute(
            "SELECT * FROM reviews WHERE slide_id = ? AND user_id = ?",
            (slide_id, user["id"]),
        ).fetchone()
        peer_summary = None
        if review:
            peer_rows = connection.execute(
                """
                SELECT revised_label, COUNT(*) AS votes
                FROM reviews WHERE slide_id = ? GROUP BY revised_label ORDER BY votes DESC, revised_label
                """,
                (slide_id,),
            ).fetchall()
            peer_summary = {
                "review_count": sum(row["votes"] for row in peer_rows),
                "label_votes": [dict(row) for row in peer_rows],
            }
    finally:
        connection.close()
    review_data = dict(review) if review else None
    if review_data:
        review_data["quality_flags"] = json.loads(review_data["quality_flags"])
    return {
        "slide": dict(slide),
        "predictions": [dict(row) for row in predictions],
        "review": review_data,
        "peer_summary": peer_summary,
    }


@app.put("/api/slides/{slide_id}/review")
async def save_review(slide_id: str, payload: ReviewRequest, user=Depends(current_user)):
    if payload.revised_label not in CLASS_ORDER:
        raise HTTPException(status_code=422, detail="Unknown revised label")
    if payload.status not in {"completed", "questionable"}:
        raise HTTPException(status_code=422, detail="Unknown review status")
    if payload.diagnostic_confidence not in {"low", "medium", "high"}:
        raise HTTPException(status_code=422, detail="Unknown diagnostic confidence")
    flags = sorted(set(payload.quality_flags))
    if not set(flags).issubset(QUALITY_FLAGS):
        raise HTTPException(status_code=422, detail="Unknown quality flag")
    now = utc_now()
    flags_json = json.dumps(flags, ensure_ascii=True)
    with transaction() as connection:
        if not connection.execute("SELECT 1 FROM slides WHERE slide_id = ?", (slide_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Slide not found")
        existing = connection.execute(
            "SELECT id, created_at FROM reviews WHERE slide_id = ? AND user_id = ?",
            (slide_id, user["id"]),
        ).fetchone()
        if existing:
            review_id = existing["id"]
            connection.execute(
                """
                UPDATE reviews SET revised_label=?, status=?, diagnostic_confidence=?,
                    quality_flags=?, notes=?, updated_at=? WHERE id=?
                """,
                (payload.revised_label, payload.status, payload.diagnostic_confidence, flags_json, payload.notes.strip(), now, review_id),
            )
        else:
            cursor = connection.execute(
                """
                INSERT INTO reviews(slide_id, user_id, revised_label, status, diagnostic_confidence,
                                    quality_flags, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (slide_id, user["id"], payload.revised_label, payload.status, payload.diagnostic_confidence, flags_json, payload.notes.strip(), now, now),
            )
            review_id = cursor.lastrowid
        connection.execute(
            """
            INSERT INTO review_history(review_id, slide_id, user_id, revised_label, status,
                                       diagnostic_confidence, quality_flags, notes, changed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (review_id, slide_id, user["id"], payload.revised_label, payload.status,
             payload.diagnostic_confidence, flags_json, payload.notes.strip(), now),
        )
    return {"ok": True, "review_id": review_id, "updated_at": now}


@app.get("/api/review-progress")
async def review_progress(user=Depends(current_user)):
    connection = connect()
    try:
        row = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM slides) AS total,
                (SELECT COUNT(*) FROM slides WHERE wrong_configurations = 14) AS core_total,
                (SELECT COUNT(*) FROM reviews WHERE user_id = ?) AS mine,
                (SELECT COUNT(*) FROM reviews WHERE user_id = ? AND status = 'questionable') AS mine_questionable,
                (SELECT COUNT(DISTINCT slide_id) FROM reviews) AS globally_reviewed
                ,(SELECT COUNT(*) FROM challenge_case_reviews WHERE user_id=? AND status='submitted') AS challenge_submitted
                ,(SELECT COUNT(*) FROM challenge_slide_resolution WHERE resolution_status='retained_challenge') AS retained_challenges
                ,(SELECT COUNT(*) FROM challenge_slide_resolution WHERE resolution_status='pending_adjudication') AS pending_adjudication
                ,(SELECT COUNT(*) FROM challenge_slide_resolution WHERE resolution_status='eligible_for_exclusion') AS eligible_for_exclusion
            """,
            (user["id"], user["id"], user["id"]),
        ).fetchone()
    finally:
        connection.close()
    return dict(row)


@app.get("/api/slides/{slide_id}/dzi")
async def slide_dzi(slide_id: str, user=Depends(current_user)):
    slide = require_wsi(slide_id)
    try:
        content = await asyncio.to_thread(get_dzi, slide["wsi_path"], slide_id)
        return Response(content, media_type="application/xml", headers={"Cache-Control": "private, max-age=3600"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to open WSI: {exc}") from exc


@app.get("/api/slides/{slide_id}/tiles/{level}/{address}.jpeg")
async def slide_tile(slide_id: str, level: int, address: str, user=Depends(current_user)):
    slide = require_wsi(slide_id)
    try:
        column_text, row_text = address.split("_", 1)
        content = await asyncio.to_thread(
            get_tile,
            slide["wsi_path"],
            level,
            (int(column_text), int(row_text)),
            slide_id,
        )
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})
    except (ValueError, IndexError):
        raise HTTPException(status_code=404, detail="Tile not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read WSI tile: {exc}") from exc


@app.get("/api/slides/{slide_id}/thumbnail")
async def slide_thumbnail(slide_id: str, user=Depends(current_user)):
    slide = require_wsi(slide_id)
    try:
        content = await asyncio.to_thread(
            get_thumbnail,
            slide["wsi_path"],
            slide_id,
            1200,
            is_priority_hp_slide(slide),
        )
        return Response(content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to create thumbnail: {exc}") from exc


@app.get("/api/slides/{slide_id}/metadata")
async def slide_metadata(slide_id: str, user=Depends(current_user)):
    slide = require_wsi(slide_id)
    try:
        metadata = await asyncio.to_thread(
            get_metadata,
            slide["wsi_path"],
            slide_id,
            is_priority_hp_slide(slide),
        )
        return {
            **metadata,
            "slide_id": slide_id,
            "format": slide["wsi_format"],
            "cache_status": wsi_service.cache_status(slide_id, slide["wsi_path"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read WSI metadata: {exc}") from exc


@app.post("/api/slides/{slide_id}/warm", status_code=202)
async def warm_slide(slide_id: str, user=Depends(current_user)):
    slide = require_wsi(slide_id)
    if slide["wsi_format"] != "isyntax":
        return {"slide_id": slide_id, "status": "ready"}
    warm_priority = 0 if is_priority_hp_slide(slide) else 10
    status_value = warm_manager.submit(
        slide_id,
        slide["wsi_path"],
        priority=warm_priority,
        pinned=is_priority_hp_slide(slide),
    )
    return {"slide_id": slide_id, "status": status_value}


@app.get("/api/cache/status")
async def cache_status(user=Depends(admin_user)):
    return wsi_service.stats()


def csv_response(filename, headers, rows):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        content=output.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/exports/reviews.csv")
async def export_reviews(user=Depends(current_user)):
    connection = connect()
    try:
        rows = connection.execute(
            """
            SELECT r.slide_id, s.original_label, u.username, u.display_name, r.revised_label,
                   r.status, r.diagnostic_confidence, r.quality_flags, r.notes,
                   r.created_at, r.updated_at
            FROM reviews r JOIN slides s ON s.slide_id=r.slide_id JOIN users u ON u.id=r.user_id
            ORDER BY r.slide_id, u.username
            """
        ).fetchall()
    finally:
        connection.close()
    headers = list(rows[0].keys()) if rows else [
        "slide_id", "original_label", "username", "display_name", "revised_label", "status",
        "diagnostic_confidence", "quality_flags", "notes", "created_at", "updated_at",
    ]
    return csv_response("challenge_reviews.csv", headers, [tuple(row) for row in rows])


def consensus_rows(min_reviews=1):
    connection = connect()
    try:
        slides = connection.execute(
            """
            SELECT s.slide_id, s.original_label, s.consensus_wrong_class, COUNT(r.id) AS review_count
            FROM slides s LEFT JOIN reviews r ON r.slide_id=s.slide_id
            GROUP BY s.slide_id HAVING COUNT(r.id) >= ? ORDER BY s.slide_id
            """,
            (min_reviews,),
        ).fetchall()
        results = []
        for slide in slides:
            votes = connection.execute(
                "SELECT revised_label, COUNT(*) AS n FROM reviews WHERE slide_id=? GROUP BY revised_label ORDER BY n DESC, revised_label",
                (slide["slide_id"],),
            ).fetchall()
            top_count = votes[0]["n"]
            winners = [row["revised_label"] for row in votes if row["n"] == top_count]
            suggested = winners[0] if len(winners) == 1 else ""
            vote_summary = "; ".join(f"{row['revised_label']}:{row['n']}" for row in votes)
            results.append(
                (
                    slide["slide_id"], slide["original_label"], slide["consensus_wrong_class"],
                    slide["review_count"], suggested, top_count / slide["review_count"],
                    int(len(winners) > 1), vote_summary,
                )
            )
    finally:
        connection.close()
    return results


@app.get("/api/exports/consensus.csv")
async def export_consensus(user=Depends(current_user)):
    headers = ["slide_id", "original_label", "model_consensus_label", "review_count", "suggested_label", "review_agreement", "tie", "review_vote_summary"]
    return csv_response("challenge_review_consensus.csv", headers, consensus_rows())


@app.get("/api/exports/manifest.csv")
async def export_manifest(min_reviews: int = Query(2, ge=1, le=20), user=Depends(current_user)):
    headers = ["slide_id", "original_label", "model_consensus_label", "review_count", "suggested_label", "review_agreement", "tie", "review_vote_summary"]
    return csv_response(f"challenge_revised_manifest_min{min_reviews}.csv", headers, consensus_rows(min_reviews))


@app.get("/api/admin/users")
async def list_users(user=Depends(admin_user)):
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT id, username, display_name, role, active, created_at FROM users ORDER BY username"
        ).fetchall()
    finally:
        connection.close()
    return [dict(row) for row in rows]


@app.post("/api/admin/users", status_code=201)
async def create_user(payload: UserRequest, user=Depends(admin_user)):
    if payload.role not in {"reviewer", "admin"}:
        raise HTTPException(status_code=422, detail="Unknown role")
    try:
        with transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users(username, password_hash, display_name, role, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (payload.username.strip(), hash_password(payload.password), payload.display_name.strip(), payload.role, utc_now()),
            )
    except Exception as exc:
        if "UNIQUE constraint" in str(exc):
            raise HTTPException(status_code=409, detail="Username already exists") from exc
        raise
    return {"id": cursor.lastrowid}


@app.put("/api/admin/users/{user_id}/password")
async def reset_password(user_id: int, payload: PasswordRequest, user=Depends(admin_user)):
    with transaction() as connection:
        cursor = connection.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (hash_password(payload.password), user_id)
        )
        if cursor.rowcount != 1:
            raise HTTPException(status_code=404, detail="User not found")
        connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    return {"ok": True}


@app.put("/api/admin/users/{user_id}/active")
async def set_user_active(user_id: int, active: bool, user=Depends(admin_user)):
    if user_id == user["id"] and not active:
        raise HTTPException(status_code=422, detail="Cannot deactivate your own account")
    with transaction() as connection:
        cursor = connection.execute("UPDATE users SET active=? WHERE id=?", (int(active), user_id))
        if cursor.rowcount != 1:
            raise HTTPException(status_code=404, detail="User not found")
        if not active:
            connection.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    return {"ok": True}
