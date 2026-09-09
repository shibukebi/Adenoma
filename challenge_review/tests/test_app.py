import os
from pathlib import Path
import tempfile


TEST_ROOT = Path(tempfile.mkdtemp(prefix="challenge-review-tests-"))
os.environ["CHALLENGE_REVIEW_DB"] = str(TEST_ROOT / "test.sqlite3")

import httpx
import pytest

import challenge_review.main as main_module
from challenge_review.db import initialize_database, transaction, utc_now
from challenge_review.main import app
from challenge_review.security import hash_password


def seed():
    initialize_database()
    with transaction() as connection:
        connection.execute("DELETE FROM challenge_review_history")
        connection.execute("DELETE FROM challenge_roi_evidence")
        connection.execute("DELETE FROM challenge_rois")
        connection.execute("DELETE FROM challenge_review_modifiers")
        connection.execute("DELETE FROM challenge_case_reviews")
        connection.execute("DELETE FROM review_history")
        connection.execute("DELETE FROM reviews")
        connection.execute("DELETE FROM sessions")
        connection.execute("DELETE FROM predictions")
        connection.execute("DELETE FROM slides")
        connection.execute("DELETE FROM users")
        connection.executemany(
            "INSERT INTO users(username,password_hash,display_name,role,active,created_at) VALUES (?,?,?,?,1,?)",
            [
                ("admin", hash_password("admin-password-123"), "Admin", "admin", utc_now()),
                ("reviewer", hash_password("review-password-123"), "Reviewer", "reviewer", utc_now()),
            ],
        )
        slides = [
            ("slide one", "SSLD", "Sessile serrated lesion", "high", "yx", 1, None, "svs", 0, 1, 14, 14, 100.0, "SSL", 14, .8, .95, "SSL:14", utc_now()),
            ("slide-two", "HP", "Hyperplastic polyp", "low", "hp", 2, None, "isyntax", 0, 2, 6, 14, 42.8, "IP", 6, .6, .8, "IP:6; HP:8", utc_now()),
        ]
        connection.executemany(
            """
            INSERT INTO slides(slide_id,original_label,pathology_type,grade,source,cv_fold,wsi_path,wsi_format,
                wsi_available,hardness_rank,wrong_configurations,total_configurations,wrong_pct,
                consensus_wrong_class,consensus_wrong_count,mean_wrong_confidence,max_wrong_confidence,
                all_prediction_summary,imported_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            slides,
        )
        predictions = []
        for model, features in [("CLAM-SB", ["2p5x", "5x", "10x", "20x"]), ("TransMIL", ["2p5x", "5x", "10x", "20x"]), ("DSMIL", ["2p5x", "5x", "10x", "20x"]), ("MIST", ["2p5x_5x", "5x_10x"])]:
            for feature in features:
                predictions.append(("slide one", model, feature, f"{model} {feature}", "SSL", .8, "HP", .15, 0))
        connection.executemany(
            """INSERT INTO predictions(
                slide_id,model,feature,configuration,predicted_label,confidence,
                second_predicted_label,second_confidence,is_correct
            ) VALUES (?,?,?,?,?,?,?,?,?)""",
            predictions,
        )


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def login(client, username, password):
    response = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return response


@pytest.mark.anyio
async def test_auth_sort_review_history_and_exports():
    seed()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/slides")).status_code == 401
        assert (await client.post("/api/auth/login", json={"username": "admin", "password": "wrong"})).status_code == 401
        await login(client, "admin", "admin-password-123")

        listing = (await client.get("/api/slides", params={"core_only": True})).json()
        assert listing["total"] == 1
        assert listing["items"][0]["slide_id"] == "slide one"

        detail = (await client.get("/api/slides/slide%20one")).json()
        assert len(detail["predictions"]) == 14
        assert detail["predictions"][0]["second_predicted_label"] == "HP"
        assert detail["predictions"][0]["second_confidence"] == .15
        assert detail["peer_summary"] is None

        review = {
            "revised_label": "SSLD",
            "status": "completed",
            "diagnostic_confidence": "high",
            "quality_flags": ["label_dispute"],
            "notes": "中文备注",
        }
        assert (await client.put("/api/slides/slide%20one/review", json=review)).status_code == 200
        review["notes"] = "第二次修改"
        assert (await client.put("/api/slides/slide%20one/review", json=review)).status_code == 200

        with transaction() as connection:
            assert connection.execute("SELECT COUNT(*) FROM review_history").fetchone()[0] == 2

        detail = (await client.get("/api/slides/slide%20one")).json()
        assert detail["review"]["revised_label"] == "SSLD"
        assert detail["peer_summary"]["review_count"] == 1
        export = await client.get("/api/exports/reviews.csv")
        assert export.status_code == 200
        assert "中文备注" not in export.content.decode("utf-8-sig")
        assert "第二次修改" in export.content.decode("utf-8-sig")


@pytest.mark.anyio
async def test_admin_creates_user_and_reviewer_cannot_access_admin():
    seed()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client, "admin", "admin-password-123")
        response = await client.post(
            "/api/admin/users",
            json={"username": "pathologist", "display_name": "Pathologist", "password": "strong-password-123", "role": "reviewer"},
        )
        assert response.status_code == 201
        await client.post("/api/auth/logout")
        await login(client, "reviewer", "review-password-123")
        assert (await client.get("/api/admin/users")).status_code == 403


@pytest.mark.anyio
async def test_warm_endpoint_and_admin_cache_status(monkeypatch):
    seed()
    with transaction() as connection:
        connection.execute(
            """
            UPDATE slides SET wsi_path='/readonly/slide.isyntax',wsi_format='isyntax',
                wsi_available=1,source='hp' WHERE slide_id='slide one'
            """
        )
    monkeypatch.setattr(main_module.warm_manager, "submit", lambda *args, **kwargs: "queued")
    monkeypatch.setattr(main_module.wsi_service, "stats", lambda: {"bytes": 123, "queue": {}})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client, "reviewer", "review-password-123")
        response = await client.post("/api/slides/slide%20one/warm")
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert (await client.get("/api/cache/status")).status_code == 403

        await client.post("/api/auth/logout")
        await login(client, "admin", "admin-password-123")
        status_response = await client.get("/api/cache/status")
        assert status_response.status_code == 200
        assert status_response.json()["bytes"] == 123


def challenge_payload(version=0, primary="HP vs SSL", disposition="retain_challenge"):
    return {
        "version": version,
        "lesion_diagnosis": "SSL",
        "hgd_status": "Absent",
        "label_action": "correct_original",
        "challenge_disposition": disposition,
        "primary_challenge": primary,
        "primary_challenge_other": "",
        "modifiers": ["focal_evidence", "competing_morphology"],
        "difficulty_note": "Basal change is focal",
        "expert_confidence": 4,
        "no_localizable_evidence": False,
        "no_roi_reason": "",
        "no_roi_reason_other": "",
    }


def roi_payload(review_version, version=0, x=100):
    return {
        "review_version": review_version,
        "version": version,
        "x_level0": x,
        "y_level0": 200,
        "width_level0": 800,
        "height_level0": 600,
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "physical_width_um": 200,
        "physical_height_um": 150,
        "viewer_zoom": 0.5,
        "diagnostic_role": "discriminative",
        "differential_direction": "supports_ssl",
        "evidence_strength": "decisive",
        "evidence": [{"evidence_code": "basal_crypt_dilation", "evidence_other": ""}],
        "note": "Key crypt base",
    }


@pytest.mark.anyio
async def test_challenge_roi_workflow_versions_limits_exports_and_locking():
    seed()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client, "reviewer", "review-password-123")
        assert (await client.get("/api/challenge-taxonomy")).status_code == 200
        assert (await client.get("/api/slides/slide-two/challenge-review")).status_code == 403

        draft_response = await client.put(
            "/api/slides/slide%20one/challenge-review/draft", json=challenge_payload()
        )
        assert draft_response.status_code == 200
        review = draft_response.json()
        assert review["status"] == "draft"
        assert review["version"] == 1
        assert sorted(review["modifiers"]) == ["competing_morphology", "focal_evidence"]

        stale = await client.put(
            "/api/slides/slide%20one/challenge-review/draft", json=challenge_payload(version=0)
        )
        assert stale.status_code == 409

        roi_ids = []
        for index in range(3):
            response = await client.post(
                f"/api/challenge-reviews/{review['id']}/rois",
                json=roi_payload(review["version"], x=100 + index * 1000),
            )
            assert response.status_code == 201
            data = response.json()
            review["version"] = data["review_version"]
            roi_ids.append(data["roi"]["id"])
        fourth = await client.post(
            f"/api/challenge-reviews/{review['id']}/rois", json=roi_payload(review["version"], x=5000)
        )
        assert fourth.status_code == 422

        reordered = await client.put(
            f"/api/challenge-reviews/{review['id']}/rois/order",
            json={"version": review["version"], "ordered_roi_ids": [roi_ids[2], roi_ids[0], roi_ids[1]]},
        )
        assert reordered.status_code == 200
        review = reordered.json()
        assert [roi["id"] for roi in review["rois"]] == [roi_ids[2], roi_ids[0], roi_ids[1]]

        submitted = await client.post(
            "/api/slides/slide%20one/challenge-review/submit",
            json=challenge_payload(version=review["version"]),
        )
        assert submitted.status_code == 200
        review = submitted.json()
        assert review["status"] == "submitted"
        assert review["peer_summary"]["review_count"] == 1

        csv_export = await client.get("/api/exports/challenge-rois.csv")
        geojson_export = await client.get("/api/exports/challenge-rois.geojson")
        assert csv_export.status_code == 200
        assert "basal_crypt_dilation" in csv_export.content.decode("utf-8-sig")
        assert len(geojson_export.json()["features"]) == 3

        await client.post("/api/auth/logout")
        await login(client, "admin", "admin-password-123")
        locked = await client.post(
            f"/api/admin/challenge-reviews/{review['id']}/lock", json={"version": review["version"]}
        )
        assert locked.status_code == 200
        locked_review = locked.json()
        assert locked_review["locked"] is True

        await client.post("/api/auth/logout")
        await login(client, "reviewer", "review-password-123")
        update = roi_payload(locked_review["version"], version=1)
        assert (await client.put(f"/api/challenge-rois/{roi_ids[0]}", json=update)).status_code == 423


@pytest.mark.anyio
async def test_challenge_no_roi_and_submit_validation():
    seed()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client, "reviewer", "review-password-123")
        payload = challenge_payload()
        payload.update({"no_localizable_evidence": True, "no_roi_reason": "genuine_diagnostic_ambiguity"})
        response = await client.post("/api/slides/slide%20one/challenge-review/submit", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "submitted"

        seed()
        await login(client, "reviewer", "review-password-123")
        incomplete = challenge_payload(primary="")
        assert (await client.post("/api/slides/slide%20one/challenge-review/submit", json=incomplete)).status_code == 422


@pytest.mark.anyio
async def test_label_error_disposition_requires_two_matching_reviewers_for_exclusion():
    seed()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await login(client, "reviewer", "review-password-123")
        first = await client.post(
            "/api/slides/slide%20one/challenge-review/submit",
            json=challenge_payload(disposition="pending_label_adjudication"),
        )
        assert first.status_code == 200
        assert first.json()["peer_summary"]["resolution_status"] == "pending_adjudication"

        filtered = await client.get("/api/slides", params={"resolution_status": "pending_adjudication"})
        assert filtered.status_code == 200
        assert filtered.json()["total"] == 1

        await client.post("/api/auth/logout")
        await login(client, "admin", "admin-password-123")
        second = await client.post(
            "/api/slides/slide%20one/challenge-review/submit",
            json=challenge_payload(disposition="exclude_label_error"),
        )
        assert second.status_code == 200
        summary = second.json()["peer_summary"]
        assert summary["resolution_status"] == "eligible_for_exclusion"
        assert summary["agreed_revised_label"] == "SSL"
        assert summary["agreed_reviewer_count"] == 2

        progress = (await client.get("/api/review-progress")).json()
        assert progress["eligible_for_exclusion"] == 1
        resolution_export = await client.get("/api/exports/challenge-resolutions.csv")
        assert "eligible_for_exclusion" in resolution_export.content.decode("utf-8-sig")
