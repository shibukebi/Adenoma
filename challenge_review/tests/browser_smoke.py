from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
import re

from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright


BASE_URL = os.environ.get("CHALLENGE_REVIEW_URL", "http://127.0.0.1:8765")
USERNAME = os.environ.get("CHALLENGE_REVIEW_USER", "admin")
PASSWORD = os.environ["CHALLENGE_REVIEW_PASSWORD"]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "browser"


def mock_tile() -> bytes:
    image = Image.new("RGB", (254, 254), "#f1ddd8")
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 28, 224, 208), fill="#c98292", outline="#8f5367", width=5)
    draw.ellipse((72, 64, 186, 175), fill="#ead0c5", outline="#a96879", width=4)
    output = BytesIO()
    image.save(output, "JPEG", quality=88)
    return output.getvalue()


MOCK_TILE = mock_tile()


def install_wsi_mock(page: Page, objective_power: float, requested_levels: list[int]) -> None:
    page.route(
        "**/api/slides/*/metadata",
        lambda route: route.fulfill(
            json={
                "width": 4096,
                "height": 2048,
                "level_count": 13,
                "tile_size": 254,
                "tile_overlap": 1,
                "objective_power": objective_power,
                "mpp_x": 0.25,
                "format": "svs",
            }
        ),
    )
    def tile_handler(route):
        match = re.search(r"/tiles/(\d+)/", route.request.url)
        if match:
            requested_levels.append(int(match.group(1)))
        route.fulfill(body=MOCK_TILE, content_type="image/jpeg")

    page.route("**/api/slides/*/tiles/**", tile_handler)


def install_challenge_mock(page: Page) -> None:
    review = {
        "id": None, "version": 0, "lesion_diagnosis": "SSL", "hgd_status": "Present",
        "label_action": "confirm_original", "challenge_disposition": "retain_challenge",
        "primary_challenge": "", "primary_challenge_other": "",
        "modifiers": [], "difficulty_note": "", "expert_confidence": 3,
        "no_localizable_evidence": 0, "no_roi_reason": "", "no_roi_reason_other": "",
        "status": "draft", "locked": False, "rois": [], "peer_summary": None,
    }
    taxonomy = {
        "lesion_diagnoses": ["HP", "SSL", "TSA", "TA", "TVA", "IP", "USA", "Other", "Ambiguous"],
        "hgd_statuses": ["Absent", "Present", "Not assessable", "Uncertain/conflicting"],
        "label_actions": ["confirm_original", "correct_original", "remains_ambiguous"],
        "challenge_dispositions": ["retain_challenge", "pending_label_adjudication", "exclude_label_error"],
        "primary_challenges": ["HP vs SSL", "Focal HGD", "Other differential"],
        "difficulty_modifiers": ["focal_evidence", "competing_morphology"],
        "no_roi_reasons": ["global_architecture_required", "other"],
        "diagnostic_roles": ["discriminative", "confirmatory"],
        "evidence_strengths": ["weak", "strong", "decisive"],
        "general_evidence": ["surface_serration", "other"],
        "evidence_by_challenge": {"HP vs SSL": ["basal_crypt_dilation", "horizontal_crypt_growth"]},
        "directions_by_challenge": {"HP vs SSL": ["supports_hp", "supports_ssl", "supports_both"]},
    }
    page.route("**/api/challenge-taxonomy", lambda route: route.fulfill(json=taxonomy))
    models = [("CLAM-SB", feature) for feature in ("2p5x", "5x", "10x", "20x")]
    models += [("TransMIL", feature) for feature in ("2p5x", "5x", "10x", "20x")]
    models += [("DSMIL", feature) for feature in ("2p5x", "5x", "10x", "20x")]
    models += [("MIST", feature) for feature in ("2p5x_5x", "5x_10x")]
    detail = {
        "slide": {
            "slide_id": "640659 1", "original_label": "SSLD", "pathology_type": "Sessile serrated adenoma",
            "grade": "high", "source": "yx", "cv_fold": 4, "wsi_path": "/mock/640659 1.svs",
            "wsi_format": "svs", "wsi_available": 1, "hardness_rank": 1,
            "wrong_configurations": 14, "total_configurations": 14, "wrong_pct": 100.0,
            "consensus_wrong_class": "SSL", "consensus_wrong_count": 14,
            "mean_wrong_confidence": 0.8, "max_wrong_confidence": 0.95,
        },
        "predictions": [
            {
                "model": model, "feature": feature, "configuration": f"{model} {feature}",
                "predicted_label": "SSL", "confidence": 0.653, "second_predicted_label": "USA",
                "second_confidence": 0.196, "is_correct": 0,
            }
            for model, feature in models
        ],
        "review": None,
        "peer_summary": None,
    }
    page.route(re.compile(r".*/api/slides/640659%201$"), lambda route: route.fulfill(json=detail))
    page.route(
        "**/api/slides/*/challenge-review",
        lambda route: route.fulfill(json={"slide_id": "640659 1", "eligible": True, "review": review}),
    )

    def draft_handler(route):
        payload = route.request.post_data_json
        review.update(payload)
        review["id"] = 900
        review["version"] += 1
        route.fulfill(json=review)

    page.route("**/api/slides/*/challenge-review/draft", draft_handler)

    def create_roi_handler(route):
        payload = route.request.post_data_json
        review["version"] += 1
        roi = {
            "id": 901, "review_id": 900, "slide_id": "640659 1", "display_order": 1,
            **{key: payload.get(key) for key in (
                "x_level0", "y_level0", "width_level0", "height_level0", "mpp_x", "mpp_y",
                "physical_width_um", "physical_height_um", "viewer_zoom", "diagnostic_role",
                "differential_direction", "evidence_strength", "note",
            )},
            "version": 1, "evidence": payload.get("evidence", []),
        }
        review["rois"] = [roi]
        route.fulfill(status=201, json={"review_version": review["version"], "roi": roi})

    page.route("**/api/challenge-reviews/*/rois", create_roi_handler)

    def update_roi_handler(route):
        payload = route.request.post_data_json
        review["version"] += 1
        roi = review["rois"][0]
        roi.update(payload)
        roi["version"] += 1
        route.fulfill(json={"review_version": review["version"], "roi": roi})

    page.route("**/api/challenge-rois/*", update_roi_handler)


def assert_no_overlap(page: Page, first: str, second: str) -> None:
    a = page.locator(first).bounding_box()
    b = page.locator(second).bounding_box()
    assert a and b, f"Missing layout box for {first} or {second}"
    overlaps = not (
        a["x"] + a["width"] <= b["x"]
        or b["x"] + b["width"] <= a["x"]
        or a["y"] + a["height"] <= b["y"]
        or b["y"] + b["height"] <= a["y"]
    )
    assert not overlaps, f"Unexpected overlap between {first} and {second}"


def capture(page: Page, width: int, height: int, objective_power: float) -> None:
    browser_errors: list[str] = []
    requested_levels: list[int] = []
    page.on("pageerror", lambda error: browser_errors.append(str(error)))
    install_wsi_mock(page, objective_power, requested_levels)
    install_challenge_mock(page)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.locator("#filter-search").evaluate("(element) => { element.value = '640659 1'; }")
    page.locator("#login-username").fill(USERNAME)
    page.locator("#login-password").fill(PASSWORD)
    page.locator("#login-form button[type='submit']").click()
    page.locator("#app-view").wait_for(state="visible")

    page.locator(".slide-row", has_text="640659 1").wait_for(timeout=30_000)
    page.locator("#slide-id", has_text="640659 1").wait_for(timeout=30_000)
    page.locator("#prediction-body tr").first.wait_for(timeout=30_000)
    try:
        canvas = page.locator(".openseadragon-canvas canvas").first
        canvas.wait_for(timeout=90_000)
        page.locator("#viewer-loading").wait_for(state="hidden", timeout=90_000)
        page.wait_for_timeout(800)
    except Exception:
        page.screenshot(path=OUTPUT_DIR / f"failure_{width}x{height}.png", full_page=False)
        viewer_error = page.locator("#viewer-error span").text_content()
        raise AssertionError(
            f"WSI canvas did not open; viewer_error={viewer_error!r}; page_errors={browser_errors!r}"
        )

    assert page.locator("#prediction-body tr").count() == 14
    assert page.locator("#review-label option").count() == 11
    first_prediction = page.locator("#prediction-body tr").first
    assert first_prediction.locator("td").nth(2).text_content().strip() == "SSL65.3%"
    assert first_prediction.locator("td").nth(3).text_content().strip() == "USA19.6%"
    canvas_image = Image.open(BytesIO(canvas.screenshot())).convert("RGB")
    extrema = canvas_image.getextrema()
    assert any(high - low > 20 for low, high in extrema), "WSI canvas is visually blank"
    assert_no_overlap(page, ".queue-pane", "#viewer-pane")
    assert_no_overlap(page, "#viewer-pane", ".review-pane")
    assert_no_overlap(page, ".topbar", ".workspace")

    assert page.locator("#challenge-review-form").is_visible()
    pending_disposition = page.locator("#challenge-disposition option[value='pending_label_adjudication']")
    assert pending_disposition.evaluate("option => option.disabled")
    page.locator("#challenge-hgd").select_option("Absent")
    assert not pending_disposition.evaluate("option => option.disabled")
    page.locator("#challenge-disposition").select_option("pending_label_adjudication")
    assert page.locator("#challenge-label-action").input_value() == "correct_original"
    assert "模型共识一致" in page.locator("#challenge-disposition-hint").text_content()
    page.locator("#challenge-primary").select_option("HP vs SSL")
    page.wait_for_timeout(900)
    page.locator("#roi-draw-toggle").click()
    viewer_box = page.locator("#viewer").bounding_box()
    assert viewer_box
    page.mouse.move(viewer_box["x"] + 180, viewer_box["y"] + 210)
    page.mouse.down()
    page.mouse.move(viewer_box["x"] + 390, viewer_box["y"] + 360, steps=5)
    page.mouse.up()
    page.locator(".roi-overlay").wait_for(timeout=10_000)
    page.locator(".roi-editor").wait_for(timeout=10_000)
    assert page.locator("#roi-count").text_content() == "1 / 3"
    roi_box = page.locator(".roi-overlay").bounding_box()
    assert roi_box and roi_box["width"] > 150 and roi_box["height"] > 100
    page.locator(".roi-editor input[data-field='evidence'][value='basal_crypt_dilation']").check()
    page.locator(".roi-editor select[data-field='diagnostic_role']").select_option("discriminative")
    page.locator(".roi-editor select[data-field='differential_direction']").select_option("supports_ssl")
    page.locator(".roi-editor select[data-field='evidence_strength']").select_option("decisive")
    page.wait_for_timeout(500)
    page.screenshot(path=OUTPUT_DIR / f"roi_{width}x{height}.png", full_page=False)

    page.locator("#magnifier-toggle").click()
    assert page.locator("#magnifier-controls").is_visible()
    magnifier_40 = page.locator("#magnifier-controls button[data-magnification='40']")
    if objective_power >= 40:
        assert not magnifier_40.is_disabled()
        magnifier_40.click()
    else:
        assert magnifier_40.is_disabled()

    viewer_box = page.locator("#viewer").bounding_box()
    assert viewer_box
    page.mouse.move(viewer_box["x"] + viewer_box["width"] / 2, viewer_box["y"] + viewer_box["height"] / 2)
    lens_panel = page.locator("#magnifier-panel:not(.inactive)")
    lens_panel.wait_for(timeout=10_000)
    page.wait_for_timeout(250)
    page.locator("#magnifier-loading").wait_for(state="hidden", timeout=10_000)
    lens_image = Image.open(BytesIO(lens_panel.screenshot())).convert("RGB")
    assert any(high - low > 20 for low, high in lens_image.getextrema())
    assert max(requested_levels) == 12
    expected_badge = "40×" if objective_power >= 40 else "20×"
    assert page.locator("#magnifier-badge").text_content() == expected_badge
    page.screenshot(path=OUTPUT_DIR / f"magnifier_{width}x{height}.png", full_page=False)

    for x, y in ((5, 5), (viewer_box["width"] - 5, viewer_box["height"] - 5)):
        page.mouse.move(viewer_box["x"] + x, viewer_box["y"] + y)
        page.wait_for_timeout(100)
        lens_box = page.locator("#magnifier-panel").bounding_box()
        assert lens_box
        assert lens_box["x"] >= viewer_box["x"]
        assert lens_box["y"] >= viewer_box["y"]
        assert lens_box["x"] + lens_box["width"] <= viewer_box["x"] + viewer_box["width"]
        assert lens_box["y"] + lens_box["height"] <= viewer_box["y"] + viewer_box["height"]

    page.keyboard.press("Escape")
    assert page.locator("#magnifier-controls").is_hidden()
    assert "inactive" in (page.locator("#magnifier-panel").get_attribute("class") or "")

    page.screenshot(path=OUTPUT_DIR / f"review_{width}x{height}.png", full_page=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for width, height, objective_power in ((1440, 900, 40.0), (1920, 1080, 20.0)):
                page = browser.new_page(viewport={"width": width, "height": height})
                capture(page, width, height, objective_power)
                page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
