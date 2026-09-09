from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
import time

from PIL import Image
from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("CHALLENGE_REVIEW_URL", "http://127.0.0.1:8765")
PASSWORD = os.environ["CHALLENGE_REVIEW_PASSWORD"]
SLIDE_ID = "2691bae9-1a36-463d-8107-217ffe9005c9"
OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "browser" / "hp_cached_1440x900.png"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.goto(BASE_URL, wait_until="domcontentloaded")
            page.locator("#filter-search").evaluate(
                f"(element) => {{ element.value = '{SLIDE_ID}'; }}"
            )
            page.locator("#login-username").fill("admin")
            page.locator("#login-password").fill(PASSWORD)
            started = time.monotonic()
            with page.expect_response(
                lambda response: response.url.endswith(f"/api/slides/{SLIDE_ID}/metadata")
            ) as response_info:
                page.locator("#login-form button[type='submit']").click()
            metadata = response_info.value.json()
            assert metadata["cache_status"] == "ready"
            page.locator("#viewer-loading").wait_for(state="hidden", timeout=10_000)
            elapsed = time.monotonic() - started
            assert elapsed < 2.0, f"Cached HP overview took {elapsed:.2f}s"

            canvas = page.locator(".openseadragon-canvas canvas").first
            canvas.wait_for(state="visible")
            page.wait_for_timeout(500)
            image = Image.open(BytesIO(canvas.screenshot())).convert("RGB")
            assert any(high - low > 20 for low, high in image.getextrema())
            assert page.locator(".navigator").is_visible()

            before = page.locator("#zoom-indicator").text_content()
            page.locator("#zoom-in").click()
            page.wait_for_timeout(700)
            after = page.locator("#zoom-indicator").text_content()
            assert before != after

            page.locator("#magnifier-toggle").click()
            page.locator("#magnifier-controls button[data-magnification='10']").click()
            viewer_box = page.locator("#viewer").bounding_box()
            assert viewer_box
            image_point = (43072, 61955)
            pointer = page.evaluate(
                """([x, y]) => {
                    const viewer = OpenSeadragon.getViewer(document.getElementById('viewer'));
                    const viewportPoint = viewer.viewport.imageToViewportCoordinates(
                        new OpenSeadragon.Point(x, y)
                    );
                    const pixel = viewer.viewport.pixelFromPoint(viewportPoint, true);
                    return {x: pixel.x, y: pixel.y};
                }""",
                image_point,
            )
            page.mouse.move(viewer_box["x"] + pointer["x"], viewer_box["y"] + pointer["y"])
            panel = page.locator("#magnifier-panel:not(.inactive)")
            panel.wait_for(timeout=10_000)
            page.wait_for_timeout(250)
            page.locator("#magnifier-loading").wait_for(state="hidden", timeout=180_000)
            page.wait_for_timeout(500)
            assert page.locator("#magnifier-loading").is_hidden()
            lens_image = Image.open(BytesIO(panel.screenshot())).convert("RGB")
            center = lens_image.crop((48, 36, lens_image.width - 48, lens_image.height - 36))
            sample = center.resize((80, 60))
            stained = sum(
                1
                for red, green, blue in sample.getdata()
                if max(red, green, blue) - min(red, green, blue) > 18 and min(red, green, blue) < 225
            )
            assert stained / (sample.width * sample.height) > 0.01
            assert page.locator("#magnifier-badge").text_content() == "10×"
            page.screenshot(path=OUTPUT, full_page=False)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
