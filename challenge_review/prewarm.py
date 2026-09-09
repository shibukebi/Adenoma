#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from .db import connect
from .wsi import disk_cache


def priority_slides():
    connection = connect()
    try:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT slide_id,wsi_path,wrong_configurations,hardness_rank
                FROM slides
                WHERE source='hp' AND wsi_available=1
                  AND (wrong_configurations=14 OR hardness_rank<=200)
                ORDER BY hardness_rank ASC,slide_id ASC
                """
            )
        ]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description="Queue priority HP iSyntax overview prewarming")
    parser.add_argument("--wait", action="store_true", help="Wait and print progress until all jobs finish")
    parser.add_argument("--dry-run", action="store_true", help="Only list the selected slide count")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    slides = priority_slides()
    if len(slides) != 114:
        raise RuntimeError(f"Expected 114 priority HP slides, found {len(slides)}")
    print(f"Priority HP slides: {len(slides)}", flush=True)
    if args.dry_run:
        return

    for slide in slides:
        disk_cache.enqueue_warm(
            slide["slide_id"],
            slide["wsi_path"],
            priority=20,
            pinned=True,
        )
    print("Queued priority overview prewarming", flush=True)
    if not args.wait:
        return

    slide_ids = [slide["slide_id"] for slide in slides]
    started = time.monotonic()
    while True:
        counts = disk_cache.warm_counts(slide_ids)
        elapsed = int(time.monotonic() - started)
        print(f"elapsed={elapsed}s status={counts} cache_bytes={disk_cache.stats()['bytes']}", flush=True)
        if counts.get("queued", 0) == 0 and counts.get("warming", 0) == 0:
            break
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
