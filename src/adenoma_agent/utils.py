import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_payload(payload):
    return sha1_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def append_jsonl(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def bbox_to_dict(x1, y1, x2, y2):
    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def bbox_width(bbox):
    return max(0, int(bbox["x2"]) - int(bbox["x1"]))


def bbox_height(bbox):
    return max(0, int(bbox["y2"]) - int(bbox["y1"]))


def bbox_area(bbox):
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_center(bbox):
    return (
        int(round((int(bbox["x1"]) + int(bbox["x2"])) / 2.0)),
        int(round((int(bbox["y1"]) + int(bbox["y2"])) / 2.0)),
    )


def bbox_intersection(a, b):
    x1 = max(int(a["x1"]), int(b["x1"]))
    y1 = max(int(a["y1"]), int(b["y1"]))
    x2 = min(int(a["x2"]), int(b["x2"]))
    y2 = min(int(a["y2"]), int(b["y2"]))
    if x2 <= x1 or y2 <= y1:
        return bbox_to_dict(0, 0, 0, 0)
    return bbox_to_dict(x1, y1, x2, y2)


def bbox_overlap_ratio(a, b):
    intersection = bbox_area(bbox_intersection(a, b))
    denom = max(1, min(bbox_area(a), bbox_area(b)))
    return float(intersection) / float(denom)


def clamp_bbox(bbox, slide_dimensions_level0):
    slide_w, slide_h = slide_dimensions_level0
    return bbox_to_dict(
        clamp(int(bbox["x1"]), 0, slide_w),
        clamp(int(bbox["y1"]), 0, slide_h),
        clamp(int(bbox["x2"]), 0, slide_w),
        clamp(int(bbox["y2"]), 0, slide_h),
    )


def map_bbox_thumb_to_level0(bbox_thumb, thumbnail_size, slide_dimensions_level0):
    thumb_w, thumb_h = thumbnail_size
    slide_w, slide_h = slide_dimensions_level0
    scale_x = float(slide_w) / float(thumb_w)
    scale_y = float(slide_h) / float(thumb_h)
    mapped = bbox_to_dict(
        round(float(bbox_thumb["x1"]) * scale_x),
        round(float(bbox_thumb["y1"]) * scale_y),
        round(float(bbox_thumb["x2"]) * scale_x),
        round(float(bbox_thumb["y2"]) * scale_y),
    )
    return clamp_bbox(mapped, slide_dimensions_level0)


def normalized_point(x, y, slide_dimensions_level0):
    slide_w, slide_h = slide_dimensions_level0
    return {
        "x": round(float(x) / float(max(1, slide_w)), 6),
        "y": round(float(y) / float(max(1, slide_h)), 6),
    }


def clamp_center_point(x, y, region_size, slide_dimensions_level0):
    slide_w, slide_h = slide_dimensions_level0
    half = int(region_size // 2)
    return (
        clamp(int(x), half, max(half, slide_w - half)),
        clamp(int(y), half, max(half, slide_h - half)),
    )


def bbox_from_center(x, y, region_size):
    half = int(region_size // 2)
    return bbox_to_dict(x - half, y - half, x + half, y + half)


def binary_label_from_type(label, positive_label):
    if label is None:
        return None
    return 1 if label == positive_label else 0


def membership_label_from_type(label, positive_labels):
    if label is None:
        return None
    return 1 if label in set(positive_labels) else 0


def binary_label_from_grade(grade, positive_grades):
    if grade is None:
        return None
    return 1 if grade in set(positive_grades) else 0


def connected_components_from_grid(active_grid):
    height = len(active_grid)
    width = len(active_grid[0]) if height else 0
    visited = [[False for _ in range(width)] for _ in range(height)]
    components = []
    for y in range(height):
        for x in range(width):
            if visited[y][x] or not active_grid[y][x]:
                continue
            stack = [(x, y)]
            visited[y][x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if visited[ny][nx] or not active_grid[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        stack.append((nx, ny))
            components.append(cells)
    return components


def env_with_cuda_visible_devices(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return {"CUDA_VISIBLE_DEVICES": value}


def run_command(command, timeout=None, cwd=None, env_overrides=None):
    started = time.time()
    env = None
    if env_overrides:
        env = os.environ.copy()
        for key, value in env_overrides.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    latency_ms = int(round((time.time() - started) * 1000.0))
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "latency_ms": latency_ms,
        "command": command,
    }


def copy_if_exists(src, dst):
    import shutil

    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return False
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
    return True


def write_text(path, text):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def load_env_file(path):
    payload = {}
    path = Path(path)
    if not path.exists():
        return payload
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            payload[key.strip()] = value.strip()
    return payload


def getenv_or_default(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value
