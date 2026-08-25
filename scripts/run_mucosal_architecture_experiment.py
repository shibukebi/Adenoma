#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from adenoma_agent.architecture_experiment import (  # noqa: E402
    ALL_OUTPUT_LABELS,
    annotation_to_targets,
    build_architecture_roi_manifest,
    build_synthetic_smoke_annotations,
    read_json,
    read_jsonl,
    safe_name,
    write_json,
    write_jsonl,
)


DEFAULT_UNI_WEIGHTS = REPO_ROOT / "models" / "UNI" / "weights" / "pytorch_model.bin"
HEAVY_RUNTIME_FLAG = "_ADENOMA_ARCHITECTURE_TORCH_READY"


def ensure_torch_runtime():
    if os.environ.get(HEAVY_RUNTIME_FLAG) == "1":
        return
    prefix = Path(sys.prefix)
    version_dir = "python{0}.{1}".format(sys.version_info.major, sys.version_info.minor)
    candidates = [
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "nvjitlink" / "lib",
        prefix / "lib" / version_dir / "site-packages" / "nvidia" / "cusparse" / "lib",
        prefix / "lib",
    ]
    new_parts = [str(path) for path in candidates if path.exists()]
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if part and part not in new_parts:
            new_parts.append(part)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_parts)
    env[HEAVY_RUNTIME_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def _device(value, torch):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _weight_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    payload = "{0}:{1}:{2}".format(path.resolve(), stat.st_size, int(stat.st_mtime))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_uni(weights_path, device, torch, timm):
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
    )
    try:
        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _transform(input_size, transforms):
    return transforms.Compose(
        [
            transforms.Resize((int(input_size), int(input_size)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _encode_items(model, items, input_size, batch_size, device, torch, transforms, Image):
    transform = _transform(input_size, transforms)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(items), max(1, int(batch_size))):
            batch_items = items[start : start + max(1, int(batch_size))]
            tensors = []
            for item in batch_items:
                with Image.open(item["image_path"]) as image:
                    tensors.append(transform(image.convert("RGB")))
            batch = torch.stack(tensors, dim=0).to(device)
            features = model(batch).to(dtype=torch.float32).detach().cpu().numpy()
            rows.append(features)
    if not rows:
        return np.zeros((0, 0), dtype=np.float16)
    return np.concatenate(rows, axis=0).astype(np.float16)


def _source_items(source, slide_rows):
    if source == "child_20x_224":
        by_key = {}
        for roi in slide_rows:
            child_rows = list(roi["children"]) + list(roi.get("context_4x4_children", []))
            for child in child_rows:
                key = child["patch_uid"]
                by_key[key] = {
                    "key": key,
                    "roi_id": roi["roi_id"],
                    "patch_uid": key,
                    "image_path": child.get("absolute_image_path") or child.get("image_path"),
                    "level0_bbox": child["level0_bbox"],
                    "relative_row": child["relative_row"],
                    "relative_col": child["relative_col"],
                }
        return [by_key[key] for key in sorted(by_key)]
    return [
        {
            "key": roi["roi_id"],
            "roi_id": roi["roi_id"],
            "patch_uid": roi["parent_patch_uid"],
            "image_path": roi["parent_image_path"],
            "level0_bbox": roi["level0_bbox"],
        }
        for roi in sorted(slide_rows, key=lambda row: row["roi_id"])
    ]


def export_embeddings(experiment_dir, weights_path, device_name="auto", batch_size=8, c_batch_size=1, resume=False):
    ensure_torch_runtime()
    import timm
    import torch
    from PIL import Image
    from torchvision import transforms

    experiment_dir = Path(experiment_dir).resolve()
    weights_path = Path(weights_path).resolve()
    roi_rows = read_jsonl(experiment_dir / "roi_manifest.jsonl")
    by_slide = {}
    for row in roi_rows:
        by_slide.setdefault(row["slide_id"], []).append(row)
    device = _device(device_name, torch)
    model = _load_uni(weights_path, device, torch, timm)
    weights_fingerprint = _weight_fingerprint(weights_path)
    sources = (
        ("child_20x_224", 224, int(batch_size)),
        ("parent_10x_224", 224, int(batch_size)),
        ("parent_20x_448", 448, int(c_batch_size)),
    )
    summary = {"device": str(device), "sources": {}, "feature_dim": None}
    for source, input_size, source_batch_size in sources:
        source_count = 0
        for slide_id, slide_rows in sorted(by_slide.items()):
            output_dir = experiment_dir / "embeddings" / source / safe_name(slide_id)
            feature_path = output_dir / "features.npy"
            index_path = output_dir / "index.jsonl"
            metadata_path = output_dir / "metadata.json"
            items = _source_items(source, slide_rows)
            if resume and feature_path.exists() and index_path.exists() and metadata_path.exists():
                metadata = read_json(metadata_path)
                reusable = (
                    metadata.get("source") == source
                    and int(metadata.get("input_size", 0)) == int(input_size)
                    and metadata.get("weights_fingerprint") == weights_fingerprint
                    and int(metadata.get("item_count", -1)) == len(items)
                )
                if reusable:
                    feature_shape = np.load(str(feature_path), mmap_mode="r").shape
                    source_count += int(feature_shape[0])
                    if len(feature_shape) == 2 and feature_shape[1]:
                        summary["feature_dim"] = int(feature_shape[1])
                    continue
            output_dir.mkdir(parents=True, exist_ok=True)
            features = _encode_items(
                model, items, input_size, source_batch_size, device, torch, transforms, Image
            )
            np.save(str(feature_path), features)
            index_rows = []
            for feature_row, item in enumerate(items):
                row = dict(item)
                row["feature_row"] = feature_row
                row["source"] = source
                index_rows.append(row)
            write_jsonl(index_path, index_rows)
            metadata = {
                "source": source,
                "slide_id": slide_id,
                "input_size": input_size,
                "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
                "item_count": len(items),
                "encoder": "UNI_vit_large_patch16_224_raw_feature",
                "weights_path": str(weights_path),
                "weights_fingerprint": weights_fingerprint,
                "normalization": "imagenet_mean_std",
                "pca_or_prismnet_applied": False,
            }
            write_json(metadata_path, metadata)
            source_count += len(items)
            if features.ndim == 2 and features.shape[1]:
                summary["feature_dim"] = int(features.shape[1])
        summary["sources"][source] = source_count
    write_json(experiment_dir / "embedding_summary.json", summary)
    return summary


def _load_feature_source(experiment_dir, source):
    values = {}
    source_dir = Path(experiment_dir) / "embeddings" / source
    for slide_dir in sorted(source_dir.iterdir() if source_dir.exists() else []):
        feature_path = slide_dir / "features.npy"
        index_path = slide_dir / "index.jsonl"
        if not feature_path.exists() or not index_path.exists():
            continue
        features = np.asarray(np.load(str(feature_path)), dtype=np.float32)
        for row in read_jsonl(index_path):
            values[row["key"]] = features[int(row["feature_row"])]
    return values


def _load_training_arrays(experiment_dir, annotation_path):
    roi_rows = read_jsonl(Path(experiment_dir) / "roi_manifest.jsonl")
    annotations = {row["roi_id"]: row for row in read_jsonl(annotation_path)}
    child_features = _load_feature_source(experiment_dir, "child_20x_224")
    parent_b = _load_feature_source(experiment_dir, "parent_10x_224")
    parent_c = _load_feature_source(experiment_dir, "parent_20x_448")
    output = []
    for roi in roi_rows:
        annotation = annotations.get(roi["roi_id"])
        if annotation is None:
            continue
        targets, masks = annotation_to_targets(annotation)
        children = [child_features[child["patch_uid"]] for child in roi["children"]]
        coordinates = []
        for child in roi["children"]:
            coordinates.append(
                [float(child["relative_col"]) - 0.5, float(child["relative_row"]) - 0.5, 1.0]
            )
        context_children = roi.get("context_4x4_children", [])
        context_features = []
        context_coordinates = []
        if len(context_children) == 16 and all(child["patch_uid"] in child_features for child in context_children):
            for child in context_children:
                context_features.append(child_features[child["patch_uid"]])
                context_coordinates.append(
                    [
                        (float(child["relative_col"]) - 1.5) / 1.5,
                        (float(child["relative_row"]) - 1.5) / 1.5,
                        1.0 if child.get("is_target_region") else 0.0,
                    ]
                )
        output.append(
            {
                "roi_id": roi["roi_id"],
                "slide_id": roi["slide_id"],
                "split": roi["split"],
                "child_features": np.stack(children).astype(np.float32),
                "parent_b": np.asarray(parent_b[roi["roi_id"]], dtype=np.float32),
                "parent_c": np.asarray(parent_c[roi["roi_id"]], dtype=np.float32),
                "coordinates": np.asarray(coordinates, dtype=np.float32),
                "context_features": np.asarray(context_features, dtype=np.float32),
                "context_coordinates": np.asarray(context_coordinates, dtype=np.float32),
                "targets": targets,
                "masks": masks,
            }
        )
    return output, annotations


def _variant_inputs(row, variant, token_window="2x2"):
    if variant == "B":
        return row["parent_b"]
    if variant == "C":
        return row["parent_c"]
    if token_window == "4x4":
        return row["context_features"]
    return row["child_features"]


def _variant_coordinates(row, token_window="2x2"):
    return row["context_coordinates"] if token_window == "4x4" else row["coordinates"]


def _metric_payload(targets, masks, probabilities):
    from sklearn.metrics import average_precision_score, roc_auc_score

    metrics = {}
    for index, label in enumerate(ALL_OUTPUT_LABELS):
        valid = masks[:, index] > 0.5
        y_true = targets[valid, index]
        y_score = probabilities[valid, index]
        payload = {"n": int(valid.sum()), "positives": int(y_true.sum()) if len(y_true) else 0}
        if len(y_true) and len(np.unique(y_true)) > 1:
            payload["auprc"] = float(average_precision_score(y_true, y_score))
            payload["auroc"] = float(roc_auc_score(y_true, y_score))
        else:
            payload["auprc"] = None
            payload["auroc"] = None
        metrics[label] = payload
    architecture_values = [metrics[label]["auprc"] for label in ALL_OUTPUT_LABELS[1:4] if metrics[label]["auprc"] is not None]
    metrics["macro_architecture_auprc"] = float(np.mean(architecture_values)) if architecture_values else None
    return metrics


def train_models(
    experiment_dir,
    annotation_path,
    variants,
    allow_synthetic=False,
    epochs=2,
    seed=17,
    device_name="auto",
    learning_rate=1e-3,
    token_window="2x2",
    run_name="",
):
    ensure_torch_runtime()
    import torch

    from adenoma_agent.architecture_models import build_model, masked_multitask_bce

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    experiment_dir = Path(experiment_dir).resolve()
    rows, annotations = _load_training_arrays(experiment_dir, annotation_path)
    synthetic = any(bool(row.get("synthetic_smoke_only")) for row in annotations.values())
    if synthetic and not allow_synthetic:
        raise RuntimeError("Synthetic smoke annotations require --allow-synthetic-smoke-labels")
    if not rows:
        raise RuntimeError("No annotated ROI rows with complete embeddings were found")
    if token_window not in ("2x2", "4x4"):
        raise ValueError("token_window must be 2x2 or 4x4")
    if token_window == "4x4":
        rows = [row for row in rows if row["context_features"].shape[0] == 16]
        if not rows:
            raise RuntimeError("No ROI has a complete 4x4 context window")
        unsupported = [variant for variant in variants if variant.upper() in ("B", "C")]
        if unsupported:
            raise ValueError("4x4 context is only supported by token models A/D0/D1/D2")
    device = _device(device_name, torch)
    feature_dim = int(rows[0]["parent_b"].shape[-1])
    targets_all = np.stack([row["targets"] for row in rows])
    masks_all = np.stack([row["masks"] for row in rows])
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    positives = (targets_all[train_mask] * masks_all[train_mask]).sum(axis=0)
    negatives = ((1.0 - targets_all[train_mask]) * masks_all[train_mask]).sum(axis=0)
    positive_weights = np.where(positives > 0, np.maximum(1.0, negatives / np.maximum(positives, 1.0)), 1.0)
    positive_weights = torch.tensor(positive_weights, dtype=torch.float32, device=device)
    results = {}
    run_dir = experiment_dir / "runs" / safe_name(run_name) if run_name else experiment_dir
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        variant = variant.upper()
        model = build_model(variant, feature_dim=feature_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        history = []
        train_rows = [row for row in rows if row["split"] == "train"]
        for epoch in range(int(epochs)):
            model.train()
            rng = random.Random(int(seed) + epoch)
            rng.shuffle(train_rows)
            losses = []
            for row in train_rows:
                features = torch.tensor(_variant_inputs(row, variant, token_window), dtype=torch.float32, device=device).unsqueeze(0)
                coordinates = torch.tensor(_variant_coordinates(row, token_window), dtype=torch.float32, device=device).unsqueeze(0)
                targets = torch.tensor(row["targets"], dtype=torch.float32, device=device).unsqueeze(0)
                masks = torch.tensor(row["masks"], dtype=torch.float32, device=device).unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features, coordinates)
                loss = masked_multitask_bce(logits, targets, masks, positive_weights=positive_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else None})
        model.eval()
        split_metrics = {}
        with torch.inference_mode():
            for split in ("train", "val", "test"):
                split_rows = [row for row in rows if row["split"] == split]
                if not split_rows:
                    split_metrics[split] = {}
                    continue
                probabilities = []
                targets = []
                masks = []
                for row in split_rows:
                    features = torch.tensor(_variant_inputs(row, variant, token_window), dtype=torch.float32, device=device).unsqueeze(0)
                    coordinates = torch.tensor(_variant_coordinates(row, token_window), dtype=torch.float32, device=device).unsqueeze(0)
                    probabilities.append(torch.sigmoid(model(features, coordinates)).cpu().numpy()[0])
                    targets.append(row["targets"])
                    masks.append(row["masks"])
                split_metrics[split] = _metric_payload(np.stack(targets), np.stack(masks), np.stack(probabilities))
        checkpoint_path = checkpoint_dir / "{0}.pt".format(variant)
        torch.save(
            {
                "variant": variant,
                "feature_dim": feature_dim,
                "state_dict": model.state_dict(),
                "labels": list(ALL_OUTPUT_LABELS),
                "synthetic_smoke_only": synthetic,
            },
            str(checkpoint_path),
        )
        results[variant] = {"history": history, "metrics": split_metrics, "checkpoint": str(checkpoint_path)}
    payload = {
        "pipeline": "mucosal_architecture_classifier_experiment_v1",
        "variants": results,
        "device": str(device),
        "synthetic_smoke_only": synthetic,
        "medical_performance_interpretation_allowed": not synthetic,
        "token_window": token_window,
        "run_name": run_name or "default",
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "metrics.json", payload)
    return payload


def run_smoke(args):
    summary = build_architecture_roi_manifest(
        args.artifact_dir,
        output_dir=args.output_dir,
        max_per_slide=args.max_per_slide,
        seed=args.seed,
        include_context=True,
    )
    experiment_dir = Path(summary["output_dir"])
    roi_rows = read_jsonl(experiment_dir / "roi_manifest.jsonl")
    annotations = build_synthetic_smoke_annotations(roi_rows)
    annotation_path = experiment_dir / "synthetic_smoke_annotations.jsonl"
    write_jsonl(annotation_path, annotations)
    embedding_summary = export_embeddings(
        experiment_dir,
        args.uni_weights_path,
        device_name=args.device,
        batch_size=args.batch_size,
        c_batch_size=args.c_batch_size,
        resume=args.resume,
    )
    metrics = train_models(
        experiment_dir,
        annotation_path,
        variants=args.variants.split(","),
        allow_synthetic=True,
        epochs=args.epochs,
        seed=args.seed,
        device_name=args.device,
        learning_rate=args.learning_rate,
        token_window="2x2",
        run_name="smoke_2x2",
    )
    report = {
        "status": "ok",
        "pipeline": "mucosal_architecture_wsi_smoke_v1",
        "real_wsi_images": True,
        "synthetic_smoke_only": True,
        "medical_performance_interpretation_allowed": False,
        "warning": "All current WSI are SSLD and labels are synthetic; this run validates software only.",
        "prepare": summary,
        "embeddings": embedding_summary,
        "trained_variants": sorted(metrics["variants"]),
        "metrics_path": str(experiment_dir / "runs" / "smoke_2x2" / "metrics.json"),
    }
    write_json(experiment_dir / "smoke_report.json", report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare, embed, train, and smoke-test the mucosal architecture experiment.")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("artifact_dir")
    prepare.add_argument("--output-dir", default="")
    prepare.add_argument("--max-per-slide", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=17)

    embed = subparsers.add_parser("embed")
    embed.add_argument("experiment_dir")
    embed.add_argument("--uni-weights-path", default=str(DEFAULT_UNI_WEIGHTS))
    embed.add_argument("--device", default="auto")
    embed.add_argument("--batch-size", type=int, default=8)
    embed.add_argument("--c-batch-size", type=int, default=1)
    embed.add_argument("--resume", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("experiment_dir")
    train.add_argument("--annotations", required=True)
    train.add_argument("--variants", default="A,B,C,D0,D1,D2")
    train.add_argument("--allow-synthetic-smoke-labels", action="store_true")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default="auto")
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--token-window", choices=["2x2", "4x4"], default="2x2")
    train.add_argument("--run-name", default="")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("artifact_dir")
    smoke.add_argument("--output-dir", default="")
    smoke.add_argument("--uni-weights-path", default=str(DEFAULT_UNI_WEIGHTS))
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--batch-size", type=int, default=8)
    smoke.add_argument("--c-batch-size", type=int, default=1)
    smoke.add_argument("--max-per-slide", type=int, default=10)
    smoke.add_argument("--variants", default="A,B,C,D0,D1,D2")
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--learning-rate", type=float, default=1e-3)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.time()
    if args.command == "prepare":
        payload = build_architecture_roi_manifest(
            args.artifact_dir,
            output_dir=args.output_dir or None,
            max_per_slide=args.max_per_slide,
            seed=args.seed,
        )
    elif args.command == "embed":
        payload = export_embeddings(
            args.experiment_dir,
            args.uni_weights_path,
            device_name=args.device,
            batch_size=args.batch_size,
            c_batch_size=args.c_batch_size,
            resume=args.resume,
        )
    elif args.command == "train":
        payload = train_models(
            args.experiment_dir,
            args.annotations,
            variants=args.variants.split(","),
            allow_synthetic=args.allow_synthetic_smoke_labels,
            epochs=args.epochs,
            seed=args.seed,
            device_name=args.device,
            learning_rate=args.learning_rate,
            token_window=args.token_window,
            run_name=args.run_name,
        )
    elif args.command == "smoke":
        payload = run_smoke(args)
    else:
        raise SystemExit("A subcommand is required: prepare, embed, train, or smoke")
    print(json.dumps({"elapsed_seconds": round(time.time() - started, 3), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
