"""Deterministic training utilities for frozen-CONCH architecture baselines."""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import classification_metrics


DEFAULT_FEATURE_DIM = 512
DEFAULT_NUM_CLASSES = 7


def set_deterministic_seed(seed: int) -> None:
    """Set the Python, NumPy and Torch RNGs used by a baseline run."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        # Older supported Torch releases do not offer this global switch.
        pass


def train_only_class_weights(labels: Sequence[int], num_classes: int = DEFAULT_NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency CE weights fitted exclusively from the train split."""

    labels = [int(value) for value in labels]
    if not labels:
        raise ValueError("Cannot fit class weights without training labels")
    counts = Counter(labels)
    missing = [index for index in range(int(num_classes)) if counts.get(index, 0) == 0]
    if missing:
        raise ValueError("Training split is missing class indices: {0}".format(missing))
    total = float(len(labels))
    return torch.tensor(
        [total / (float(num_classes) * float(counts[index])) for index in range(int(num_classes))],
        dtype=torch.float32,
    )


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_case_ids(path: Path) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("case_aliases", payload.get("cases", []))
    else:
        raise ValueError("Unsupported split manifest payload: {0}".format(path))
    case_ids = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("case_alias", value.get("slide_id", ""))
        value = str(value or "").strip()
        if value:
            case_ids.append(value)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case aliases in split manifest: {0}".format(path))
    return case_ids


def _resolve_label(row: Mapping[str, object], class_to_index: Optional[Mapping[str, int]]) -> int:
    for key in ("label_index", "class_index"):
        if key in row:
            return int(row[key])
    if "label" in row and class_to_index is not None:
        label = str(row["label"])
        if label not in class_to_index:
            raise ValueError("Unknown slide label: {0}".format(label))
        return int(class_to_index[label])
    raise ValueError("Each label row requires label_index/class_index or label plus class_to_index")


def _validate_cache_index(index_rows: Sequence[Mapping[str, object]], case_alias: str, row_count: int) -> None:
    if len(index_rows) != int(row_count):
        raise ValueError("Feature/index row mismatch for {0}".format(case_alias))
    patch_ids = set()
    required = ("case_alias", "patch_id", "grid_index", "level0_bbox", "mucosa_coverage", "image_sha256")
    for row in index_rows:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError("Cache index row is missing {0}".format(missing))
        if str(row["case_alias"]) != case_alias:
            raise ValueError("Cache index case_alias mismatch for {0}".format(case_alias))
        patch_id = str(row["patch_id"])
        if not patch_id or patch_id in patch_ids:
            raise ValueError("Duplicate or empty patch_id in {0}".format(case_alias))
        patch_ids.add(patch_id)
        bbox = row["level0_bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Invalid level0_bbox for {0}/{1}".format(case_alias, patch_id))


def load_cached_bags(
    cache_root: Path,
    labels_jsonl: Path,
    split_dir: Optional[Path] = None,
    min_mucosa_coverage: float = 0.60,
    class_to_index: Optional[Mapping[str, int]] = None,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> List[dict]:
    """Load cache-backed slide bags without reading source WSIs or labels from paths.

    The cache contract is one directory per ``case_alias`` with ``features.npy``
    and ``index.jsonl``.  The index contains image provenance, while labels are
    supplied independently to prevent diagnosis labels entering the cache.
    """

    cache_root = Path(cache_root)
    labels = _read_jsonl(labels_jsonl)
    split_by_case = {}
    if split_dir is not None:
        for split in ("train", "val", "test"):
            path = Path(split_dir) / "{0}_cases.json".format(split)
            if not path.is_file():
                continue
            for case_alias in _read_case_ids(path):
                if case_alias in split_by_case:
                    raise ValueError("Case appears in multiple split manifests: {0}".format(case_alias))
                split_by_case[case_alias] = split

    output = []
    seen_cases = set()
    for label_row in labels:
        case_alias = str(label_row.get("case_alias", label_row.get("slide_id", ""))).strip()
        if not case_alias:
            raise ValueError("Label row lacks case_alias/slide_id")
        # A canonical label inventory may contain audit-only HP cases and YX
        # slides that did not pass the post-Mucosa eligibility gate.  When a
        # formal fold is supplied, load exactly its cases instead of requiring
        # embedding caches for every row in the broader inventory.
        if split_by_case and case_alias not in split_by_case:
            continue
        if case_alias in seen_cases:
            raise ValueError("Duplicate case label: {0}".format(case_alias))
        seen_cases.add(case_alias)
        case_dir = cache_root / case_alias
        features_path = case_dir / "features.npy"
        index_path = case_dir / "index.jsonl"
        metadata_path = case_dir / "metadata.json"
        if not features_path.is_file() or not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Incomplete embedding cache for {0}".format(case_alias))
        features = np.load(str(features_path), mmap_mode="r")
        if features.ndim != 2 or int(features.shape[1]) != int(feature_dim):
            raise ValueError("Expected [N,{0}] features for {1}".format(feature_dim, case_alias))
        if features.dtype not in (np.dtype("float16"), np.dtype("float32")):
            raise ValueError("Features must be float16/float32 for {0}".format(case_alias))
        if not np.isfinite(np.asarray(features)).all():
            raise ValueError("NaN/Inf feature in {0}".format(case_alias))
        index_rows = _read_jsonl(index_path)
        _validate_cache_index(index_rows, case_alias, features.shape[0])
        selected = [
            index for index, row in enumerate(index_rows)
            if float(row["mucosa_coverage"]) >= float(min_mucosa_coverage)
        ]
        if not selected:
            raise ValueError(
                "No patches meet mucosa_coverage>={0} for {1}".format(min_mucosa_coverage, case_alias)
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        output.append(
            {
                "case_alias": case_alias,
                "slide_id": str(label_row.get("slide_id", case_alias)),
                "family_id": str(label_row.get("family_id", case_alias)),
                "label": _resolve_label(label_row, class_to_index),
                "split": split_by_case.get(case_alias, str(label_row.get("split", ""))),
                "features": np.asarray(features[selected], dtype=np.float32),
                "patches": [dict(index_rows[index]) for index in selected],
                "cache_metadata": metadata,
                "original_patch_count": int(features.shape[0]),
                "used_patch_count": len(selected),
                "sampling_method": "all_mucosa_coverage_gte_{0:.2f}".format(float(min_mucosa_coverage)),
            }
        )
    if split_by_case:
        missing_labels = sorted(set(split_by_case) - seen_cases)
        if missing_labels:
            raise ValueError("Split cases are absent from labels JSONL: {0}".format(missing_labels[:10]))
    return output


def _bag_tensor(row: Mapping[str, object], device: torch.device, feature_dim: int) -> torch.Tensor:
    features = np.asarray(row["features"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != int(feature_dim):
        raise ValueError("Invalid bag features for {0}".format(row.get("case_alias", "unknown")))
    return torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)


def _output_loss(output: Mapping[str, torch.Tensor], label: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    if output["logits"].ndim != 2:
        raise ValueError(
            "Weak-MIL training requires one [batch, classes] slide logit per bag; "
            "do not apply slide labels to LinearProbe/SmallMLP patch logits"
        )
    bag_loss = F.cross_entropy(output["logits"], label, weight=class_weights)
    max_logits = output.get("max_instance_logits")
    if max_logits is None:
        return bag_loss
    max_loss = F.cross_entropy(max_logits, label, weight=class_weights)
    return 0.5 * bag_loss + 0.5 * max_loss


def _jsonable_prediction(row: Mapping[str, object], output: Mapping[str, torch.Tensor]) -> dict:
    logits = output["logits"].detach().cpu()[0]
    probabilities = torch.softmax(logits, dim=0)
    prediction = {
        "case_alias": str(row.get("case_alias", "")),
        "slide_id": str(row.get("slide_id", row.get("case_alias", ""))),
        "family_id": str(row.get("family_id", row.get("case_alias", ""))),
        "split": str(row.get("split", "")),
        "label": int(row["label"]),
        "prediction": int(torch.argmax(probabilities).item()),
        "probabilities": [float(value) for value in probabilities.tolist()],
        "original_patch_count": int(row.get("original_patch_count", len(row.get("patches", [])))),
        "used_patch_count": int(row.get("used_patch_count", len(row.get("patches", [])))),
    }
    attention = output.get("attention")
    if attention is not None:
        prediction["attention"] = [float(value) for value in attention.detach().cpu()[0].tolist()]
    instance_scores = output.get("instance_scores")
    if instance_scores is not None:
        prediction["instance_scores"] = [
            [float(value) for value in patch_scores]
            for patch_scores in instance_scores.detach().cpu()[0].tolist()
        ]
    class_attention = output.get("class_attention")
    if class_attention is not None:
        prediction["class_attention"] = [
            [float(value) for value in patch_scores]
            for patch_scores in class_attention.detach().cpu()[0].tolist()
        ]
    max_instance_logits = output.get("max_instance_logits")
    if max_instance_logits is not None:
        prediction["max_instance_logits"] = [
            float(value) for value in max_instance_logits.detach().cpu()[0].tolist()
        ]
    return prediction


def predict_bags(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, object]],
    device: Optional[str] = None,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> List[dict]:
    """Run batch-one variable-length bag inference and retain explanation tensors."""

    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(resolved)
    model.eval()
    predictions = []
    with torch.no_grad():
        for row in rows:
            output = model(_bag_tensor(row, resolved, feature_dim))
            predictions.append(_jsonable_prediction(row, output))
    return predictions


def train_baseline(
    model: torch.nn.Module,
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    model_name: str,
    seed: int = 17,
    max_epochs: int = 100,
    patience: int = 15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_classes: int = DEFAULT_NUM_CLASSES,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    device: Optional[str] = None,
) -> dict:
    """Fit one head with train-only weights and macro-F1 early stopping.

    ``val_macro_f1`` is the primary checkpoint criterion.  Equal F1 is broken
    by lower validation loss, never by test performance.
    """

    if not train_rows or not val_rows:
        raise ValueError("train_baseline requires non-empty train and validation bags")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(seed)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(resolved)
    weights = train_only_class_weights([row["label"] for row in train_rows], num_classes=num_classes).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))
    best_state = None
    best_epoch = 0
    best_f1 = float("-inf")
    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        ordered_rows = list(train_rows)
        rng.shuffle(ordered_rows)
        losses = []
        for row in ordered_rows:
            optimizer.zero_grad()
            output = model(_bag_tensor(row, resolved, feature_dim))
            label = torch.tensor([int(row["label"])], dtype=torch.long, device=resolved)
            loss = _output_loss(output, label, weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_predictions, val_loss = _evaluate_with_loss(model, val_rows, weights, resolved, feature_dim)
        val_metrics = classification_metrics(val_predictions, num_classes=num_classes)
        val_f1 = float(val_metrics["macro_f1"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        }
        history.append(record)
        print(
            json.dumps(
                {
                    "event": "architecture_training_epoch",
                    "model": str(model_name),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "train_loss": record["train_loss"],
                    "val_loss": record["val_loss"],
                    "val_macro_f1": record["val_macro_f1"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        improved = val_f1 > best_f1 or (val_f1 == best_f1 and val_loss < best_loss)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_f1 = val_f1
            best_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(
        {
            "model_name": str(model_name),
            "seed": int(seed),
            "feature_dim": int(feature_dim),
            "num_classes": int(num_classes),
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_f1),
            "best_val_loss": float(best_loss),
            "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
            "model_config": dict(getattr(model, "architecture_baseline_config", {})),
            "state_dict": best_state,
        },
        str(checkpoint_path),
    )
    val_predictions = predict_bags(model, val_rows, device=str(resolved), feature_dim=feature_dim)
    predictions_path = output_dir / "val_predictions.json"
    predictions_path.write_text(json.dumps(val_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "model_name": str(model_name),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
        "best_val_loss": float(best_loss),
        "checkpoint": str(checkpoint_path),
        "val_predictions": str(predictions_path),
        "history": history,
        "class_weights": [float(value) for value in weights.detach().cpu().tolist()],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _evaluate_with_loss(model, rows, weights, device, feature_dim):
    model.eval()
    predictions = []
    losses = []
    with torch.no_grad():
        for row in rows:
            output = model(_bag_tensor(row, device, feature_dim))
            label = torch.tensor([int(row["label"])], dtype=torch.long, device=device)
            losses.append(float(_output_loss(output, label, weights).detach().cpu().item()))
            predictions.append(_jsonable_prediction(row, output))
    return predictions, float(np.mean(losses)) if losses else float("inf")
