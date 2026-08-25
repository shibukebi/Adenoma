"""Metrics, cluster-bootstrap statistics and weak-MIL explanation summaries."""

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def _as_arrays(predictions: Sequence[Mapping[str, object]], num_classes: int):
    if not predictions:
        raise ValueError("At least one prediction is required")
    labels = np.asarray([int(row["label"]) for row in predictions], dtype=np.int64)
    probabilities = np.asarray([row["probabilities"] for row in predictions], dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != int(num_classes):
        raise ValueError("Prediction probabilities must have shape [n, num_classes]")
    if np.any(labels < 0) or np.any(labels >= int(num_classes)):
        raise ValueError("Prediction labels are outside the frozen class map")
    if not np.isfinite(probabilities).all():
        raise ValueError("Prediction probabilities contain NaN or Inf")
    return labels, probabilities


def _binary_auc(labels: np.ndarray, scores: np.ndarray):
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int(labels.shape[0] - n_positive)
    if not n_positive or not n_negative:
        return None
    # Average ranks handle tied scores without requiring sklearn/scipy.
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        end = start + 1
        while end < scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (float(start + 1) + float(end)) / 2.0
        start = end
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def classification_metrics(predictions: Sequence[Mapping[str, object]], num_classes: int = 7) -> dict:
    """Compute slide-level multiclass metrics from JSON-safe predictions."""

    labels, probabilities = _as_arrays(predictions, num_classes)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for truth, estimate in zip(labels, predicted):
        matrix[int(truth), int(estimate)] += 1
    supports = matrix.sum(axis=1)
    per_class = []
    macro_f1_values = []
    supported_f1_values = []
    f1_weights = []
    recalls = []
    auc_values = []
    auc_weights = []
    for index in range(int(num_classes)):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(supports[index])
        precision = float(tp) / float(tp + fp) if tp + fp else 0.0
        recall = float(tp) / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = _binary_auc((labels == index).astype(np.int64), probabilities[:, index])
        per_class.append(
            {
                "class_index": index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            }
        )
        # The experiment has a frozen seven-class taxonomy.  Macro-F1 must
        # therefore average all seven classes, assigning zero to a class with
        # no support/predictions in a resample instead of silently changing
        # the metric denominator.
        macro_f1_values.append(f1)
        if support:
            supported_f1_values.append(f1)
            f1_weights.append(support)
            recalls.append(recall)
        if auc is not None:
            auc_values.append(auc)
            auc_weights.append(support)
    weighted_f1 = float(np.average(supported_f1_values, weights=f1_weights)) if f1_weights else 0.0
    weighted_auc = float(np.average(auc_values, weights=auc_weights)) if auc_weights else None
    return {
        "n_cases": int(labels.shape[0]),
        "accuracy": float((labels == predicted).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
        "weighted_f1": weighted_f1,
        "macro_auc": float(np.mean(auc_values)) if auc_values else None,
        "weighted_auc": weighted_auc,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def oof_metrics(predictions: Sequence[Mapping[str, object]], num_classes: int = 7) -> dict:
    """Validate and summarize a complete out-of-fold prediction set."""

    case_aliases = [str(row.get("case_alias", row.get("slide_id", ""))) for row in predictions]
    if not all(case_aliases) or len(case_aliases) != len(set(case_aliases)):
        raise ValueError("OOF predictions require one non-empty row per case_alias")
    return classification_metrics(predictions, num_classes=num_classes)


def _metric_value(predictions, metric_name: str, num_classes: int) -> float:
    value = classification_metrics(predictions, num_classes=num_classes).get(metric_name)
    if value is None:
        raise ValueError("Metric {0} is undefined for this resample".format(metric_name))
    return float(value)


def family_cluster_bootstrap(
    predictions: Sequence[Mapping[str, object]],
    metric_name: str = "macro_f1",
    num_classes: int = 7,
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> dict:
    """Cluster-bootstrap a metric, resampling whole heuristic families."""

    groups = defaultdict(list)
    for row in predictions:
        family = str(row.get("family_id", "")).strip()
        if not family:
            raise ValueError("Family cluster bootstrap requires family_id")
        groups[family].append(row)
    families = sorted(groups)
    if len(families) < 2:
        raise ValueError("At least two families are required for cluster bootstrap")
    rng = np.random.RandomState(int(seed))
    samples = []
    attempts = 0
    max_attempts = max(int(n_bootstrap) * 10, 100)
    while len(samples) < int(n_bootstrap) and attempts < max_attempts:
        attempts += 1
        chosen = rng.choice(families, size=len(families), replace=True)
        rows = [row for family in chosen for row in groups[str(family)]]
        try:
            samples.append(_metric_value(rows, metric_name, num_classes))
        except ValueError:
            # A rare-class-free resample cannot define macro AUC; retain only
            # resamples where the requested metric is mathematically defined.
            continue
    if len(samples) < int(n_bootstrap):
        raise RuntimeError("Could not obtain enough valid family bootstrap samples")
    point = _metric_value(predictions, metric_name, num_classes)
    return {
        "metric": metric_name,
        "point_estimate": point,
        "n_families": len(families),
        "n_bootstrap": len(samples),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "bootstrap_values": [float(value) for value in samples],
    }


def paired_family_bootstrap(
    predictions_a: Sequence[Mapping[str, object]],
    predictions_b: Sequence[Mapping[str, object]],
    metric_name: str = "macro_f1",
    num_classes: int = 7,
    n_bootstrap: int = 2000,
    seed: int = 17,
) -> dict:
    """Paired family-cluster bootstrap for a preplanned model comparison."""

    by_case_a = {str(row.get("case_alias", row.get("slide_id", ""))): row for row in predictions_a}
    by_case_b = {str(row.get("case_alias", row.get("slide_id", ""))): row for row in predictions_b}
    if not by_case_a or set(by_case_a) != set(by_case_b):
        raise ValueError("Paired bootstrap requires identical case_alias sets")
    groups = defaultdict(list)
    for case_alias in sorted(by_case_a):
        row_a = by_case_a[case_alias]
        row_b = by_case_b[case_alias]
        family = str(row_a.get("family_id", "")).strip()
        if not family or family != str(row_b.get("family_id", "")).strip():
            raise ValueError("Paired predictions require matching non-empty family_id")
        groups[family].append((row_a, row_b))
    families = sorted(groups)
    rng = np.random.RandomState(int(seed))
    deltas = []
    attempts = 0
    while len(deltas) < int(n_bootstrap) and attempts < max(int(n_bootstrap) * 10, 100):
        attempts += 1
        chosen = rng.choice(families, size=len(families), replace=True)
        pairs = [pair for family in chosen for pair in groups[str(family)]]
        try:
            delta = _metric_value([pair[0] for pair in pairs], metric_name, num_classes) - _metric_value(
                [pair[1] for pair in pairs], metric_name, num_classes
            )
            deltas.append(float(delta))
        except ValueError:
            continue
    if len(deltas) < int(n_bootstrap):
        raise RuntimeError("Could not obtain enough paired family bootstrap samples")
    point = _metric_value(predictions_a, metric_name, num_classes) - _metric_value(
        predictions_b, metric_name, num_classes
    )
    values = np.asarray(deltas, dtype=np.float64)
    p_value = min(1.0, 2.0 * min(float((values <= 0.0).mean()), float((values >= 0.0).mean())))
    return {
        "metric": metric_name,
        "point_delta": float(point),
        "n_families": len(families),
        "n_bootstrap": len(deltas),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "p_value": p_value,
        "bootstrap_deltas": [float(value) for value in values.tolist()],
    }


def holm_adjust(p_values: Mapping[str, float], alpha: float = 0.05) -> dict:
    """Apply Holm correction to the three preplanned baseline comparisons."""

    items = sorted(
        ((str(name), float(value)) for name, value in p_values.items()),
        key=lambda item: (item[1], item[0]),
    )
    count = len(items)
    adjusted = {}
    running = 0.0
    for index, (name, value) in enumerate(items):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = {
            "raw_p_value": value,
            "holm_adjusted_p_value": running,
            "reject_alpha": bool(running <= float(alpha)),
        }
    return adjusted


def _ranked_indices(scores: Sequence[float], descending: bool = True):
    return sorted(range(len(scores)), key=lambda index: (float(scores[index]), -index), reverse=descending)


def top_concentration(scores: Sequence[float], fractions=(0.01, 0.05, 0.10)) -> dict:
    """Return top-fraction signal mass for attention or instance-score vectors."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("top_concentration requires one finite non-empty score vector")
    # Attention is already non-negative mass.  Scores may be raw instance
    # logits, where softmax gives a stable comparable concentration measure.
    if np.all(values >= 0.0) and np.isclose(values.sum(), 1.0, rtol=1e-4, atol=1e-6):
        mass = values
    else:
        shifted = values - values.max()
        mass = np.exp(shifted)
        mass /= mass.sum()
    ordered = np.sort(mass)[::-1]
    output = {}
    for fraction in fractions:
        key = "top_{0:g}_percent".format(float(fraction) * 100.0)
        count = max(1, int(np.ceil(float(fraction) * len(values))))
        output[key] = {"patch_count": count, "signal_mass": float(ordered[:count].sum())}
    return output


def _grid_value(patch: Mapping[str, object]):
    value = patch.get("grid_index")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _bbox_adjacent(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    a = first.get("level0_bbox")
    b = second.get("level0_bbox")
    if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) == 4):
        return False
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    width = max(abs(float(a[2]) - float(a[0])), abs(float(b[2]) - float(b[0])))
    height = max(abs(float(a[3]) - float(a[1])), abs(float(b[3]) - float(b[1])))
    return abs(ax - bx) <= width and abs(ay - by) <= height


def spatial_component_summary(patches: Sequence[Mapping[str, object]], ranked_indices: Sequence[int]) -> dict:
    """Measure spatial diversity among ranked patches using 8-connected grid cells."""

    selected = [int(index) for index in ranked_indices]
    if not selected:
        return {"component_count": 0, "largest_component_fraction": 0.0, "component_sizes": []}
    for index in selected:
        if index < 0 or index >= len(patches):
            raise ValueError("Ranked patch index is outside the patch list")
    pending = set(selected)
    components = []
    while pending:
        start = pending.pop()
        component = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            current_grid = _grid_value(patches[current])
            neighbours = []
            for candidate in list(pending):
                candidate_grid = _grid_value(patches[candidate])
                if current_grid is not None and candidate_grid is not None:
                    adjacent = max(abs(current_grid[0] - candidate_grid[0]), abs(current_grid[1] - candidate_grid[1])) <= 1
                else:
                    adjacent = _bbox_adjacent(patches[current], patches[candidate])
                if adjacent:
                    neighbours.append(candidate)
            for candidate in neighbours:
                pending.remove(candidate)
                component.add(candidate)
                frontier.append(candidate)
        components.append(component)
    sizes = sorted((len(component) for component in components), reverse=True)
    return {
        "component_count": len(sizes),
        "largest_component_fraction": float(sizes[0]) / float(len(selected)),
        "component_sizes": sizes,
    }


def patch_ranking_summary(
    patches: Sequence[Mapping[str, object]],
    scores: Sequence[float],
    top_k: int = 10,
) -> dict:
    """Create top/bottom patch rankings plus concentration and spatial summaries."""

    if len(patches) != len(scores):
        raise ValueError("Patch metadata and score vectors must have equal length")
    ranked = _ranked_indices(scores, descending=True)
    def rows(indices):
        result = []
        for rank, index in enumerate(indices, 1):
            patch = dict(patches[index])
            result.append(
                {
                    "rank": rank,
                    "patch_id": patch.get("patch_id"),
                    "level0_bbox": patch.get("level0_bbox"),
                    "grid_index": patch.get("grid_index"),
                    "score": float(scores[index]),
                }
            )
        return result
    top = ranked[: min(int(top_k), len(ranked))]
    bottom = list(reversed(ranked[-min(int(top_k), len(ranked)) :]))
    return {
        "top": rows(top),
        "bottom": rows(bottom),
        "concentration": top_concentration(scores),
        "spatial_diversity": spatial_component_summary(patches, top),
    }


def explain_weak_mil_prediction(
    bag: Mapping[str, object], prediction: Mapping[str, object], top_k: int = 10
) -> dict:
    """Produce explicitly non-ground-truth patch relevance summaries.

    ABMIL uses the bag attention vector.  DSMIL adds an instance-score ranking
    and uses the predicted class column of its class-specific attention map.
    The returned text is intentionally fixed to prevent downstream reports
    from presenting weak-MIL scores as pathologist-confirmed morphology.
    """

    patches = bag.get("patches", [])
    if not isinstance(patches, list) or not patches:
        raise ValueError("Bag patch metadata is required for weak-MIL explanations")
    result = {
        "case_alias": str(prediction.get("case_alias", bag.get("case_alias", ""))),
        "slide_id": str(prediction.get("slide_id", bag.get("slide_id", ""))),
        "prediction": int(prediction["prediction"]),
        "interpretation": "candidate_diagnostic_relevance_not_pathologist_confirmed_architecture",
    }
    predicted_class = int(prediction["prediction"])
    if "class_attention" in prediction:
        class_attention = np.asarray(prediction["class_attention"], dtype=np.float64)
        if class_attention.ndim != 2 or class_attention.shape[0] != len(patches):
            raise ValueError("DSMIL class_attention does not match the bag patch count")
        if predicted_class < 0 or predicted_class >= class_attention.shape[1]:
            raise ValueError("Predicted class is outside DSMIL class_attention")
        result["dsmil_attention"] = patch_ranking_summary(
            patches, class_attention[:, predicted_class].tolist(), top_k=top_k
        )
    elif "attention" in prediction:
        result["abmil_attention"] = patch_ranking_summary(patches, prediction["attention"], top_k=top_k)
    if "instance_scores" in prediction:
        instance_scores = np.asarray(prediction["instance_scores"], dtype=np.float64)
        if instance_scores.ndim != 2 or instance_scores.shape[0] != len(patches):
            raise ValueError("Instance scores do not match the bag patch count")
        if predicted_class < 0 or predicted_class >= instance_scores.shape[1]:
            raise ValueError("Predicted class is outside instance scores")
        result["dsmil_instance_score"] = patch_ranking_summary(
            patches, instance_scores[:, predicted_class].tolist(), top_k=top_k
        )
    return result
