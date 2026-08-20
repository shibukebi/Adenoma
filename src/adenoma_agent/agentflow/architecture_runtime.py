import json
from pathlib import Path
from typing import Mapping

from adenoma_agent.agentflow.contracts import ArchitecturePatchPrediction


class ArchitectureModelUnavailableError(RuntimeError):
    pass


def load_five_x_manifest(path, min_mucosa_coverage=0.30):
    """Load the canonical Mucosa -> Architecture boundary from Agent_workflow."""

    rows = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [
                key
                for key in ("slide_id", "patch_id", "level0_bbox", "mucosa_coverage", "target_magnification")
                if key not in row
            ]
            if missing:
                raise ValueError("five_x manifest line {0} lacks {1}".format(line_number, missing))
            magnification = str(row["target_magnification"]).lower().replace("x", "").strip()
            if abs(float(magnification) - 5.0) > 1e-6:
                raise ValueError("Architecture runtime accepts only canonical 5x manifest rows")
            bbox = tuple(int(value) for value in row["level0_bbox"])
            if (
                len(bbox) != 4
                or bbox[0] < 0
                or bbox[1] < 0
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError("Invalid level0_bbox at line {0}".format(line_number))
            patch_id = str(row["patch_id"])
            if patch_id in seen:
                raise ValueError("Duplicate patch_id in five_x manifest: {0}".format(patch_id))
            seen.add(patch_id)
            coverage = float(row["mucosa_coverage"])
            if coverage < float(min_mucosa_coverage):
                continue
            item = dict(row)
            item["level0_bbox"] = bbox
            item["mucosa_coverage"] = coverage
            rows.append(item)
    return tuple(rows)


class ArchitecturePredictor(object):
    synthetic_stub = False

    def predict(self, manifest_rows):
        raise NotImplementedError


class UnavailableArchitecturePredictor(ArchitecturePredictor):
    def __init__(self, reason="A clinically validated 5x architecture checkpoint is not configured"):
        self.reason = reason

    def predict(self, manifest_rows):
        raise ArchitectureModelUnavailableError(self.reason)


class ScriptedArchitecturePredictor(ArchitecturePredictor):
    """Explicit non-clinical predictor for control-flow and schema tests."""

    synthetic_stub = True

    def __init__(self, predictions, source_model="scripted-architecture-v1"):
        self.predictions = dict(predictions)
        self.source_model = source_model

    def predict(self, manifest_rows):
        output = []
        for row in manifest_rows:
            patch_id = row["patch_id"]
            if patch_id not in self.predictions:
                raise ArchitectureModelUnavailableError("No scripted architecture output for {0}".format(patch_id))
            item = self.predictions[patch_id]
            output.append(
                ArchitecturePatchPrediction(
                    patch_id=patch_id,
                    slide_id=row["slide_id"],
                    level0_bbox=tuple(row["level0_bbox"]),
                    mucosa_coverage=float(row["mucosa_coverage"]),
                    evaluable=float(item.get("evaluable", 1.0)),
                    architecture=dict(item["architecture"]),
                    context=dict(item["context"]),
                    uncertainty=float(item.get("uncertainty", row.get("mean_uncertainty", 0.5) or 0.5)),
                    source_model=self.source_model,
                    component_ids=tuple(row.get("source_component_ids", [])),
                    dysplasia_risk=float(item.get("dysplasia_risk", 0.0)),
                    abnormal_epithelial_score=float(item.get("abnormal_epithelial_score", 0.0)),
                    embedding_ref=item.get("embedding_ref"),
                    image_path=item.get("image_path"),
                    metadata={"synthetic_stub": True, "non_clinical": True},
                )
            )
        return tuple(output)


class ArchitectureInferenceRuntime(object):
    def __init__(self, predictor=None, min_mucosa_coverage=0.30):
        self.predictor = predictor or UnavailableArchitecturePredictor()
        self.min_mucosa_coverage = float(min_mucosa_coverage)

    def run(self, five_x_manifest_path):
        rows = load_five_x_manifest(
            five_x_manifest_path,
            min_mucosa_coverage=self.min_mucosa_coverage,
        )
        if not rows:
            raise ValueError("No 5x patches meet the configured mucosa coverage threshold")
        predictions = tuple(self.predictor.predict(rows))
        self._validate_predictions(rows, predictions)
        return predictions

    def _validate_predictions(self, manifest_rows, predictions):
        manifest_by_id = {str(row["patch_id"]): row for row in manifest_rows}
        prediction_by_id = {}
        for prediction in predictions:
            if prediction.patch_id in prediction_by_id:
                raise ValueError(
                    "Architecture predictor returned duplicate patch_id: {0}".format(
                        prediction.patch_id
                    )
                )
            prediction_by_id[prediction.patch_id] = prediction
        if set(prediction_by_id) != set(manifest_by_id):
            missing = sorted(set(manifest_by_id) - set(prediction_by_id))
            extra = sorted(set(prediction_by_id) - set(manifest_by_id))
            raise ValueError(
                "Architecture predictions must match the canonical 5x manifest exactly; "
                "missing={0}, extra={1}".format(missing, extra)
            )
        for patch_id, prediction in prediction_by_id.items():
            manifest_row = manifest_by_id[patch_id]
            if prediction.slide_id != str(manifest_row["slide_id"]):
                raise ValueError("Architecture prediction slide_id provenance mismatch")
            if tuple(prediction.level0_bbox) != tuple(manifest_row["level0_bbox"]):
                raise ValueError("Architecture prediction bbox provenance mismatch")
            if abs(float(prediction.mucosa_coverage) - float(manifest_row["mucosa_coverage"])) > 1e-6:
                raise ValueError("Architecture prediction mucosa coverage provenance mismatch")
            if not prediction.source_model:
                raise ValueError("Architecture prediction requires source_model provenance")
            if not getattr(self.predictor, "synthetic_stub", False) and not prediction.embedding_ref:
                raise ValueError("Production architecture prediction requires embedding_ref provenance")


def write_architecture_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            payload = prediction.to_dict()
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path
