import math
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from adenoma_agent.agentflow.contracts import (
    ArchitecturePatchPrediction,
    EvidenceRecord,
    ROICandidate,
    SpatialEvidenceSummary,
)


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def _weighted_mean(rows, value_fn, weight_fn):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = max(0.0, float(weight_fn(row)))
        numerator += weight * float(value_fn(row))
        denominator += weight
    return numerator / denominator if denominator > 0.0 else 0.0


class SpatialEvidenceEvaluator(object):
    """Deterministic soft-score aggregation for architecture evidence."""

    def evaluate(self, predictions):
        predictions = tuple(predictions)
        if not predictions:
            raise ValueError("SpatialEvidenceEvaluator requires at least one prediction")
        slide_ids = {row.slide_id for row in predictions}
        if len(slide_ids) != 1:
            raise ValueError("Spatial evidence must be evaluated one slide at a time")
        slide_id = next(iter(slide_ids))
        category_values = {
            "serrated": lambda row: row.architecture["serrated"],
            "tubular": lambda row: row.architecture["tubular"],
            "villous": lambda row: row.architecture["villous"],
            "normal": lambda row: row.context["normal_mucosa_present"],
            "reactive": lambda row: row.context["reactive_inflammatory_present"],
        }
        masses = {}
        for label, value_fn in category_values.items():
            masses[label] = sum(
                row.mucosa_coverage * row.evaluable * float(value_fn(row)) for row in predictions
            )
        valid_mass = sum(masses.values())
        ratios = {
            "{0}_ratio".format(label): (mass / valid_mass if valid_mass > 0.0 else 0.0)
            for label, mass in masses.items()
        }
        total_coverage = sum(row.mucosa_coverage for row in predictions)
        ratios["non_evaluable_ratio"] = (
            sum(row.mucosa_coverage * (1.0 - row.evaluable) for row in predictions) / total_coverage
            if total_coverage > 0.0
            else 1.0
        )

        groups = defaultdict(list)
        for row in predictions:
            group_id = "component_{0}".format(row.component_ids[0]) if row.component_ids else "patch_{0}".format(row.patch_id)
            groups[group_id].append(row)

        def cluster_score(label):
            scores = []
            for rows in groups.values():
                scores.append(
                    _weighted_mean(
                        rows,
                        lambda row: row.architecture[label],
                        lambda row: row.mucosa_coverage * row.evaluable,
                    )
                )
            return max(scores) if scores else 0.0

        mixing = _weighted_mean(
            predictions,
            lambda row: 2.0 * min(row.architecture["tubular"], row.architecture["villous"]),
            lambda row: row.mucosa_coverage * row.evaluable,
        )
        colocalization = _weighted_mean(
            predictions,
            lambda row: 2.0 * min(row.architecture["serrated"], row.architecture["villous"]),
            lambda row: row.mucosa_coverage * row.evaluable,
        )
        entropy_values = []
        entropy_weights = []
        for row in predictions:
            values = [
                row.architecture["serrated"],
                row.architecture["tubular"],
                row.architecture["villous"],
                row.context["normal_mucosa_present"],
                row.context["reactive_inflammatory_present"],
            ]
            total = sum(values)
            if total <= 0.0:
                entropy = 1.0
            else:
                probabilities = [value / total for value in values if value > 0.0]
                entropy = -sum(value * math.log(value) for value in probabilities) / math.log(5.0)
            entropy_values.append(entropy)
            entropy_weights.append(row.mucosa_coverage * row.evaluable)
        entropy_denominator = sum(entropy_weights)
        heterogeneity = (
            sum(value * weight for value, weight in zip(entropy_values, entropy_weights)) / entropy_denominator
            if entropy_denominator > 0.0
            else 1.0
        )
        spatial = {
            "serrated_cluster_score": cluster_score("serrated"),
            "villous_cluster_score": cluster_score("villous"),
            "tubular_cluster_score": cluster_score("tubular"),
            "tubular_villous_mixing": _clamp(mixing),
            "serrated_villous_colocalization": _clamp(colocalization),
            "architecture_heterogeneity": _clamp(heterogeneity),
        }
        uncertain_groups = 0
        for rows in groups.values():
            group_uncertainty = _weighted_mean(rows, lambda row: row.uncertainty, lambda row: row.mucosa_coverage)
            if group_uncertainty >= 0.5:
                uncertain_groups += 1
        uncertainty = {
            "high_uncertainty_cluster_count": int(uncertain_groups),
            "max_uncertainty": max(row.uncertainty for row in predictions),
            "mean_uncertainty": _weighted_mean(
                predictions,
                lambda row: row.uncertainty,
                lambda row: row.mucosa_coverage,
            ),
        }
        return SpatialEvidenceSummary(
            slide_id=slide_id,
            ratios=ratios,
            spatial=spatial,
            uncertainty=uncertainty,
            source_patch_ids=tuple(row.patch_id for row in predictions),
        )

    def to_evidence(self, case_id, predictions, summary, source_version="untrained_or_external"):
        records = []
        feature_map = {
            "serrated": "serrated_architecture",
            "tubular": "tubular_architecture",
            "villous": "villous_architecture",
        }
        for row in predictions:
            for label, feature in feature_map.items():
                score = float(row.architecture[label])
                status, confidence, evaluability = self._score_status(score, row.evaluable)
                records.append(
                    EvidenceRecord(
                        evidence_id="ARCH_{0}_{1}".format(row.patch_id, feature),
                        case_id=case_id,
                        evidence_type="architecture_patch_evidence",
                        feature=feature,
                        status=status,
                        confidence=confidence,
                        source=row.source_model,
                        source_version=source_version,
                        quality=row.evaluable,
                        feature_evaluability=evaluability,
                        roi_id=row.patch_id,
                        scale=row.scale,
                        level0_bbox=row.level0_bbox,
                        patch_id=row.patch_id,
                        cluster_id=("component_{0}".format(row.component_ids[0]) if row.component_ids else None),
                        limitations=("Architecture classifier checkpoint requires clinical validation.",),
                        metadata={
                            "raw_score": score,
                            "mucosa_coverage": row.mucosa_coverage,
                            "uncertainty": row.uncertainty,
                            "embedding_ref": row.embedding_ref,
                            "non_clinical": True,
                        },
                    )
                )
            reactive_score = float(row.context["reactive_inflammatory_present"])
            status, confidence, evaluability = self._score_status(reactive_score, row.evaluable)
            records.append(
                EvidenceRecord(
                    evidence_id="ARCH_{0}_reactive_context".format(row.patch_id),
                    case_id=case_id,
                    evidence_type="architecture_patch_evidence",
                    feature="reactive_inflammatory_context",
                    status=status,
                    confidence=confidence,
                    source=row.source_model,
                    source_version=source_version,
                    quality=row.evaluable,
                    feature_evaluability=evaluability,
                    roi_id=row.patch_id,
                    scale=row.scale,
                    level0_bbox=row.level0_bbox,
                    patch_id=row.patch_id,
                    limitations=("Architecture classifier checkpoint requires clinical validation.",),
                    metadata={"raw_score": reactive_score, "non_clinical": True},
                )
            )
        for section, values in (("ratio", summary.ratios), ("spatial", summary.spatial), ("uncertainty", summary.uncertainty)):
            for feature, value in values.items():
                if isinstance(value, int):
                    numeric = float(value)
                else:
                    numeric = float(value)
                records.append(
                    EvidenceRecord(
                        evidence_id="SPATIAL_{0}_{1}".format(summary.slide_id, feature),
                        case_id=case_id,
                        evidence_type="spatial_evidence",
                        feature=feature,
                        status="measurement",
                        confidence=1.0,
                        value=numeric,
                        source="SpatialEvidenceEvaluator",
                        source_version="v1",
                        quality=1.0,
                        feature_evaluability="adequate",
                        limitations=tuple(),
                        metadata={"section": section, "source_patch_ids": list(summary.source_patch_ids)},
                    )
                )
        mixing = float(summary.spatial.get("tubular_villous_mixing", 0.0))
        colocalization = float(summary.spatial.get("serrated_villous_colocalization", 0.0))
        records.extend(
            [
                self._derived_spatial_status(
                    case_id,
                    summary,
                    "tubular_villous_mixing",
                    mixing,
                ),
                self._derived_spatial_status(
                    case_id,
                    summary,
                    "serrated_villous_colocalization",
                    colocalization,
                ),
                self._derived_spatial_status(
                    case_id,
                    summary,
                    "mixed_serrated_villous_architecture",
                    max(mixing, colocalization),
                ),
            ]
        )
        return records

    @staticmethod
    def _score_status(score, evaluable):
        if evaluable < 0.5:
            return "not_evaluable", _clamp(1.0 - evaluable), "not_evaluable"
        if score >= 0.55:
            return "present", _clamp(score), "adequate"
        if score <= 0.20:
            return "absent", _clamp(1.0 - score), "adequate"
        return "uncertain", _clamp(1.0 - abs(score - 0.5)), "limited"

    @staticmethod
    def _derived_spatial_status(case_id, summary, feature, value):
        if value >= 0.45:
            status = "present"
            confidence = value
            evaluability = "adequate"
        elif value <= 0.15:
            status = "absent"
            confidence = 1.0 - value
            evaluability = "adequate"
        else:
            status = "uncertain"
            confidence = 1.0 - abs(value - 0.5)
            evaluability = "limited"
        return EvidenceRecord(
            evidence_id="SPATIAL_STATUS_{0}_{1}".format(summary.slide_id, feature),
            case_id=case_id,
            evidence_type="spatial_derived_evidence",
            feature=feature,
            status=status,
            confidence=_clamp(confidence),
            source="SpatialEvidenceEvaluator",
            source_version="v1",
            quality=1.0,
            feature_evaluability=evaluability,
            limitations=tuple(),
            metadata={"raw_value": float(value)},
        )


class ROIManager(object):
    """Builds a deterministic candidate pool; Planner never invents coordinates."""

    FIXED_EXTENTS = {2.5: 4096, 5.0: 2048, 10.0: 1024, 20.0: 512}

    def build_candidates(self, predictions, slide_dimensions=None):
        predictions = tuple(predictions)
        if not predictions:
            return tuple()
        candidates = []
        groups = defaultdict(list)
        for row in predictions:
            group_id = "component_{0}".format(row.component_ids[0]) if row.component_ids else "patch_{0}".format(row.patch_id)
            groups[group_id].append(row)
        for group_id in sorted(groups):
            rows = groups[group_id]
            overview_bbox = self._cluster_bbox(rows, 2.5, slide_dimensions)
            cluster_bbox = self._cluster_bbox(rows, 5.0, slide_dimensions)
            cluster_features, reviewers = self._cluster_features(rows)
            overview_features, overview_reviewers = self._overview_features(rows)
            candidates.append(
                ROICandidate(
                    roi_id="ROI_2P5X_{0}".format(group_id),
                    slide_id=rows[0].slide_id,
                    level0_bbox=overview_bbox,
                    scale=2.5,
                    roi_semantics="roi_overview_context",
                    candidate_features=tuple(overview_features),
                    allowed_reviewers=tuple(overview_reviewers),
                    suitability=_clamp(
                        _weighted_mean(rows, lambda row: row.evaluable, lambda row: row.mucosa_coverage)
                    ),
                    spatial_coverage=_clamp(
                        sum(row.mucosa_coverage for row in rows) / max(1.0, float(len(rows)))
                    ),
                    estimated_cost=0.15,
                    source_patch_ids=tuple(row.patch_id for row in rows),
                    source_cluster_id=group_id,
                    metadata={"candidate_source": "large_component_overview"},
                )
            )
            candidates.append(
                ROICandidate(
                    roi_id="ROI_5X_{0}".format(group_id),
                    slide_id=rows[0].slide_id,
                    level0_bbox=cluster_bbox,
                    scale=5.0,
                    roi_semantics="architecture_cluster_overview",
                    candidate_features=tuple(cluster_features),
                    allowed_reviewers=tuple(reviewers),
                    suitability=_clamp(_weighted_mean(rows, lambda row: row.evaluable, lambda row: row.mucosa_coverage)),
                    spatial_coverage=_clamp(sum(row.mucosa_coverage for row in rows) / max(1.0, float(len(rows)))),
                    estimated_cost=0.25,
                    source_patch_ids=tuple(row.patch_id for row in rows),
                    source_cluster_id=group_id,
                    metadata={
                        "candidate_source": "cluster_weighted_centroid",
                        "parent_roi_id": "ROI_2P5X_{0}".format(group_id),
                    },
                )
            )
        for row in sorted(predictions, key=lambda item: item.patch_id):
            group_id = "component_{0}".format(row.component_ids[0]) if row.component_ids else "patch_{0}".format(row.patch_id)
            parent_5x_roi_id = "ROI_5X_{0}".format(group_id)
            parent_bbox = tuple(row.level0_bbox)
            ten_bbox = self._child_bbox(parent_bbox, 10.0, slide_dimensions)
            architecture_features = self._architecture_features(row)
            candidates.append(
                ROICandidate(
                    roi_id="ROI_10X_{0}".format(row.patch_id),
                    slide_id=row.slide_id,
                    level0_bbox=ten_bbox,
                    scale=10.0,
                    roi_semantics="crypt_architecture_detail",
                    candidate_features=tuple(architecture_features),
                    allowed_reviewers=(
                        "QualityMucosaReviewer",
                        "SerratedArchitectureReviewer",
                        "TSAReviewer",
                        "ConventionalArchitectureReviewer",
                        "InflammatoryReactiveReviewer",
                    ),
                    suitability=_clamp(row.evaluable * (1.0 - 0.25 * row.uncertainty)),
                    spatial_coverage=_clamp(row.mucosa_coverage),
                    estimated_cost=0.5,
                    source_patch_ids=(row.patch_id,),
                    source_cluster_id=("component_{0}".format(row.component_ids[0]) if row.component_ids else None),
                    metadata={
                        "parent_5x_bbox": list(parent_bbox),
                        "parent_roi_id": parent_5x_roi_id,
                        "source_5x_image_path": row.image_path,
                    },
                )
            )
            hotspot = max(row.dysplasia_risk, row.abnormal_epithelial_score, row.uncertainty)
            for child_index, twenty_bbox in enumerate(self._child_quadrants(ten_bbox, 20.0)):
                candidates.append(
                    ROICandidate(
                        roi_id="ROI_20X_{0}_q{1:02d}".format(row.patch_id, child_index),
                        slide_id=row.slide_id,
                        level0_bbox=twenty_bbox,
                        scale=20.0,
                        roi_semantics="dysplasia_hotspot_candidate",
                        candidate_features=("high_grade_or_definite_dysplasia",),
                        allowed_reviewers=("QualityMucosaReviewer", "DysplasiaReviewer"),
                        suitability=_clamp(0.25 + 0.75 * hotspot - 0.01 * child_index),
                        spatial_coverage=_clamp(row.mucosa_coverage),
                        estimated_cost=0.75,
                        source_patch_ids=(row.patch_id,),
                        source_cluster_id=("component_{0}".format(row.component_ids[0]) if row.component_ids else None),
                        metadata={
                            "parent_10x_bbox": list(ten_bbox),
                            "parent_roi_id": "ROI_10X_{0}".format(row.patch_id),
                            "child_index": child_index,
                            "dysplasia_risk": row.dysplasia_risk,
                            "abnormal_epithelial_score": row.abnormal_epithelial_score,
                            "uncertainty": row.uncertainty,
                        },
                    )
                )
        return self._deduplicate(candidates)

    def _cluster_features(self, rows):
        serrated = _weighted_mean(rows, lambda row: row.architecture["serrated"], lambda row: row.mucosa_coverage)
        tubular = _weighted_mean(rows, lambda row: row.architecture["tubular"], lambda row: row.mucosa_coverage)
        villous = _weighted_mean(rows, lambda row: row.architecture["villous"], lambda row: row.mucosa_coverage)
        reactive = _weighted_mean(
            rows,
            lambda row: row.context["reactive_inflammatory_present"],
            lambda row: row.mucosa_coverage,
        )
        features = ["reviewable_mucosa"]
        reviewers = ["QualityMucosaReviewer"]
        if serrated >= 0.3:
            features.append("mixed_serrated_villous_architecture")
            reviewers.append("SerratedArchitectureReviewer")
        if tubular >= 0.3 or villous >= 0.3:
            features.extend(["villous_component_present", "tubular_villous_mixing"])
            reviewers.append("ConventionalArchitectureReviewer")
        if reactive >= 0.3:
            features.extend(
                [
                    "reactive_regenerative_change",
                    "adenomatous_architecture_absent",
                    "serrated_architecture_absent",
                ]
            )
            reviewers.append("InflammatoryReactiveReviewer")
        return sorted(set(features)), sorted(set(reviewers))

    def _overview_features(self, rows):
        serrated = _weighted_mean(rows, lambda row: row.architecture["serrated"], lambda row: row.mucosa_coverage)
        tubular = _weighted_mean(rows, lambda row: row.architecture["tubular"], lambda row: row.mucosa_coverage)
        villous = _weighted_mean(rows, lambda row: row.architecture["villous"], lambda row: row.mucosa_coverage)
        reactive = _weighted_mean(
            rows,
            lambda row: row.context["reactive_inflammatory_present"],
            lambda row: row.mucosa_coverage,
        )
        features = ["reviewable_mucosa", "epithelium_present"]
        reviewers = ["QualityMucosaReviewer"]
        if serrated >= 0.25:
            features.extend(
                [
                    "serration_present",
                    "surface_limited_serration",
                    "straight_crypt_bases",
                    "ectopic_crypt_formation",
                    "slit_like_serration",
                    "villiform_or_filiform_serrated_architecture",
                ]
            )
            reviewers.extend(["SerratedArchitectureReviewer", "TSAReviewer"])
        if tubular >= 0.25 or villous >= 0.25:
            features.extend(
                [
                    "tubular_architecture",
                    "villous_component_present",
                    "tubular_villous_mixing",
                    "conventional_adenoma_like_epithelium",
                ]
            )
            reviewers.append("ConventionalArchitectureReviewer")
        if reactive >= 0.25:
            features.extend(
                [
                    "erosion",
                    "ulceration",
                    "granulation_tissue",
                    "mixed_inflammation",
                    "lymphoid_or_stromal_dominance",
                ]
            )
            reviewers.append("InflammatoryReactiveReviewer")
        return sorted(set(features)), sorted(set(reviewers))

    @staticmethod
    def _architecture_features(row):
        features = ["reviewable_mucosa"]
        if row.architecture["serrated"] >= 0.25:
            features.extend(
                [
                    "serration_to_crypt_base",
                    "basal_crypt_dilation",
                    "surface_limited_serration",
                    "straight_crypt_bases",
                    "crypt_branching",
                    "horizontal_or_boot_shaped_crypt",
                    "ectopic_crypt_formation",
                    "slit_like_serration",
                    "villiform_or_filiform_serrated_architecture",
                    "cytoplasmic_eosinophilia",
                    "pencillate_nuclei",
                    "mixed_serrated_villous_architecture",
                ]
            )
        if row.architecture["tubular"] >= 0.25 or row.architecture["villous"] >= 0.25:
            features.extend(
                [
                    "villous_component_present",
                    "tubular_villous_mixing",
                    "crowded_adenomatous_glands",
                ]
            )
        if row.context["reactive_inflammatory_present"] >= 0.25:
            features.extend(
                [
                    "reactive_regenerative_change",
                    "adenomatous_architecture_absent",
                    "serrated_architecture_absent",
                    "dysplasia_not_evaluable_due_to_reactive_change",
                    "erosion",
                    "mixed_inflammation",
                ]
            )
        return sorted(set(features))

    def _cluster_bbox(self, rows, scale, slide_dimensions):
        weights = [max(1e-6, row.mucosa_coverage * row.evaluable) for row in rows]
        centers_x = [(row.level0_bbox[0] + row.level0_bbox[2]) / 2.0 for row in rows]
        centers_y = [(row.level0_bbox[1] + row.level0_bbox[3]) / 2.0 for row in rows]
        total = sum(weights)
        center_x = sum(value * weight for value, weight in zip(centers_x, weights)) / total
        center_y = sum(value * weight for value, weight in zip(centers_y, weights)) / total
        return self._centered_bbox(center_x, center_y, self.FIXED_EXTENTS[scale], slide_dimensions)

    def _child_bbox(self, parent_bbox, scale, slide_dimensions):
        extent = self.FIXED_EXTENTS[scale]
        center_x = (parent_bbox[0] + parent_bbox[2]) / 2.0
        center_y = (parent_bbox[1] + parent_bbox[3]) / 2.0
        child = self._centered_bbox(center_x, center_y, extent, slide_dimensions)
        # If slide clipping shifts the child, clamp it back inside its parent.
        x1 = min(max(child[0], parent_bbox[0]), parent_bbox[2] - extent)
        y1 = min(max(child[1], parent_bbox[1]), parent_bbox[3] - extent)
        return (int(x1), int(y1), int(x1 + extent), int(y1 + extent))

    def _child_quadrants(self, parent_bbox, scale):
        extent = self.FIXED_EXTENTS[scale]
        x1, y1, x2, y2 = parent_bbox
        if x2 - x1 < 2 * extent or y2 - y1 < 2 * extent:
            return (self._child_bbox(parent_bbox, scale, None),)
        return (
            (x1, y1, x1 + extent, y1 + extent),
            (x1 + extent, y1, x1 + 2 * extent, y1 + extent),
            (x1, y1 + extent, x1 + extent, y1 + 2 * extent),
            (x1 + extent, y1 + extent, x1 + 2 * extent, y1 + 2 * extent),
        )

    @staticmethod
    def _centered_bbox(center_x, center_y, extent, slide_dimensions):
        x1 = int(round(center_x - extent / 2.0))
        y1 = int(round(center_y - extent / 2.0))
        x1 = max(0, x1)
        y1 = max(0, y1)
        if slide_dimensions:
            width, height = [int(value) for value in slide_dimensions]
            x1 = max(0, min(x1, max(0, width - extent)))
            y1 = max(0, min(y1, max(0, height - extent)))
        return (x1, y1, x1 + extent, y1 + extent)

    @staticmethod
    def _deduplicate(candidates):
        output = []
        seen = set()
        for candidate in sorted(candidates, key=lambda item: (item.scale, item.roi_id)):
            key = (candidate.scale, candidate.level0_bbox, candidate.allowed_reviewers, candidate.candidate_features)
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        return tuple(output)
