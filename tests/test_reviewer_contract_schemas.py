import copy
import json
import unittest
from collections import Counter
from pathlib import Path


try:
    from jsonschema import Draft202012Validator, FormatChecker, RefResolver
except ImportError:  # Optional contract-only dependency.
    Draft202012Validator = None
    FormatChecker = None
    RefResolver = None

try:
    from referencing import Registry, Resource
except ImportError:  # jsonschema 4.0-4.17 use the legacy resolver path below.
    Registry = None
    Resource = None


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "model" / "contracts" / "schemas"
EXAMPLE_DIR = ROOT / "docs" / "model" / "contracts" / "examples" / "reviewer"

SCHEMA_FILES = {
    "common": SCHEMA_DIR / "reviewer_common_v1.schema.json",
    "registry": SCHEMA_DIR / "reviewer_registry_v1.schema.json",
    "request": SCHEMA_DIR / "reviewer_task_request_v1.schema.json",
    "observation": SCHEMA_DIR / "reviewer_observation_v1.schema.json",
    "ledger": SCHEMA_DIR / "reviewer_ledger_record_v1.schema.json",
}

REVIEWERS = {
    "QualityMucosaReviewer",
    "SerratedArchitectureReviewer",
    "TSAReviewer",
    "ConventionalArchitectureReviewer",
    "DysplasiaReviewer",
    "InflammatoryReactiveReviewer",
}

OVERVIEW_PROFILES = {
    "overview_evaluability",
    "serrated_overview",
    "tsa_overview",
    "conventional_overview",
    "inflammatory_overview",
}

FORBIDDEN_MODEL_INPUT_KEYS = {
    "hypothesis",
    "hypotheses",
    "hypothesis_ranking",
    "expected_answer",
    "expected_effect",
    "final_diagnosis",
    "final_label",
    "final_label_mapping",
    "branch_context",
    "branch_preference",
    "action_score",
}

FORBIDDEN_REVIEWER_OUTPUT_KEYS = {
    "final_diagnosis",
    "final_label",
    "final_label_mapping",
    "D_suffix_label",
    "hypothesis_ranking",
    "next_action",
    "roi_selection",
}


def _load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            for child_key in _walk_keys(nested):
                yield child_key
    elif isinstance(value, list):
        for item in value:
            for child_key in _walk_keys(item):
                yield child_key


class ReviewerContractSemanticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = _load_json(SCHEMA_DIR / "reviewer_registry_v1.json")
        cls.requests = {
            item["request_id"]: item
            for item in (
                _load_json(path)
                for path in sorted(EXAMPLE_DIR.glob("*_request.json"))
            )
        }
        cls.observations = {
            item["request_id"]: item
            for item in (
                _load_json(path)
                for path in sorted(EXAMPLE_DIR.glob("*_observation*.json"))
            )
        }
        cls.ledger_records = [
            _load_json(path) for path in sorted(EXAMPLE_DIR.glob("ledger_*.json"))
        ]
        cls.profile_index = {}
        cls.reviewer_index = {}
        for reviewer in cls.registry["reviewers"]:
            cls.reviewer_index[reviewer["reviewer"]] = reviewer
            for profile in reviewer["task_profiles"]:
                cls.profile_index[(reviewer["reviewer"], profile["task_profile"])] = profile

    def test_registry_has_six_reviewers_and_four_scale_semantics(self):
        self.assertEqual(set(self.reviewer_index), REVIEWERS)
        self.assertEqual(
            [item["magnification"] for item in self.registry["scale_semantics"]],
            [2.5, 5, 10, 20],
        )
        self.assertTrue(self.registry["logical_reviewers_may_share_backend"])
        self.assertEqual(self.registry["retry_policy"]["transport_or_model_retries"], 1)
        self.assertEqual(self.registry["retry_policy"]["schema_repair_attempts"], 1)
        self.assertEqual(self.registry["retry_policy"]["max_conflict_resolution_rounds"], 2)
        self.assertFalse(self.registry["retry_policy"]["retry_same_roi_when_not_evaluable"])

    def test_registry_profiles_are_unique_and_allowlists_are_consistent(self):
        profile_names = []
        for reviewer in self.registry["reviewers"]:
            reviewer_features = set(reviewer["feature_allowlist"])
            for profile in reviewer["task_profiles"]:
                profile_names.append(profile["task_profile"])
                targets = set(profile["target_feature_allowlist"])
                incidental = set(profile["incidental_feature_allowlist"])
                self.assertTrue(targets <= reviewer_features)
                self.assertTrue(incidental <= reviewer_features)
                self.assertFalse(targets & incidental)
                if 2.5 in profile["allowed_primary_magnifications"]:
                    self.assertIn(profile["task_profile"], OVERVIEW_PROFILES)
                for scale in profile["allowed_primary_magnifications"]:
                    self.assertIn(scale, {2.5, 5, 10, 20})
                context_roles = [item["view_role"] for item in profile["allowed_context_views"]]
                self.assertEqual(len(context_roles), len(set(context_roles)))
        self.assertEqual(len(profile_names), 14)
        self.assertEqual(len(profile_names), len(set(profile_names)))

    def test_tsa_signature_and_hgd_feature_ownership_are_disjoint(self):
        tsa_features = set(self.reviewer_index["TSAReviewer"]["feature_allowlist"])
        dysplasia_features = set(
            self.reviewer_index["DysplasiaReviewer"]["feature_allowlist"]
        )
        self.assertFalse(tsa_features & dysplasia_features)
        self.assertNotIn("high_grade_or_definite_dysplasia", tsa_features)
        self.assertNotIn("high_grade_focus", tsa_features)
        self.assertNotIn("cytoplasmic_eosinophilia", dysplasia_features)
        self.assertNotIn("pencillate_nuclei", dysplasia_features)
        tsa_profile = self.profile_index[("TSAReviewer", "tsa_signature_cytology")]
        tsa_allowed = set(tsa_profile["target_feature_allowlist"]) | set(
            tsa_profile["incidental_feature_allowlist"]
        )
        self.assertNotIn("high_grade_or_definite_dysplasia", tsa_allowed)
        dysplasia_profile = self.profile_index[
            ("DysplasiaReviewer", "high_grade_dysplasia_assessment")
        ]
        self.assertEqual(dysplasia_profile["allowed_primary_magnifications"], [20])

    def test_examples_cover_every_reviewer_and_pair_requests_with_observations(self):
        self.assertEqual({item["reviewer"] for item in self.requests.values()}, REVIEWERS)
        self.assertEqual(set(self.requests), set(self.observations))
        for request_id, request in self.requests.items():
            observation = self.observations[request_id]
            model_input = request["model_input"]
            profile_key = (request["reviewer"], model_input["task_profile"])
            self.assertIn(profile_key, self.profile_index)
            profile = self.profile_index[profile_key]

            self.assertEqual(observation["reviewer"], request["reviewer"])
            self.assertEqual(observation["task_profile"], model_input["task_profile"])
            self.assertEqual(observation["primary_roi_id"], model_input["primary_roi"]["roi_id"])
            self.assertEqual(
                observation["primary_magnification"],
                model_input["primary_roi"]["magnification"],
            )
            self.assertEqual(set(observation["target_features"]), set(model_input["target_features"]))

            target_ids = [item["feature_id"] for item in observation["findings"]]
            incidental_ids = [item["feature_id"] for item in observation["incidental_findings"]]
            self.assertEqual(Counter(target_ids), Counter(model_input["target_features"]))
            self.assertEqual(len(incidental_ids), len(set(incidental_ids)))
            self.assertFalse(set(target_ids) & set(incidental_ids))
            self.assertTrue(
                set(incidental_ids) <= set(profile["incidental_feature_allowlist"])
            )

            self.assertIn(
                model_input["primary_roi"]["magnification"],
                profile["allowed_primary_magnifications"],
            )
            self._assert_context_views_match_profile(model_input["context_views"], profile)
            self._assert_bbox_is_ordered(model_input["primary_roi"]["level0_bbox"])
            for context in model_input["context_views"]:
                self._assert_bbox_is_ordered(context["roi"]["level0_bbox"])

            forbidden_keys = set(_walk_keys(model_input)) & FORBIDDEN_MODEL_INPUT_KEYS
            self.assertFalse(forbidden_keys)
            self._assert_finding_semantics(observation["findings"])
            self._assert_finding_semantics(observation["incidental_findings"])
            self.assertEqual(self._request_contract_errors(request), [])
            self.assertEqual(
                self._observation_contract_errors(request, observation), []
            )

    def test_semantic_checks_reject_registry_and_pairing_mismatches(self):
        unknown_reviewer = copy.deepcopy(
            self.requests["review_request_serrated_001"]
        )
        unknown_reviewer["reviewer"] = "ConflictReviewer"
        self.assertTrue(self._request_contract_errors(unknown_reviewer))

        wrong_profile = copy.deepcopy(
            self.requests["review_request_serrated_001"]
        )
        wrong_profile["model_input"]["task_profile"] = "tsa_architecture"
        self.assertTrue(self._request_contract_errors(wrong_profile))

        dysplasia_request = copy.deepcopy(self.requests["review_request_dysplasia_001"])
        dysplasia_request["model_input"]["primary_roi"]["magnification"] = 10
        dysplasia_profile = self.profile_index[
            ("DysplasiaReviewer", "high_grade_dysplasia_assessment")
        ]
        self.assertNotIn(
            dysplasia_request["model_input"]["primary_roi"]["magnification"],
            dysplasia_profile["allowed_primary_magnifications"],
        )
        self.assertTrue(self._request_contract_errors(dysplasia_request))

        tsa_observation = copy.deepcopy(self.observations["review_request_tsa_001"])
        tsa_observation["findings"][0]["feature_id"] = "high_grade_or_definite_dysplasia"
        tsa_profile = self.profile_index[("TSAReviewer", "tsa_signature_cytology")]
        tsa_allowed = set(tsa_profile["target_feature_allowlist"]) | set(
            tsa_profile["incidental_feature_allowlist"]
        )
        self.assertFalse(
            {item["feature_id"] for item in tsa_observation["findings"]} <= tsa_allowed
        )
        self.assertTrue(
            self._observation_contract_errors(
                self.requests["review_request_tsa_001"], tsa_observation
            )
        )

        serrated_request = self.requests["review_request_serrated_001"]
        missing_finding = copy.deepcopy(self.observations["review_request_serrated_001"])
        missing_finding["findings"].pop()
        self.assertNotEqual(
            Counter(item["feature_id"] for item in missing_finding["findings"]),
            Counter(serrated_request["model_input"]["target_features"]),
        )
        self.assertTrue(
            self._observation_contract_errors(serrated_request, missing_finding)
        )

        duplicate_finding = copy.deepcopy(
            self.observations["review_request_serrated_001"]
        )
        duplicate = copy.deepcopy(duplicate_finding["findings"][0])
        duplicate["evidence_text"] = "Duplicate return for the same target feature."
        duplicate_finding["findings"].append(duplicate)
        self.assertNotEqual(
            Counter(item["feature_id"] for item in duplicate_finding["findings"]),
            Counter(serrated_request["model_input"]["target_features"]),
        )
        self.assertTrue(
            self._observation_contract_errors(serrated_request, duplicate_finding)
        )

        leaked_context = copy.deepcopy(serrated_request["model_input"])
        leaked_context["hypothesis_ranking"] = ["ssl", "hp"]
        self.assertTrue(set(_walk_keys(leaked_context)) & FORBIDDEN_MODEL_INPUT_KEYS)
        leaked_request = copy.deepcopy(serrated_request)
        leaked_request["model_input"] = leaked_context
        self.assertTrue(self._request_contract_errors(leaked_request))

        dysplasia_with_suffix = copy.deepcopy(
            self.observations["review_request_dysplasia_001"]
        )
        dysplasia_with_suffix["final_label"] = "TAD"
        self.assertTrue(
            set(_walk_keys(dysplasia_with_suffix)) & FORBIDDEN_MODEL_INPUT_KEYS
        )
        self.assertTrue(
            self._observation_contract_errors(
                self.requests["review_request_dysplasia_001"],
                dysplasia_with_suffix,
            )
        )

    def _assert_context_views_match_profile(self, context_views, profile):
        policies = {item["view_role"]: item for item in profile["allowed_context_views"]}
        counts = Counter(item["view_role"] for item in context_views)
        for view in context_views:
            role = view["view_role"]
            self.assertIn(role, policies)
            self.assertIn(view["roi"]["magnification"], policies[role]["allowed_magnifications"])
        for role, count in counts.items():
            self.assertLessEqual(count, policies[role]["max_count"])

    def _assert_bbox_is_ordered(self, bbox):
        self.assertLess(bbox[0], bbox[2])
        self.assertLess(bbox[1], bbox[3])

    def _assert_finding_semantics(self, findings):
        for finding in findings:
            self.assertIn(finding["scope"], {"roi_local", "roi_overview"})
            self.assertGreaterEqual(finding["status_confidence"], 0)
            self.assertLessEqual(finding["status_confidence"], 1)
            if finding["status"] == "absent":
                self.assertEqual(finding["feature_evaluability"], "adequate")
            if finding["status"] == "not_evaluable":
                self.assertEqual(finding["feature_evaluability"], "not_evaluable")
            if finding["feature_evaluability"] == "not_evaluable":
                self.assertEqual(finding["status"], "not_evaluable")
            if finding["feature_id"] == "villous_component_extent_estimate":
                self.assertIn("quantitation", finding)
            else:
                self.assertNotIn("quantitation", finding)

    def _request_contract_errors(self, request):
        """Test-only registry-aware checks that JSON Schema cannot express alone."""
        errors = []
        reviewer = request.get("reviewer")
        model_input = request.get("model_input", {})
        profile_name = model_input.get("task_profile")
        profile = self.profile_index.get((reviewer, profile_name))

        if reviewer not in self.reviewer_index:
            errors.append("unknown reviewer")
        if profile is None:
            errors.append("reviewer/profile mismatch")

        forbidden = set(_walk_keys(model_input)) & FORBIDDEN_MODEL_INPUT_KEYS
        if forbidden:
            errors.append("diagnostic context leaked into model_input")

        targets = model_input.get("target_features", [])
        if len(targets) != len(set(targets)):
            errors.append("duplicate target feature")

        if profile is None:
            return errors

        primary_roi = model_input.get("primary_roi", {})
        primary_scale = primary_roi.get("magnification")
        if primary_scale not in profile["allowed_primary_magnifications"]:
            errors.append("primary scale not allowed for profile")
        if primary_scale == 2.5 and profile_name not in OVERVIEW_PROFILES:
            errors.append("2.5x used outside overview/evaluability profile")

        target_allowlist = set(profile["target_feature_allowlist"])
        if not set(targets) <= target_allowlist:
            errors.append("target feature outside profile allowlist")

        policies = {
            item["view_role"]: item for item in profile["allowed_context_views"]
        }
        context_counts = Counter()
        for context in model_input.get("context_views", []):
            role = context.get("view_role")
            context_counts[role] += 1
            policy = policies.get(role)
            if policy is None:
                errors.append("context role not allowed for profile")
                continue
            scale = context.get("roi", {}).get("magnification")
            if scale not in policy["allowed_magnifications"]:
                errors.append("context scale not allowed for role")
        for role, count in context_counts.items():
            policy = policies.get(role)
            if policy is not None and count > policy["max_count"]:
                errors.append("too many context views for role")
        return errors

    def _observation_contract_errors(self, request, observation):
        """Test-only request/observation pairing and feature-ownership checks."""
        errors = []
        model_input = request["model_input"]
        profile = self.profile_index.get(
            (request.get("reviewer"), model_input.get("task_profile"))
        )

        identity_pairs = [
            (observation.get("request_id"), request.get("request_id")),
            (observation.get("reviewer"), request.get("reviewer")),
            (observation.get("task_profile"), model_input.get("task_profile")),
            (
                observation.get("primary_roi_id"),
                model_input.get("primary_roi", {}).get("roi_id"),
            ),
            (
                observation.get("primary_magnification"),
                model_input.get("primary_roi", {}).get("magnification"),
            ),
        ]
        if any(actual != expected for actual, expected in identity_pairs):
            errors.append("request/observation identity mismatch")

        requested = model_input.get("target_features", [])
        returned_targets = [
            item.get("feature_id") for item in observation.get("findings", [])
        ]
        if Counter(returned_targets) != Counter(requested):
            errors.append("target features not returned exactly once")
        if Counter(observation.get("target_features", [])) != Counter(requested):
            errors.append("observation target_features mismatch")

        incidental = [
            item.get("feature_id")
            for item in observation.get("incidental_findings", [])
        ]
        if len(incidental) != len(set(incidental)):
            errors.append("duplicate incidental feature")
        if set(incidental) & set(requested):
            errors.append("incidental finding duplicates a target")
        if profile is None:
            errors.append("unknown reviewer/profile")
        elif not set(incidental) <= set(profile["incidental_feature_allowlist"]):
            errors.append("incidental feature outside profile allowlist")

        findings = observation.get("findings", []) + observation.get(
            "incidental_findings", []
        )
        for finding in findings:
            status = finding.get("status")
            evaluability = finding.get("feature_evaluability")
            if status == "absent" and evaluability != "adequate":
                errors.append("absent finding lacks adequate evaluability")
            if status == "not_evaluable" and evaluability != "not_evaluable":
                errors.append("not_evaluable status/evaluability mismatch")
            if evaluability == "not_evaluable" and status != "not_evaluable":
                errors.append("not_evaluable evaluability/status mismatch")
            if finding.get("feature_id") == "villous_component_extent_estimate":
                if "quantitation" not in finding:
                    errors.append("villous extent missing quantitation")
            elif "quantitation" in finding:
                errors.append("quantitation used outside villous extent")

        forbidden = set(_walk_keys(observation)) & FORBIDDEN_REVIEWER_OUTPUT_KEYS
        if forbidden:
            errors.append("reviewer emitted diagnostic or routing output")
        if observation.get("does_not_decide_final_diagnosis") is not True:
            errors.append("final-diagnosis boundary flag is not true")
        return errors

    def test_not_evaluable_is_valid_evidence_and_failure_is_separate(self):
        quality = self.observations["review_request_quality_001"]
        self.assertEqual(quality["quality"]["overall_evaluability"], "not_evaluable")
        self.assertGreater(quality["quality"]["status_confidence"], 0.9)

        records_by_name = {item["record_id"]: item for item in self.ledger_records}
        not_evaluable = records_by_name["reviewer_ledger_not_evaluable_001"]
        failure = records_by_name["reviewer_ledger_failure_001"]
        self.assertEqual(not_evaluable["record_type"], "evidence")
        self.assertEqual(
            not_evaluable["observation"]["quality"]["overall_evaluability"],
            "not_evaluable",
        )
        self.assertEqual(failure["record_type"], "invocation_failure")
        self.assertNotIn("observation", failure)

    def test_ledger_references_are_consistent(self):
        for record in self.ledger_records:
            provenance = record["provenance"]
            self.assertEqual(provenance["request_id"], record["request_id"])
            self.assertEqual(provenance["action_id"], record["action_id"])
            self.assertEqual(provenance["snapshot_id"], record["snapshot_id"])
            self.assertEqual(provenance["reviewer"], record["reviewer"])
            if record["record_type"] == "evidence":
                observation = record["observation"]
                self.assertEqual(observation["request_id"], record["request_id"])
                self.assertEqual(observation["reviewer"], record["reviewer"])
                self.assertEqual(observation["task_profile"], record["task_profile"])
                self.assertEqual(observation["primary_roi_id"], provenance["roi_id"])
                self.assertEqual(observation["primary_magnification"], provenance["magnification"])

    def test_design_documents_name_the_six_reviewers_and_scale_boundary(self):
        paths = [
            ROOT / "docs" / "Agent_workflow.md",
            ROOT / "docs" / "model" / "architecture_evidence_reviewer_design.md",
            ROOT / "docs" / "model" / "contracts" / "reviewer.md",
            ROOT / "docs" / "model" / "contracts" / "chief_pathologist.md",
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for reviewer in REVIEWERS:
                self.assertIn(reviewer, text, msg=str(path))
            for scale in ("2.5x", "5x", "10x", "20x"):
                self.assertIn(scale, text, msg=str(path))
            self.assertIn("high-grade", text, msg=str(path))


@unittest.skipUnless(
    Draft202012Validator is not None,
    "install the test-only contract extra: pip install -e '.[contracts]'",
)
class ReviewerContractJsonSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = {}
        for name, path in SCHEMA_FILES.items():
            schema = _load_json(path)
            schema["$id"] = path.resolve().as_uri()
            cls.schemas[name] = schema
        if Registry is not None:
            resources = [
                (schema["$id"], Resource.from_contents(schema))
                for schema in cls.schemas.values()
            ]
            resource_registry = Registry().with_resources(resources)
            cls.validators = {
                name: Draft202012Validator(
                    schema,
                    registry=resource_registry,
                    format_checker=FormatChecker(),
                )
                for name, schema in cls.schemas.items()
                if name != "common"
            }
        else:
            store = {}
            for name, schema in cls.schemas.items():
                store[schema["$id"]] = schema
                store[SCHEMA_FILES[name].name] = schema
            cls.validators = {}
            for name, schema in cls.schemas.items():
                if name == "common":
                    continue
                resolver = RefResolver.from_schema(schema, store=store)
                cls.validators[name] = Draft202012Validator(
                    schema,
                    resolver=resolver,
                    format_checker=FormatChecker(),
                )

    def assertValid(self, schema_name, instance):
        errors = sorted(
            self.validators[schema_name].iter_errors(instance),
            key=lambda error: list(error.absolute_path),
        )
        self.assertFalse(errors, "\n".join(error.message for error in errors))

    def assertInvalid(self, schema_name, instance):
        self.assertTrue(list(self.validators[schema_name].iter_errors(instance)))

    def test_schemas_pass_draft_2020_12_meta_validation(self):
        for schema in self.schemas.values():
            Draft202012Validator.check_schema(schema)

    def test_registry_and_all_examples_validate(self):
        self.assertValid("registry", _load_json(SCHEMA_DIR / "reviewer_registry_v1.json"))
        for path in sorted(EXAMPLE_DIR.glob("*.json")):
            instance = _load_json(path)
            schema_name = {
                "reviewer_task_request_v1": "request",
                "reviewer_observation_v1": "observation",
                "reviewer_ledger_record_v1": "ledger",
            }[instance["schema_version"]]
            self.assertValid(schema_name, instance)

    def test_schema_rejects_contract_boundary_violations(self):
        request = _load_json(EXAMPLE_DIR / "serrated_architecture_request.json")

        evaluability_recheck = copy.deepcopy(request)
        evaluability_recheck["model_input"]["feature_disagreements"] = [
            {
                "feature_id": "basal_crypt_dilation",
                "disagreement_kind": "evaluability_disagreement",
                "observed_statuses": ["absent", "absent"],
                "observed_evaluabilities": ["adequate", "limited"],
                "source_observation_ids": ["observation_a", "observation_b"],
            }
        ]
        self.assertValid("request", evaluability_recheck)

        bad_reviewer = copy.deepcopy(request)
        bad_reviewer["reviewer"] = "ConflictReviewer"
        self.assertInvalid("request", bad_reviewer)

        bad_profile = copy.deepcopy(request)
        bad_profile["model_input"]["task_profile"] = "unknown_profile"
        self.assertInvalid("request", bad_profile)

        serrated = _load_json(EXAMPLE_DIR / "serrated_architecture_observation.json")
        final_diagnosis = copy.deepcopy(serrated)
        final_diagnosis["final_diagnosis"] = "SSL"
        self.assertInvalid("observation", final_diagnosis)

        bad_confidence = copy.deepcopy(serrated)
        bad_confidence["findings"][0]["status_confidence"] = 1.01
        self.assertInvalid("observation", bad_confidence)

        invalid_absence = copy.deepcopy(serrated)
        invalid_absence["findings"][2]["feature_evaluability"] = "limited"
        self.assertInvalid("observation", invalid_absence)

        invalid_status = copy.deepcopy(serrated)
        invalid_status["findings"][0]["status"] = "negative"
        self.assertInvalid("observation", invalid_status)

        dysplasia = _load_json(EXAMPLE_DIR / "dysplasia_observation.json")
        dysplasia_with_suffix = copy.deepcopy(dysplasia)
        dysplasia_with_suffix["final_label"] = "TAD"
        self.assertInvalid("observation", dysplasia_with_suffix)

        ledger = _load_json(EXAMPLE_DIR / "ledger_evidence.json")
        missing_provenance = copy.deepcopy(ledger)
        del missing_provenance["provenance"]
        self.assertInvalid("ledger", missing_provenance)


if __name__ == "__main__":
    unittest.main()
