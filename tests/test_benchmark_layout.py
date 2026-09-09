import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "baseline/adenoma_11class"


def test_benchmark_class_mapping_and_report_order():
    mapping = json.loads((BENCHMARK / "class_mapping.json").read_text(encoding="utf-8"))
    classes = mapping["model_class_order"]
    assert [item["id"] for item in classes] == list(range(11))
    assert [item["name"] for item in classes] == [
        "SSL", "HP", "TSA", "USA", "TA", "TVA", "IP", "SSLD", "TSAD", "TAD", "TVAD"
    ]
    assert mapping["report_display_order"] == [
        "IP", "HP", "SSL", "SSLD", "TSA", "TSAD", "USA", "TA", "TAD", "TVA", "TVAD"
    ]


def test_weight_manifest_has_all_70_expected_checkpoints():
    payload = json.loads((BENCHMARK / "weights/weights_manifest.json").read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    assert payload["expected_artifacts"] == 70
    assert len(artifacts) == 70
    keys = {(item["model"], item["feature"], item["fold"]) for item in artifacts}
    assert len(keys) == 70
    assert all(item["historical_fold"] == 5 for item in artifacts if item["fold"] == 4)
    assert all(item["historical_fold"] == item["fold"] for item in artifacts if item["fold"] < 4)
