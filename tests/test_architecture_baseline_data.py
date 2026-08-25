import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from adenoma_agent.architecture_baselines.data import (
    CanonicalLabel,
    annotation_guard,
    audit_architecture_data,
    audit_label_rows,
    build_case_paths,
    build_canonical_labels,
    build_five_fold_splits,
    load_xlsx_rows,
    write_five_fold_split_artifacts,
    yx_family_heuristic,
)


def _write_xlsx(path, rows):
    shared = []
    for row in rows:
        for value in row:
            if value not in shared:
                shared.append(value)
    shared_xml = '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{0}</sst>'.format(
        "".join("<si><t>{0}</t></si>".format(value) for value in shared)
    )
    sheet_rows = []
    for row_index, row in enumerate(rows, 1):
        cells = []
        for col_index, value in enumerate(row):
            ref = chr(ord("A") + col_index) + str(row_index)
            cells.append('<c r="{0}" t="s"><v>{1}</v></c>'.format(ref, shared.index(value)))
        sheet_rows.append('<row r="{0}">{1}</row>'.format(row_index, "".join(cells)))
    sheet_xml = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>{0}</sheetData></worksheet>"
    ).format("".join(sheet_rows))
    workbook_xml = (
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" '
        'Target="sharedStrings.xml"/></Relationships>'
    )
    with ZipFile(str(path), "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_xml)


def _seven_class_rows(count_per_class=5):
    labels = [
        "Hyperplastic polyps",
        "Inflammatory polyp",
        "Sessile serrated adenoma",
        "Traditional serrated adenoma",
        "Tubular adenoma",
        "Tubulovillous adenoma",
        "Unclassified serrated adenoma",
    ]
    rows = []
    for class_index, label in enumerate(labels):
        for item in range(count_per_class):
            rows.append(
                CanonicalLabel(
                    case_alias="CASE_{0}_{1}".format(class_index, item),
                    source_code="YX",
                    label=label,
                    grade="low",
                    family_id="FAMILY_{0}_{1}".format(class_index, item),
                    label_source_sha256="workbook-sha",
                )
            )
    return rows


class ArchitectureBaselineDataTest(unittest.TestCase):
    def test_xlsx_loader_conflicts_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workbook = Path(tmpdir) / "labels.xlsx"
            _write_xlsx(
                workbook,
                [
                    ["slide_name", "type", "grade"],
                    ["a", "Tubular adenoma", "low"],
                    ["a", "Tubular adenoma", "low"],
                    ["b", "Hyperplastic polyps", "low"],
                    ["b", "Tubular adenoma", "high"],
                ],
            )
            rows = load_xlsx_rows(workbook)
            audit = audit_label_rows(rows)
            self.assertEqual(rows[0]["slide_name"], "a")
            self.assertEqual(audit["total_rows"], 4)
            self.assertEqual(audit["unique_slide_keys"], 2)
            self.assertEqual(audit["conflicting_slide_keys"], ["b"])
            self.assertEqual(audit["class_vocabulary"], ["Tubular adenoma"])

    def test_alias_provenance_and_family_are_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yx = Path(tmpdir) / "yx"
            yx.mkdir()
            (yx / "138265_746091001.svs").write_bytes(b"wsi")
            (yx / "138266_746091002.svs").write_bytes(b"wsi")
            rows = [
                {"slide_name": "138265_746091001", "type": "Tubular adenoma", "grade": "low"},
                {"slide_name": "138266_746091002", "type": "Hyperplastic polyps", "grade": "low"},
            ]
            cases = build_case_paths({"YX": yx}, rows)
            labels = build_canonical_labels(cases, rows, "sha")
            self.assertEqual(yx_family_heuristic("138265_746091001"), "YX_FAMILY_746091")
            self.assertEqual(cases[0].family_id, cases[1].family_id)
            self.assertNotIn("source_path", cases[0].inference_identity())
            self.assertTrue(cases[0].local_provenance()["source_path"].endswith("138265_746091001.svs"))
            self.assertEqual(labels[0].source_code, "YX")

    def test_five_fold_split_has_no_group_leakage_and_writes_training_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            labels = _seven_class_rows()
            vocabulary = sorted({row.label for row in labels})
            folds = build_five_fold_splits(labels, seed=23, expected_labels=vocabulary)
            self.assertEqual(len(folds), 5)
            for fold in folds:
                self.assertTrue(set(fold.test_case_aliases).isdisjoint(fold.val_case_aliases))
                self.assertTrue(set(fold.test_case_aliases).isdisjoint(fold.train_case_aliases))
                self.assertTrue(set(fold.val_case_aliases).isdisjoint(fold.train_case_aliases))
                self.assertEqual(set(fold.label_support["test"]), set(vocabulary))
                self.assertEqual(set(fold.label_support["val"]), set(vocabulary))
                self.assertEqual(set(fold.label_support["train"]), set(vocabulary))
            output = Path(tmpdir) / "splits"
            write_five_fold_split_artifacts(
                labels,
                output,
                eligible_aliases=[row.case_alias for row in labels],
            )
            payload = json.loads((output / "fold_0" / "test_cases.json").read_text(encoding="utf-8"))
            training_rows = [
                json.loads(line)
                for line in (output / "training_labels.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(payload["case_aliases"])
            self.assertTrue(payload["sha256"])
            self.assertEqual(len(training_rows), len(labels))
            self.assertEqual(set(row["label_index"] for row in training_rows), set(range(7)))

    def test_annotation_missing_is_blocked_and_never_broadcast(self):
        result = annotation_guard(
            [{"slide_id": "slide-a", "patch_id": "p1"}],
            annotation_rows=[
                {"roi_id": "synthetic", "synthetic_smoke_only": True, "label": "Tubular adenoma"}
            ],
            slide_labels={"slide-a": "Tubular adenoma"},
        )
        self.assertEqual(result["status"], "annotation_blocked")
        self.assertEqual(result["targets"], [])
        self.assertFalse(result["slide_label_broadcast"])

    def test_inventory_separates_local_paths_from_shareable_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workbook = root / "labels.xlsx"
            _write_xlsx(workbook, [["slide_name", "type", "grade"], ["a", "Tubular adenoma", "low"]])
            yx = root / "yx"
            hp = root / "hp"
            yx.mkdir()
            hp.mkdir()
            (yx / "a.svs").write_bytes(b"a")
            (hp / "unknown.isyntax").write_bytes(b"h")
            audit = audit_architecture_data(workbook, yx_root=yx, hp_root=hp, output_dir=root / "audit")
            case_row = json.loads(Path(audit.artifact_paths["case_paths"]).read_text(encoding="utf-8").splitlines()[0])
            summary = json.loads(Path(audit.artifact_paths["audit_summary"]).read_text(encoding="utf-8"))
            self.assertEqual(audit.splits, ())
            self.assertIn("source_path", case_row)
            self.assertNotIn("root", summary["sources"]["YX"])
            self.assertNotIn("source_path", json.dumps(summary))
            self.assertTrue(Path(audit.artifact_paths["canonical_labels"]).is_file())


if __name__ == "__main__":
    unittest.main()
