import tempfile
import unittest
from pathlib import Path

from adenoma_agent.adapters.manifest import AdenomaManifestAdapter


class ManifestAdapterTest(unittest.TestCase):
    def test_build_pilot_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            labels = Path(tmpdir) / "labels.csv"
            manifest.write_text(
                "slide_id,slide_filename,slide_path\ncase_pos,case_pos.svs,/tmp/case_pos.svs\ncase_neg,case_neg.svs,/tmp/case_neg.svs\n",
                encoding="utf-8",
            )
            labels.write_text(
                "slide_id,type,grade\ncase_pos,Sessile serrated adenoma,low\ncase_neg,Hyperplastic polyps,low\n",
                encoding="utf-8",
            )
            adapter = AdenomaManifestAdapter(str(manifest), str(labels))
            cases = adapter.list_cases()
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].binary_target, 1)
            self.assertEqual(cases[1].binary_target, 0)
            output_path = Path(tmpdir) / "pilot.json"
            payload = adapter.build_pilot_subset(output_path, positives=1, negatives=1)
            self.assertEqual(payload["selected_case_count"], 2)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
