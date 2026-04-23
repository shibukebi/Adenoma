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
                "slide_id,slide_filename,slide_path\n"
                "case_ssl,case_ssl.svs,/tmp/case_ssl.svs\n"
                "case_hp,case_hp.svs,/tmp/case_hp.svs\n"
                "case_ta,case_ta.svs,/tmp/case_ta.svs\n",
                encoding="utf-8",
            )
            labels.write_text(
                "slide_id,type,grade\n"
                "case_ssl,Sessile serrated adenoma,low\n"
                "case_hp,Hyperplastic polyps,low\n"
                "case_ta,Tubular adenoma,high\n",
                encoding="utf-8",
            )
            adapter = AdenomaManifestAdapter(
                str(manifest),
                str(labels),
                serrated_labels=(
                    "Hyperplastic polyps",
                    "Sessile serrated adenoma",
                    "Traditional serrated adenoma",
                    "Unclassified serrated adenoma",
                ),
                ssl_like_positive_labels=("Sessile serrated adenoma",),
                dysplasia_positive_grades=("high",),
            )
            cases = adapter.list_cases()
            self.assertEqual(len(cases), 3)
            self.assertEqual(cases[0].serrated_target, 1)
            self.assertEqual(cases[0].ssl_like_target, 1)
            self.assertEqual(cases[0].dysplasia_proxy_target, 0)
            self.assertEqual(cases[1].serrated_target, 1)
            self.assertEqual(cases[1].ssl_like_target, 0)
            self.assertEqual(cases[2].serrated_target, 0)
            self.assertEqual(cases[2].dysplasia_proxy_target, 1)
            output_path = Path(tmpdir) / "pilot.json"
            payload = adapter.build_pilot_subset(output_path, positives=2, negatives=1)
            self.assertEqual(payload["selected_case_count"], 3)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
