from adenoma_agent.schemas import CaseSpec
from adenoma_agent.utils import (
    binary_label_from_grade,
    membership_label_from_type,
    read_csv_rows,
    write_json,
)


class AdenomaManifestAdapter(object):
    def __init__(
        self,
        manifest_csv,
        labels_csv,
        serrated_labels=None,
        ssl_like_positive_labels=None,
        dysplasia_positive_grades=None,
    ):
        self.manifest_csv = manifest_csv
        self.labels_csv = labels_csv
        self.serrated_labels = tuple(serrated_labels or ())
        self.ssl_like_positive_labels = tuple(ssl_like_positive_labels or ("Sessile serrated adenoma",))
        self.dysplasia_positive_grades = tuple(dysplasia_positive_grades or ("high",))
        self._cases = None

    def _load(self):
        manifest_rows = read_csv_rows(self.manifest_csv)
        label_rows = read_csv_rows(self.labels_csv)
        labels = {}
        for row in label_rows:
            labels[row["slide_id"]] = row
        cases = []
        for row in manifest_rows:
            slide_id = row["slide_id"]
            label_row = labels.get(slide_id, {})
            label = label_row.get("type")
            question = (
                "Review this whole-slide image through a layered serrated workflow. "
                "First decide whether this is a serrated lesion, then assess whether the crypt architecture supports an SSL-like pattern, "
                "and finally inspect high-magnification cytology for dysplasia or atypia."
            )
            cases.append(
                CaseSpec(
                    case_id=slide_id,
                    slide_path=row["slide_path"],
                    task_type="serrated_ssl_dysplasia_huge_region_agent",
                    question=question,
                    label=label,
                    serrated_target=membership_label_from_type(label, self.serrated_labels),
                    ssl_like_target=membership_label_from_type(label, self.ssl_like_positive_labels),
                    dysplasia_proxy_target=binary_label_from_grade(label_row.get("grade"), self.dysplasia_positive_grades),
                    metadata={
                        "slide_filename": row.get("slide_filename"),
                        "grade": label_row.get("grade"),
                        "proxy_task": "serrated_ssl_dysplasia_hierarchy",
                    },
                )
            )
        self._cases = cases

    def list_cases(self):
        if self._cases is None:
            self._load()
        return list(self._cases)

    def get_case(self, case_id):
        for case in self.list_cases():
            if case.case_id == case_id:
                return case
        raise KeyError("Unknown case_id: {0}".format(case_id))

    def build_pilot_subset(self, output_path, positives=12, negatives=12):
        positive_cases = [case for case in self.list_cases() if case.serrated_target == 1]
        negative_cases = [case for case in self.list_cases() if case.serrated_target == 0]
        selected = positive_cases[: int(positives)] + negative_cases[: int(negatives)]
        payload = {
            "pilot_target": "serrated_lesion_vs_non_serrated",
            "positives_requested": int(positives),
            "negatives_requested": int(negatives),
            "selected_case_count": len(selected),
            "cases": [
                {
                    "case_id": case.case_id,
                    "slide_path": case.slide_path,
                    "label": case.label,
                    "serrated_target": case.serrated_target,
                    "ssl_like_target": case.ssl_like_target,
                    "dysplasia_proxy_target": case.dysplasia_proxy_target,
                    "pilot_split": "pilot_eval",
                    "selection_reason": "aligned_manifest_and_labels_csv",
                }
                for case in selected
            ],
        }
        write_json(output_path, payload)
        return payload
