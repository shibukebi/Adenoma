from adenoma_agent.schemas import CaseSpec
from adenoma_agent.utils import binary_label_from_type, read_csv_rows, write_json


class AdenomaManifestAdapter(object):
    def __init__(self, manifest_csv, labels_csv, positive_label="Sessile serrated adenoma"):
        self.manifest_csv = manifest_csv
        self.labels_csv = labels_csv
        self.positive_label = positive_label
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
                "Review this whole-slide image for SSA versus others. "
                "Focus on serration to crypt base, mucus cap, abnormal maturation, basal dilatation, "
                "crypt branching, horizontal growth, and boot/L/T-shaped crypts."
            )
            cases.append(
                CaseSpec(
                    case_id=slide_id,
                    slide_path=row["slide_path"],
                    task_type="ssa_vs_others_huge_region_agent",
                    question=question,
                    label=label,
                    binary_target=binary_label_from_type(label, self.positive_label),
                    metadata={
                        "slide_filename": row.get("slide_filename"),
                        "grade": label_row.get("grade"),
                        "proxy_task": "ssa_vs_others",
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
        positive_cases = [case for case in self.list_cases() if case.binary_target == 1]
        negative_cases = [case for case in self.list_cases() if case.binary_target == 0]
        selected = positive_cases[: int(positives)] + negative_cases[: int(negatives)]
        payload = {
            "positive_label": self.positive_label,
            "positives_requested": int(positives),
            "negatives_requested": int(negatives),
            "selected_case_count": len(selected),
            "cases": [
                {
                    "case_id": case.case_id,
                    "slide_path": case.slide_path,
                    "label": case.label,
                    "binary_target": case.binary_target,
                    "pilot_split": "pilot_eval",
                    "selection_reason": "aligned_manifest_and_labels_csv",
                }
                for case in selected
            ],
        }
        write_json(output_path, payload)
        return payload
