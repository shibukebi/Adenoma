#!/usr/bin/env python3
from pathlib import Path
import sys

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import generate_slideB_case_attention_figures as case_figures  # noqa: E402
from build_group_meeting_error_slides import load_all_predictions  # noqa: E402


RESULT_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
OUT_ROOT = case_figures.PROJECT_ROOT / "outputs" / "slideB_case_figures" / "major_confusions"
COMPONENT_ROOT = OUT_ROOT / "components"

CASES = [
    {
        "index": 101,
        "show_case_number": False,
        "zoom_upper_left": True,
        "slide_id": "f3c17793-02f0-4a93-aa8e-4cee12d76ae7",
        "pattern": "USA -> SSL",
        "true_class": "USA",
        "fold": 1,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-1/s_1_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb/Adenoma_hp/f3c17793-02f0-4a93-aa8e-4cee12d76ae7.isyntax"
        ),
        "review_focus": (
            "Overlapping serrated crypt architecture may make an unclassified serrated "
            "lesion resemble SSL."
        ),
    },
    {
        "index": 102,
        "show_case_number": False,
        "slide_id": "141540_673559001",
        "pattern": "USA -> HP",
        "true_class": "USA",
        "fold": 3,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-3/s_3_checkpoint.pt",
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/141540_673559001.svs"),
        "review_focus": (
            "Superficial hyperplastic morphology may dominate while diagnostically useful "
            "serrated architecture is focal."
        ),
    },
    {
        "index": 103,
        "show_case_number": False,
        "slide_id": "674102 1",
        "pattern": "TSA -> SSL",
        "true_class": "TSA",
        "fold": 1,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-1/s_1_checkpoint.pt",
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/674102 1.svs"),
        "review_focus": (
            "Shared serrated morphology and limited representation of ectopic crypt formation "
            "may favor SSL."
        ),
    },
    {
        "index": 104,
        "show_case_number": False,
        "slide_id": "140937_713536002",
        "pattern": "SSL -> HP",
        "true_class": "SSL",
        "fold": 1,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-1/s_1_checkpoint.pt",
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/140937_713536002.svs"),
        "review_focus": (
            "Surface maturation may resemble HP when basal crypt dilation and distortion are "
            "under-sampled or receive low attention."
        ),
    },
    {
        "index": 105,
        "show_case_number": False,
        "zoom_upper_left": True,
        "slide_id": "1eed0804-d0c9-436c-a416-0bfc0db0609b",
        "pattern": "TSAD -> TVA",
        "true_class": "TSAD",
        "fold": 1,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-1/s_1_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/"
            "1eed0804-d0c9-436c-a416-0bfc0db0609b.isyntax"
        ),
        "review_focus": (
            "The high-grade serrated component may be focal, while villous or conventional "
            "architecture dominates the slide-level representation."
        ),
    },
    {
        "index": 106,
        "show_case_number": False,
        "zoom_upper_left": True,
        "slide_id": "2929670e-28be-489e-825e-9f0e6e386598",
        "pattern": "SSLD -> TA",
        "true_class": "SSLD",
        "fold": 1,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-1/s_1_checkpoint.pt",
        "wsi_path": Path(
            "/data15/zhengke_usb2/yuexin_data/Adenoma_hp/"
            "2929670e-28be-489e-825e-9f0e6e386598.isyntax"
        ),
        "review_focus": (
            "Dysplastic serrated glands may resemble conventional tubular adenoma when basal "
            "serrated architecture is limited or receives low attention."
        ),
    },
    {
        "index": 107,
        "show_case_number": False,
        "slide_id": "640659 1",
        "pattern": "SSLD -> SSL",
        "true_class": "SSLD",
        "fold": 3,
        "checkpoint": RESULT_ROOT / "CLAM-SB/11class/10x/fold-3/s_3_checkpoint.pt",
        "wsi_path": Path("/data15/zhengke_usb/Adenoma_yx/640659 1.svs"),
        "review_focus": (
            "The dysplastic component may be focal or receive low attention, leaving the "
            "underlying SSL-like crypt architecture as the dominant signal."
        ),
    },
]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    COMPONENT_ROOT.mkdir(parents=True, exist_ok=True)
    case_figures.OUT_ROOT = OUT_ROOT
    case_figures.COMPONENT_ROOT = COMPONENT_ROOT

    predictions = load_all_predictions()
    rows = []
    for case in CASES:
        print(f"Processing {case['pattern']}: {case['slide_id']}", flush=True)
        rows.append(case_figures.process_case(case, predictions))

    index = pd.DataFrame(rows)
    index.to_csv(OUT_ROOT / "major_confusion_case_index.csv", index=False)
    print(index.to_string(index=False))


if __name__ == "__main__":
    main()
