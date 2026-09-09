from pathlib import Path
import os


APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = APP_ROOT / "data"
STATIC_ROOT = APP_ROOT / "static"

DATABASE_PATH = Path(os.environ.get("CHALLENGE_REVIEW_DB", DATA_ROOT / "challenge_review.sqlite3"))
SESSION_HOURS = int(os.environ.get("CHALLENGE_REVIEW_SESSION_HOURS", "12"))
COOKIE_SECURE = os.environ.get("CHALLENGE_REVIEW_COOKIE_SECURE", "0") == "1"
WSI_CACHE_SIZE = int(os.environ.get("CHALLENGE_REVIEW_WSI_CACHE_SIZE", "6"))
TILE_JPEG_QUALITY = int(os.environ.get("CHALLENGE_REVIEW_TILE_QUALITY", "86"))
WSI_DISK_CACHE_ROOT = Path(
    os.environ.get("CHALLENGE_REVIEW_WSI_DISK_CACHE", APP_ROOT / "cache" / "wsi")
)
WSI_DISK_CACHE_MAX_BYTES = int(
    float(os.environ.get("CHALLENGE_REVIEW_WSI_CACHE_MAX_GB", "300")) * 1024**3
)
WSI_DISK_CACHE_TRIM_BYTES = int(
    float(os.environ.get("CHALLENGE_REVIEW_WSI_CACHE_TRIM_GB", "270")) * 1024**3
)
WSI_OVERVIEW_MAX_DIMENSION = int(
    os.environ.get("CHALLENGE_REVIEW_WSI_OVERVIEW_MAX_DIMENSION", "4096")
)
ISYNTAX_WORKER_TIMEOUT_SECONDS = int(
    os.environ.get("CHALLENGE_REVIEW_ISYNTAX_TIMEOUT", "180")
)

RESULT_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/5fold_11class")
OLD_RESULT_ROOT = Path("/data15/zhengke_usb2/yuexin_data/result/fold5_hp+yx")
SUMMARY_ROOT = RESULT_ROOT / "summary_11class_5fold"
CHALLENGE_CSV = SUMMARY_ROOT / "confusion_error_analysis/hard_slides_cross_experiment.csv"
METADATA_CSV = Path(
    "/data15/zhengke_usb2/yuexin_data/splits/adenoma_uni_hp_yx_ssl_5fold/"
    "joint_hp_yx_ssl_ready.csv"
)

CLASS_ORDER = ["IP", "HP", "SSL", "SSLD", "TSA", "TSAD", "USA", "TA", "TAD", "TVA", "TVAD"]
CLASS_BY_ID = {
    0: "SSL",
    1: "HP",
    2: "TSA",
    3: "USA",
    4: "TA",
    5: "TVA",
    6: "IP",
    7: "SSLD",
    8: "TSAD",
    9: "TAD",
    10: "TVAD",
}

HP_WSI_ROOTS = [
    Path("/data15/zhengke_usb2/yuexin_data/Adenoma_hp"),
    Path("/data15/zhengke_usb/Adenoma_hp"),
]
YX_WSI_ROOTS = [
    Path("/data15/zhengke_usb/Adenoma_yx"),
    Path("/data15/zhengke_usb2/yuexin_data/Adenoma_yx"),
]
