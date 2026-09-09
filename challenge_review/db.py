from contextlib import contextmanager
from datetime import datetime, timezone
import sqlite3

from .config import DATABASE_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reviewer' CHECK(role IN ('reviewer', 'admin')),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slides (
    slide_id TEXT PRIMARY KEY,
    original_label TEXT NOT NULL,
    pathology_type TEXT,
    grade TEXT,
    source TEXT,
    cv_fold INTEGER,
    wsi_path TEXT,
    wsi_format TEXT,
    wsi_available INTEGER NOT NULL DEFAULT 0,
    hardness_rank INTEGER,
    wrong_configurations INTEGER NOT NULL,
    total_configurations INTEGER NOT NULL DEFAULT 14,
    wrong_pct REAL NOT NULL,
    consensus_wrong_class TEXT,
    consensus_wrong_count INTEGER NOT NULL,
    mean_wrong_confidence REAL,
    max_wrong_confidence REAL,
    all_prediction_summary TEXT,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    feature TEXT NOT NULL,
    configuration TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    second_predicted_label TEXT,
    second_confidence REAL,
    is_correct INTEGER NOT NULL,
    UNIQUE(slide_id, model, feature)
);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    revised_label TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed', 'questionable')),
    diagnostic_confidence TEXT NOT NULL CHECK(diagnostic_confidence IN ('low', 'medium', 'high')),
    quality_flags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(slide_id, user_id)
);

CREATE TABLE IF NOT EXISTS review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES reviews(id) ON DELETE CASCADE,
    slide_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    revised_label TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostic_confidence TEXT NOT NULL,
    quality_flags TEXT NOT NULL,
    notes TEXT NOT NULL,
    changed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slides_default_sort
ON slides(consensus_wrong_count DESC, wrong_configurations DESC, mean_wrong_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_slides_labels ON slides(original_label, consensus_wrong_class);
CREATE INDEX IF NOT EXISTS idx_predictions_slide ON predictions(slide_id);
CREATE INDEX IF NOT EXISTS idx_reviews_user_status ON reviews(user_id, status);

CREATE TABLE IF NOT EXISTS challenge_case_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lesion_diagnosis TEXT NOT NULL,
    hgd_status TEXT NOT NULL,
    label_action TEXT NOT NULL,
    challenge_disposition TEXT NOT NULL DEFAULT 'retain_challenge'
        CHECK(challenge_disposition IN ('retain_challenge','pending_label_adjudication','exclude_label_error')),
    primary_challenge TEXT NOT NULL DEFAULT '',
    primary_challenge_other TEXT NOT NULL DEFAULT '',
    difficulty_note TEXT NOT NULL DEFAULT '',
    expert_confidence INTEGER NOT NULL DEFAULT 3 CHECK(expert_confidence BETWEEN 1 AND 5),
    no_localizable_evidence INTEGER NOT NULL DEFAULT 0,
    no_roi_reason TEXT NOT NULL DEFAULT '',
    no_roi_reason_other TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft', 'submitted')),
    version INTEGER NOT NULL DEFAULT 1,
    locked_by INTEGER REFERENCES users(id),
    locked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    submitted_at TEXT,
    UNIQUE(slide_id, user_id)
);

CREATE TABLE IF NOT EXISTS challenge_review_modifiers (
    review_id INTEGER NOT NULL REFERENCES challenge_case_reviews(id) ON DELETE CASCADE,
    modifier_code TEXT NOT NULL,
    PRIMARY KEY(review_id, modifier_code)
);

CREATE TABLE IF NOT EXISTS challenge_rois (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES challenge_case_reviews(id) ON DELETE CASCADE,
    slide_id TEXT NOT NULL REFERENCES slides(slide_id) ON DELETE CASCADE,
    display_order INTEGER NOT NULL,
    x_level0 REAL NOT NULL,
    y_level0 REAL NOT NULL,
    width_level0 REAL NOT NULL CHECK(width_level0 > 0),
    height_level0 REAL NOT NULL CHECK(height_level0 > 0),
    mpp_x REAL,
    mpp_y REAL,
    physical_width_um REAL,
    physical_height_um REAL,
    viewer_zoom REAL,
    diagnostic_role TEXT NOT NULL DEFAULT '',
    differential_direction TEXT NOT NULL DEFAULT '',
    evidence_strength TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(review_id, display_order)
);

CREATE TABLE IF NOT EXISTS challenge_roi_evidence (
    roi_id INTEGER NOT NULL REFERENCES challenge_rois(id) ON DELETE CASCADE,
    evidence_code TEXT NOT NULL,
    evidence_other TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(roi_id, evidence_code, evidence_other)
);

CREATE TABLE IF NOT EXISTS challenge_review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES challenge_case_reviews(id) ON DELETE CASCADE,
    slide_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    changed_by INTEGER NOT NULL REFERENCES users(id),
    changed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_challenge_review_user_status
ON challenge_case_reviews(user_id, status, slide_id);
CREATE INDEX IF NOT EXISTS idx_challenge_review_slide
ON challenge_case_reviews(slide_id, status);
CREATE INDEX IF NOT EXISTS idx_challenge_roi_review
ON challenge_rois(review_id, display_order);

CREATE VIEW IF NOT EXISTS challenge_slide_resolution AS
SELECT s.slide_id,
       SUM(CASE WHEN r.status='submitted' THEN 1 ELSE 0 END) AS submitted_reviews,
       SUM(CASE WHEN r.status='submitted' AND r.challenge_disposition='retain_challenge' THEN 1 ELSE 0 END) AS retain_votes,
       SUM(CASE WHEN r.status='submitted'
                     AND r.challenge_disposition IN ('pending_label_adjudication','exclude_label_error')
                     AND (CASE
                           WHEN r.hgd_status='Present' AND r.lesion_diagnosis='SSL' THEN 'SSLD'
                           WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TSA' THEN 'TSAD'
                           WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TA' THEN 'TAD'
                           WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TVA' THEN 'TVAD'
                           WHEN r.hgd_status='Absent' AND r.lesion_diagnosis IN ('HP','SSL','TSA','TA','TVA','IP','USA') THEN r.lesion_diagnosis
                         END)=s.consensus_wrong_class
                     AND s.consensus_wrong_class<>s.original_label THEN 1 ELSE 0 END) AS consensus_label_error_votes,
       CASE
         WHEN SUM(CASE WHEN r.status='submitted'
                            AND r.challenge_disposition IN ('pending_label_adjudication','exclude_label_error')
                            AND (CASE
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='SSL' THEN 'SSLD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TSA' THEN 'TSAD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TA' THEN 'TAD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TVA' THEN 'TVAD'
                                  WHEN r.hgd_status='Absent' AND r.lesion_diagnosis IN ('HP','SSL','TSA','TA','TVA','IP','USA') THEN r.lesion_diagnosis
                                END)=s.consensus_wrong_class
                            AND s.consensus_wrong_class<>s.original_label THEN 1 ELSE 0 END)>=2
           THEN 'eligible_for_exclusion'
         WHEN SUM(CASE WHEN r.status='submitted'
                            AND r.challenge_disposition IN ('pending_label_adjudication','exclude_label_error')
                            AND (CASE
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='SSL' THEN 'SSLD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TSA' THEN 'TSAD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TA' THEN 'TAD'
                                  WHEN r.hgd_status='Present' AND r.lesion_diagnosis='TVA' THEN 'TVAD'
                                  WHEN r.hgd_status='Absent' AND r.lesion_diagnosis IN ('HP','SSL','TSA','TA','TVA','IP','USA') THEN r.lesion_diagnosis
                                END)=s.consensus_wrong_class
                            AND s.consensus_wrong_class<>s.original_label THEN 1 ELSE 0 END)=1
           THEN 'pending_adjudication'
         WHEN SUM(CASE WHEN r.status='submitted' AND r.challenge_disposition='retain_challenge' THEN 1 ELSE 0 END)>0
           THEN 'retained_challenge'
         WHEN SUM(CASE WHEN r.status='submitted' THEN 1 ELSE 0 END)=0 THEN 'unreviewed'
         ELSE 'unresolved'
       END AS resolution_status
FROM slides s LEFT JOIN challenge_case_reviews r ON r.slide_id=s.slide_id
WHERE s.wrong_configurations=14
GROUP BY s.slide_id;
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


@contextmanager
def transaction():
    connection = connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with transaction() as connection:
        connection.executescript(SCHEMA)
        prediction_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        if "second_predicted_label" not in prediction_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN second_predicted_label TEXT")
        if "second_confidence" not in prediction_columns:
            connection.execute("ALTER TABLE predictions ADD COLUMN second_confidence REAL")
        challenge_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(challenge_case_reviews)")
        }
        if "challenge_disposition" not in challenge_columns:
            connection.execute(
                "ALTER TABLE challenge_case_reviews ADD COLUMN challenge_disposition TEXT NOT NULL DEFAULT 'retain_challenge'"
            )
            connection.execute("DROP VIEW IF EXISTS challenge_slide_resolution")
            connection.executescript(SCHEMA)
        from .challenge_reviews import migrate_legacy_reviews

        migrate_legacy_reviews(connection)
