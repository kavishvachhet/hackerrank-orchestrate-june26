"""
Configuration constants for the Multi-Modal Evidence Review system.
All allowed values, file paths, and API settings.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"
IMAGES_DIR = DATASET_DIR  # image_paths in CSV are relative to dataset/

# ── API Settings ───────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
VISION_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.1  # Low temperature for deterministic output

# Rate limiting
MAX_CONCURRENT_CALLS = 3
RETRY_MAX_ATTEMPTS = 5
RETRY_BASE_DELAY = 10.0  # seconds

# ── Allowed Values ─────────────────────────────────────────────────────
CLAIM_STATUS_VALUES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE_VALUES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
]

# Per-object filtered issue_type lists. Keeps the model from picking
# implausible types for the wrong object (e.g. torn_packaging for a car).
ISSUE_TYPES_BY_OBJECT = {
    "car":     ["dent", "scratch", "crack", "glass_shatter", "broken_part",
                "missing_part", "stain", "none", "unknown"],
    "laptop":  ["dent", "scratch", "crack", "glass_shatter", "broken_part",
                "missing_part", "water_damage", "stain", "none", "unknown"],
    "package": ["torn_packaging", "crushed_packaging", "water_damage",
                "stain", "missing_part", "broken_part", "none", "unknown"],
}

CAR_OBJECT_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown"
]

LAPTOP_OBJECT_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown"
]

PACKAGE_OBJECT_PARTS = [
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown"
]

OBJECT_PARTS = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

SEVERITY_VALUES = ["none", "low", "medium", "high", "unknown"]

RISK_FLAG_VALUES = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
]

# ── Output Schema ──────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]
