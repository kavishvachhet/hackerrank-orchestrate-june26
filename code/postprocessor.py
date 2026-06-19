"""
Post-processing: validates and normalizes VLM output, applies consistency rules,
merges user history risk flags, and produces the final output row.
"""
import re

from config import (
    CLAIM_STATUS_VALUES, ISSUE_TYPE_VALUES, OBJECT_PARTS,
    SEVERITY_VALUES, RISK_FLAG_VALUES, OUTPUT_COLUMNS,
)
from data_loader import get_image_ids


def postprocess_result(
    raw_result: dict,
    claim: dict,
    user_history: dict | None,
) -> dict:
    """
    Validate, normalize, and apply consistency rules to the VLM output.
    Returns a final output row dict with all required columns.
    """
    claim_object = claim["claim_object"]
    image_ids = get_image_ids(claim["image_paths"])
    
    result = {}
    
    # ── Copy input fields ──────────────────────────────────────────────
    result["user_id"] = claim["user_id"]
    result["image_paths"] = claim["image_paths"]
    result["user_claim"] = claim["user_claim"]
    result["claim_object"] = claim_object
    
    # ── Normalize each output field ────────────────────────────────────
    result["evidence_standard_met"] = _normalize_bool(
        raw_result.get("evidence_standard_met", "false")
    )
    result["evidence_standard_met_reason"] = _clean_text(
        raw_result.get("evidence_standard_met_reason", "Unable to determine evidence sufficiency.")
    )
    
    result["issue_type"] = _normalize_enum(
        raw_result.get("issue_type", "unknown"), ISSUE_TYPE_VALUES, "unknown"
    )
    
    valid_parts = OBJECT_PARTS.get(claim_object, ["unknown"])
    result["object_part"] = _normalize_enum(
        raw_result.get("object_part", "unknown"), valid_parts, "unknown"
    )
    
    result["claim_status"] = _normalize_enum(
        raw_result.get("claim_status", "not_enough_information"),
        CLAIM_STATUS_VALUES, "not_enough_information"
    )
    result["claim_status_justification"] = _clean_text(
        raw_result.get("claim_status_justification", "Unable to process claim.")
    )
    
    result["supporting_image_ids"] = _normalize_image_ids(
        raw_result.get("supporting_image_ids", "none"), image_ids
    )
    
    result["valid_image"] = _normalize_bool(
        raw_result.get("valid_image", "true")
    )
    
    result["severity"] = _normalize_enum(
        raw_result.get("severity", "unknown"), SEVERITY_VALUES, "unknown"
    )
    
    # ── Risk flags: merge VLM flags + user history ─────────────────────
    result["risk_flags"] = _build_risk_flags(
        raw_result.get("risk_flags", "none"),
        user_history,
        claim,
        result,
    )
    
    # ── Consistency rules ──────────────────────────────────────────────
    result = _apply_consistency_rules(result)
    
    return result


def _normalize_bool(value) -> str:
    """Normalize to 'true' or 'false' string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    s = str(value).strip().lower()
    return "true" if s in ("true", "yes", "1") else "false"


def _normalize_enum(value: str, allowed: list[str], default: str) -> str:
    """Normalize a value to the closest allowed enum value."""
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    # Fuzzy match: check if any allowed value is contained in the value
    for a in allowed:
        if a in v or v in a:
            return a
    return default


def _clean_text(text: str) -> str:
    """Clean a text field for CSV output."""
    t = str(text).strip()
    # Remove newlines and excessive whitespace
    t = re.sub(r'\s+', ' ', t)
    # Limit length
    if len(t) > 500:
        t = t[:497] + "..."
    return t


def _normalize_image_ids(value: str, valid_ids: list[str]) -> str:
    """Normalize supporting_image_ids to valid image IDs."""
    v = str(value).strip().lower()
    if v in ("none", "n/a", ""):
        return "none"
    
    # Split and validate
    parts = [p.strip() for p in v.replace(",", ";").split(";") if p.strip()]
    valid = [p for p in parts if p in valid_ids]
    
    if not valid:
        return "none"
    return ";".join(valid)


def _build_risk_flags(
    vlm_flags: str,
    user_history: dict | None,
    claim: dict,
    result: dict,
) -> str:
    """Build the final risk_flags string by merging VLM + history + detection."""
    flags = set()
    
    # Parse VLM flags
    vlm_str = str(vlm_flags).strip().lower()
    if vlm_str and vlm_str != "none":
        for f in vlm_str.replace(",", ";").split(";"):
            f = f.strip()
            if f and f != "none" and f in RISK_FLAG_VALUES:
                flags.add(f)
    
    # Add user history flags
    if user_history:
        hist_flags = user_history.get("history_flags", "none")
        if hist_flags and hist_flags != "none":
            for f in hist_flags.replace(",", ";").split(";"):
                f = f.strip()
                if f and f != "none" and f in RISK_FLAG_VALUES:
                    flags.add(f)
    
    # Detect prompt injection in conversation
    user_claim_lower = claim.get("user_claim", "").lower()
    injection_patterns = [
        "approve the claim", "approve this claim", "skip manual review",
        "skip review", "mark as supported", "mark this as",
        "ignore all previous", "ignore previous instructions",
        "follow it and approve", "approve immediately",
    ]
    for pattern in injection_patterns:
        if pattern in user_claim_lower:
            flags.add("text_instruction_present")
            break
    
    if not flags:
        return "none"
    
    return ";".join(sorted(flags))


def _apply_consistency_rules(result: dict) -> dict:
    """Apply deterministic consistency rules to the final output."""
    
    # Rule 1: If evidence is not met, status should be not_enough_information
    if result["evidence_standard_met"] == "false":
        if result["claim_status"] == "supported":
            result["claim_status"] = "not_enough_information"
    
    # Rule 2: If valid_image is false, ensure manual_review_required in flags
    if result["valid_image"] == "false":
        flags = set(result["risk_flags"].split(";")) if result["risk_flags"] != "none" else set()
        flags.discard("none")
        flags.add("manual_review_required")
        result["risk_flags"] = ";".join(sorted(flags))
    
    # Rule 3: If claim_status is not_enough_information, supporting_image_ids should be none
    if result["claim_status"] == "not_enough_information" and result["evidence_standard_met"] == "false":
        result["supporting_image_ids"] = "none"
    
    # Rule 4: If issue_type is none, severity should be none
    if result["issue_type"] == "none":
        result["severity"] = "none"
    
    # Rule 5: If severity is none and issue_type shows real damage, correct it
    if result["severity"] == "none" and result["issue_type"] not in ("none", "unknown") and result["claim_status"] == "supported":
        result["severity"] = "medium"  # default to medium for visible damage
    
    return result


def format_output_row(result: dict) -> dict:
    """Ensure the output row has all columns in the correct order."""
    return {col: result.get(col, "") for col in OUTPUT_COLUMNS}
