"""
Data loading utilities for claims, user history, evidence requirements, and images.
"""
import csv
import base64
from pathlib import Path
from typing import Optional

from config import DATASET_DIR, USER_HISTORY_CSV, EVIDENCE_REQUIREMENTS_CSV


def load_claims(csv_path: Path) -> list[dict]:
    """Load claims from a CSV file. Works for both sample (with labels) and test (input-only)."""
    claims = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            claims.append(cleaned)
    return claims


def load_user_history(csv_path: Optional[Path] = None) -> dict:
    """Load user history keyed by user_id."""
    csv_path = csv_path or USER_HISTORY_CSV
    history = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            uid = cleaned["user_id"]
            history[uid] = cleaned
    return history


def load_evidence_requirements(csv_path: Optional[Path] = None) -> list[dict]:
    """Load evidence requirements as a list of rule dicts."""
    csv_path = csv_path or EVIDENCE_REQUIREMENTS_CSV
    reqs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            reqs.append(cleaned)
    return reqs


def get_relevant_requirements(requirements: list[dict], claim_object: str, issue_family: str = "") -> list[str]:
    """Get evidence requirement descriptions relevant to a claim object and optional issue family."""
    relevant = []
    for req in requirements:
        obj = req.get("claim_object", "")
        if obj == "all" or obj == claim_object:
            relevant.append(f"[{req.get('requirement_id', '')}] ({req.get('applies_to', '')}): {req.get('minimum_image_evidence', '')}")
    return relevant


import io
import pillow_avif
from PIL import Image

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load a single image, compress it if it's too large, and return (image_id, base64_encoded_data).
    
    image_path is relative to dataset/ directory, e.g. 'images/test/case_001/img_1.jpg'
    """
    full_path = DATASET_DIR / image_path
    image_id = Path(image_path).stem  # e.g., 'img_1'
    
    # Open image with Pillow and compress it
    with Image.open(full_path) as img:
        # Convert to RGB if needed (e.g. for PNGs with alpha)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        # Resize if dimensions are very large
        max_size = (1024, 1024)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save to bytes buffer as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
    
    return image_id, data


def load_images_for_claim(image_paths_str: str) -> list[tuple[str, str]]:
    """Load all images for a claim. Returns list of (image_id, base64_data) tuples."""
    paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    images = []
    for p in paths:
        try:
            img_id, data = load_image_as_base64(p)
            images.append((img_id, data))
        except FileNotFoundError:
            print(f"  [WARN] Image not found: {p}")
        except Exception as e:
            print(f"  [WARN] Error loading {p}: {e}")
    return images


def get_image_ids(image_paths_str: str) -> list[str]:
    """Extract image IDs from the image_paths string without loading the images."""
    paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    return [Path(p).stem for p in paths]
