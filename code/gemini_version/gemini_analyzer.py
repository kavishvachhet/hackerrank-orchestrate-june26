"""
Image analysis module — VLM calls to Google Gemini for damage claim review.
Implements Strategy A (single-pass).
"""
import json
import time
import re
import sys
import os
import requests
from pathlib import Path

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RETRY_MAX_ATTEMPTS, RETRY_BASE_DELAY,
    DATASET_DIR,
)
from prompts import (
    get_system_prompt,
    get_analysis_prompt_strategy_a,
)
from data_loader import load_images_for_claim, get_image_ids, get_relevant_requirements

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Try these model names in order until one works
MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-pro-vision",
    "gemini-pro",
]
VISION_MODEL_NAME = MODEL_CANDIDATES[0]  # Start with best option
_model_resolved = False  # Flag to avoid re-testing once we find a working model

# Track token usage
token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "total_images_processed": 0,
}

def _call_vlm_with_images(
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, str]],
) -> dict:
    global VISION_MODEL_NAME, _model_resolved
    
    parts = []
    
    # Gemini requires the text to come first or be mixed with images
    full_text = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\nUSER PROMPT:\n{user_prompt}"
    parts.append({"text": full_text})
    
    for img_id, img_data in images:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_data
            }
        })
        
    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.0
        }
    }

    # If model not yet resolved, try all candidates
    if not _model_resolved:
        for model_name in MODEL_CANDIDATES:
            test_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
            print(f"\n  [AUTO] Trying model: {model_name}...", end="", flush=True)
            try:
                resp = requests.post(test_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60.0)
                if resp.status_code == 404:
                    print(f" 404 (not available)")
                    continue
                elif resp.status_code == 429:
                    # Model exists but rate limited - that's fine, we found it!
                    print(f" Found! (rate limited, will retry)")
                    VISION_MODEL_NAME = model_name
                    _model_resolved = True
                    break
                elif resp.status_code == 200:
                    print(f" Success!")
                    VISION_MODEL_NAME = model_name
                    _model_resolved = True
                    # Parse and return this response directly
                    data = resp.json()
                    if "usageMetadata" in data:
                        token_usage["total_input_tokens"] += data["usageMetadata"].get("promptTokenCount", 0)
                        token_usage["total_output_tokens"] += data["usageMetadata"].get("candidatesTokenCount", 0)
                    token_usage["total_calls"] += 1
                    token_usage["total_images_processed"] += len(images)
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return _parse_json_response(text)
                    except (KeyError, IndexError) as e:
                        print(f"  [WARN] Unexpected response format: {e}")
                        return _empty_result()
                else:
                    print(f" Error {resp.status_code}: {resp.text[:200]}")
                    continue
            except Exception as e:
                print(f" Error: {e}")
                continue
        
        if not _model_resolved:
            print(f"\n  [FAIL] No working Gemini model found for this API key!")
            return _empty_result()

    # Normal path: model is resolved, make the call with retries
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{VISION_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            response = requests.post(
                url, 
                headers={'Content-Type': 'application/json'}, 
                json=payload,
                timeout=60.0
            )
            
            # Rate limit check
            if response.status_code == 429:
                wait = max(RETRY_BASE_DELAY * (2 ** attempt), 60)
                print(f"  [WAIT] Rate limited by Gemini. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            # Track usage
            if "usageMetadata" in data:
                token_usage["total_input_tokens"] += data["usageMetadata"].get("promptTokenCount", 0)
                token_usage["total_output_tokens"] += data["usageMetadata"].get("candidatesTokenCount", 0)
            token_usage["total_calls"] += 1
            token_usage["total_images_processed"] += len(images)

            # Parse response
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                return _parse_json_response(text)
            except (KeyError, IndexError) as e:
                print(f"  [WARN] Unexpected Gemini response format: {e}")
                return _empty_result()

        except Exception as e:
            err_str = str(e).lower()
            wait = RETRY_BASE_DELAY * (2 ** attempt)

            if "429" in err_str or "quota" in err_str:
                wait = max(wait, 60)
                print(f"  [WAIT] Rate limited. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            elif attempt < RETRY_MAX_ATTEMPTS - 1:
                print(f"  [WARN] API error: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] API call failed after {RETRY_MAX_ATTEMPTS} attempts: {e}")
                return _empty_result()

    return _empty_result()

def _parse_json_response(text: str) -> dict:
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parsing failed: {e}. Returning empty format.")
        return _empty_result()

def _empty_result() -> dict:
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "API call failed or parsing error",
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "API call failed",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown"
    }

def process_claim_strategy_a(
    claim_record: dict,
    user_history: dict | None,
    evidence_requirements: list[str],
) -> dict:
    start_time = time.time()
    user_id = claim_record["user_id"]
    claim_object = claim_record["claim_object"]
    user_claim = claim_record["user_claim"]
    image_paths_str = claim_record["image_paths"]

    image_ids = get_image_ids(image_paths_str)
    images_b64 = load_images_for_claim(image_paths_str)

    if not images_b64:
        return _empty_result()

    sys_prompt = get_system_prompt()
    user_prompt = get_analysis_prompt_strategy_a(
        claim_object=claim_object,
        user_claim=user_claim,
        image_ids=image_ids,
        user_history=user_history,
        evidence_requirements=evidence_requirements,
    )

    result = _call_vlm_with_images(sys_prompt, user_prompt, images_b64)

    # Convert evidence_standard_met and valid_image to boolean if they aren't
    if isinstance(result.get("evidence_standard_met"), str):
        result["evidence_standard_met"] = result["evidence_standard_met"].lower() == "true"
    if isinstance(result.get("valid_image"), str):
        result["valid_image"] = result["valid_image"].lower() == "true"
        
    return result

def get_token_usage():
    return token_usage
