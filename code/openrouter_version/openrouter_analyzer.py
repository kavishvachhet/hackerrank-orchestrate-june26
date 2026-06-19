"""
Image analysis module — VLM calls via OpenRouter for damage claim review.
OpenRouter uses OpenAI-compatible API format.
Implements Strategy A (single-pass).
"""
import json
import time
import re
import sys
import os
from pathlib import Path

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from config import (
    RETRY_MAX_ATTEMPTS, RETRY_BASE_DELAY,
    DATASET_DIR,
)
from prompts import (
    get_system_prompt,
    get_analysis_prompt_strategy_a,
)
from data_loader import load_images_for_claim, get_image_ids, get_relevant_requirements

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# Using Google's Gemini 2.5 Flash via OpenRouter
VISION_MODEL_NAME = "google/gemini-2.5-flash"
TEMPERATURE = 0.0

# Track token usage
token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "total_images_processed": 0,
}

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            timeout=90.0,
        )
    return _client

def _call_vlm_with_images(
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, str]],
) -> dict:
    client = _get_client()

    content = []
    for img_id, img_data in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_data}"
            }
        })
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=4096,
            )

            if response.usage:
                token_usage["total_input_tokens"] += response.usage.prompt_tokens or 0
                token_usage["total_output_tokens"] += response.usage.completion_tokens or 0
            token_usage["total_calls"] += 1
            token_usage["total_images_processed"] += len(images)

            text = response.choices[0].message.content.strip()
            return _parse_json_response(text)

        except Exception as e:
            err_str = str(e).lower()
            wait = RETRY_BASE_DELAY * (2 ** attempt)

            if "429" in err_str or "rate" in err_str or "resource" in err_str:
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

    if isinstance(result.get("evidence_standard_met"), str):
        result["evidence_standard_met"] = result["evidence_standard_met"].lower() == "true"
    if isinstance(result.get("valid_image"), str):
        result["valid_image"] = result["valid_image"].lower() == "true"
        
    return result

def get_token_usage():
    return token_usage
