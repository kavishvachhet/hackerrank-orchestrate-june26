"""
Image analysis module — VLM calls to Groq for damage claim review.
Implements both Strategy A (single-pass) and Strategy B (two-pass).
Uses OpenAI-compatible API via the Groq SDK.
"""
import json
import time
import re
from pathlib import Path

from groq import Groq

from config import (
    GROQ_API_KEY, VISION_MODEL_NAME, TEXT_MODEL_NAME, TEMPERATURE,
    RETRY_MAX_ATTEMPTS, RETRY_BASE_DELAY,
    DATASET_DIR,
)
from prompts import (
    get_system_prompt,
    get_analysis_prompt_strategy_a,
    get_analysis_prompt_strategy_b_pass1,
    get_analysis_prompt_strategy_b_pass2,
)
from data_loader import load_images_for_claim, get_image_ids, get_relevant_requirements

# Track token usage for operational analysis
token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_calls": 0,
    "total_images_processed": 0,
}

# Initialize client
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY, timeout=60.0)
    return _client


def _call_vlm_with_images(
    system_prompt: str,
    user_prompt: str,
    images: list[tuple[str, str]],  # list of (image_id, base64_data)
) -> dict:
    """Make an API call to Groq with images. Returns parsed JSON response."""
    client = _get_client()

    # Build message content: images as data URLs, then text
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

            # Track usage
            if response.usage:
                token_usage["total_input_tokens"] += response.usage.prompt_tokens or 0
                token_usage["total_output_tokens"] += response.usage.completion_tokens or 0
            token_usage["total_calls"] += 1
            token_usage["total_images_processed"] += len(images)

            # Parse response
            text = response.choices[0].message.content.strip()
            return _parse_json_response(text)

        except Exception as e:
            err_str = str(e).lower()
            # Parse retry-after delay from error if available
            wait = _extract_retry_delay(str(e))
            if wait is None:
                wait = RETRY_BASE_DELAY * (2 ** attempt)

            if "429" in err_str or "rate" in err_str or "resource" in err_str:
                wait = max(wait, RETRY_BASE_DELAY)  # At least base delay
                print(f"  [WAIT] Rate limited. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            elif attempt < RETRY_MAX_ATTEMPTS - 1:
                print(f"  [WARN] API error: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] API call failed after {RETRY_MAX_ATTEMPTS} attempts: {e}")
                return _empty_result()

    return _empty_result()


def _call_text_only(
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """Make a text-only API call to Groq. Returns parsed JSON response."""
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=TEXT_MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=4096,
            )

            if response.usage:
                token_usage["total_input_tokens"] += response.usage.prompt_tokens or 0
                token_usage["total_output_tokens"] += response.usage.completion_tokens or 0
            token_usage["total_calls"] += 1

            text = response.choices[0].message.content.strip()
            return _parse_json_response(text)

        except Exception as e:
            err_str = str(e).lower()
            wait = _extract_retry_delay(str(e))
            if wait is None:
                wait = RETRY_BASE_DELAY * (2 ** attempt)

            if "429" in err_str or "rate" in err_str or "resource" in err_str:
                wait = max(wait, RETRY_BASE_DELAY)
                print(f"  [WAIT] Rate limited. Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            elif attempt < RETRY_MAX_ATTEMPTS - 1:
                print(f"  [WARN] API error: {e}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] API call failed: {e}")
                return {}

    return {}


def _extract_retry_delay(error_msg: str) -> float | None:
    """Parse retry delay from API error message (e.g., 'Please retry in 20.5s')."""
    match = re.search(r'retry\s+(?:in|after)\s+([\d.]+)\s*s', error_msg, re.IGNORECASE)
    if match:
        delay = float(match.group(1))
        return delay + 2.0  # Add 2s buffer so we don't hit the limit again immediately
    # Also check for retry-after header style
    match = re.search(r'retry[_-]?after["\s:]+(\d+)', error_msg, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0
    return None


def _parse_json_response(text: str) -> dict:
    """Parse JSON from the model response, handling markdown code blocks."""
    # Remove markdown code blocks if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"  [WARN] Failed to parse JSON response: {text[:200]}...")
        return _empty_result()


def _empty_result() -> dict:
    """Return a default empty result when API call fails."""
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "API call failed; manual review required",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Unable to process this claim automatically due to API error.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def analyze_claim_strategy_a(
    claim: dict,
    user_history: dict | None,
    evidence_requirements: list[dict],
) -> dict:
    """Strategy A: Single comprehensive VLM call with all images + claim context."""

    claim_object = claim["claim_object"]
    user_claim = claim["user_claim"]
    image_paths_str = claim["image_paths"]

    # Load images
    images = load_images_for_claim(image_paths_str)
    image_ids = [img[0] for img in images]

    if not images:
        return _empty_result()

    # Get relevant evidence requirements
    req_descriptions = get_relevant_requirements(evidence_requirements, claim_object)

    # Build prompts
    system_prompt = get_system_prompt()
    user_prompt = get_analysis_prompt_strategy_a(
        claim_object=claim_object,
        user_claim=user_claim,
        image_ids=image_ids,
        user_history=user_history,
        evidence_requirements=req_descriptions,
    )

    # Call VLM
    result = _call_vlm_with_images(system_prompt, user_prompt, images)

    return result


def analyze_claim_strategy_b(
    claim: dict,
    user_history: dict | None,
    evidence_requirements: list[dict],
) -> dict:
    """Strategy B: Two-pass analysis (describe images, then match against claim)."""

    claim_object = claim["claim_object"]
    user_claim = claim["user_claim"]
    image_paths_str = claim["image_paths"]

    # Load images
    images = load_images_for_claim(image_paths_str)
    image_ids = [img[0] for img in images]

    if not images:
        return _empty_result()

    # Pass 1: Describe images (with images, no claim context)
    system_prompt = get_system_prompt()
    pass1_prompt = get_analysis_prompt_strategy_b_pass1(
        claim_object=claim_object,
        image_ids=image_ids,
    )

    image_descriptions = _call_vlm_with_images(system_prompt, pass1_prompt, images)

    if not image_descriptions:
        return _empty_result()

    # Pass 2: Match against claim (text-only, no images)
    req_descriptions = get_relevant_requirements(evidence_requirements, claim_object)

    pass2_prompt = get_analysis_prompt_strategy_b_pass2(
        claim_object=claim_object,
        user_claim=user_claim,
        image_descriptions=json.dumps(image_descriptions, indent=2),
        image_ids=image_ids,
        user_history=user_history,
        evidence_requirements=req_descriptions,
    )

    result = _call_text_only(system_prompt, pass2_prompt)

    return result


def get_token_usage() -> dict:
    """Return current token usage statistics."""
    return token_usage.copy()


def reset_token_usage():
    """Reset token usage counters."""
    global token_usage
    token_usage = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_calls": 0,
        "total_images_processed": 0,
    }
