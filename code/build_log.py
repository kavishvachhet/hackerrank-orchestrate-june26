"""
Build the chat transcript log file for HackerRank submission.
Reads last_chat.md, redacts secrets, and writes to the required path.
"""
import re
import os
from pathlib import Path

REPO_DIR = Path(r"c:\Users\KAVISH\OneDrive\Documents\Downloads\Desktop\Hackerrank\hackerrank-orchestrate-june26")
LAST_CHAT = REPO_DIR / "last_chat.md"
LOG_DIR = Path(os.environ["USERPROFILE"]) / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"

# Patterns to redact
SECRET_PATTERNS = [
    # Groq keys
    (r'gsk_[A-Za-z0-9_-]{20,}', '[REDACTED_GROQ_API_KEY]'),
    # OpenRouter keys
    (r'sk-or-v1-[a-f0-9]{20,}', '[REDACTED_OPENROUTER_API_KEY]'),
    # Gemini/Google keys (AQ. prefix)
    (r'AQ\.[A-Za-z0-9_-]{20,}', '[REDACTED_GEMINI_API_KEY]'),
    # Anthropic keys
    (r'sk-ant-[A-Za-z0-9_-]{20,}', '[REDACTED_ANTHROPIC_API_KEY]'),
    # Generic sk- keys
    (r'sk-[A-Za-z0-9_-]{30,}', '[REDACTED_API_KEY]'),
    # Any env var assignments with keys
    (r'(\$env:(?:GROQ_API_KEY|OPENROUTER_API_KEY|GEMINI_API_KEY|ANTHROPIC_API_KEY)\s*=\s*")[^"]+(")', r'\1[REDACTED]\2'),
]

# Metadata patterns to strip (system metadata, not conversation content)
METADATA_STRIP_PATTERNS = [
    r'<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>',
    r'<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>',
]

def redact_secrets(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text

def strip_metadata(text: str) -> str:
    for pattern in METADATA_STRIP_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text

def clean_transcript(text: str) -> str:
    # Strip XML-like tags
    text = re.sub(r'</?USER_REQUEST>', '', text)
    text = re.sub(r'</?ADDITIONAL_METADATA>', '', text)
    text = re.sub(r'</?USER_SETTINGS_CHANGE>', '', text)
    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {LAST_CHAT}")
    content = LAST_CHAT.read_text(encoding="utf-8")

    # Process
    content = redact_secrets(content)
    content = strip_metadata(content)
    content = clean_transcript(content)

    # Build final log
    header = """================================================================================
HACKERRANK ORCHESTRATE — CHAT TRANSCRIPT
Multi-Modal Evidence Review
================================================================================

TOOL: Gemini CLI / Antigravity IDE (Google DeepMind)
DATE: June 19-20, 2026
PARTICIPANT: kavishvachhet

NOTE: All API keys and secrets have been redacted from this transcript.

================================================================================
SESSION 1: Initial Build + V1 (Groq) + V2 (OpenRouter) Pipeline
Tool: Claude Opus 4.6 / Gemini 3.5 Flash / Gemini 3.1 Pro (via Antigravity IDE)
Date: June 19, 2026
================================================================================

"""

    session2 = """

================================================================================
SESSION 2: V3 Anti-Hallucination Pipeline + Final Evaluation Report
Tool: Gemini 3.1 Pro / Claude Opus 4.6 (via Antigravity IDE)
Date: June 20, 2026
================================================================================

--- SESSION 2 SUMMARY ---

In this session, we:

1. ANALYZED the V2 pipeline's errors in detail against ground truth (76.53% composite on 20 sample claims)

2. IDENTIFIED 7 root causes of accuracy loss:
   - Severity over-estimation (5 claims)
   - crack vs glass_shatter confusion (2 claims)
   - stain vs water_damage confusion (1 claim)
   - Hallucinated damage — VLM imagines damage that isn't there (2 claims, most critical)
   - Missing risk flags (3 claims)
   - valid_image always says true (2 claims)
   - object_part too generic (2 claims)

3. BUILT V3 pipeline (code/openrouter_v3/) with targeted fixes:
   - Anti-hallucination anchor in system prompt
   - Issue type disambiguation (crack vs glass_shatter, stain vs water_damage)
   - Calibrated severity boundary examples
   - Mandatory 5-point risk checklist
   - Redefined valid_image criteria
   - Object part specificity guidance
   - 5 new postprocessor rules (zero regression risk)

4. CREATED comprehensive evaluation report (code/evaluation/evaluation_report.md) covering:
   - Metrics on sample_claims.csv for all 4 strategies
   - Strategy comparison (V1 Groq, V1 OpenRouter, V2 OpenRouter, V3 OpenRouter)
   - Final strategy selection rationale
   - Operational analysis (calls, tokens, images, cost, runtime, TPM/RPM)

5. RAN V3 on 44 test claims to generate final output3.csv

--- END SESSION 2 ---
"""

    final_content = header + content + session2

    LOG_FILE.write_text(final_content, encoding="utf-8")
    print(f"Written: {LOG_FILE}")
    print(f"Size: {LOG_FILE.stat().st_size:,} bytes")
    print(f"Lines: {len(final_content.splitlines()):,}")
    print("Done! Secrets redacted.")

if __name__ == "__main__":
    main()
