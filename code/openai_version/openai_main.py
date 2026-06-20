"""
Main orchestration pipeline for OpenAI Multi-Modal Evidence Review.
Throttled to 1 request per minute.
"""
import csv
import sys
import time
import argparse
import os
from pathlib import Path

# Add parent directory to path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SAMPLE_CLAIMS_CSV, CLAIMS_CSV, OUTPUT_CSV, DATASET_DIR, OUTPUT_COLUMNS,
)
from data_loader import load_claims, load_user_history, load_evidence_requirements
from openai_version.openai_analyzer import (
    process_claim_strategy_a,
    get_token_usage,
)
from postprocessor import postprocess_result, format_output_row

def process_claims(
    input_csv: Path,
    output_csv: Path,
) -> list[dict]:
    
    print(f"\n{'='*60}")
    print(f"OpenAI Evidence Review Agent (1 Request / Min Limit)")
    print(f"{'='*60}")
    print(f"Input:    {input_csv}")
    print(f"Output:   {output_csv}")
    print(f"{'='*60}\n")
    
    # Load reference data
    print("[*] Loading reference data...")
    user_history = load_user_history()
    evidence_requirements = load_evidence_requirements()
    claims = load_claims(input_csv)
    print(f"   + {len(claims)} claims loaded")
    print(f"   + {len(user_history)} user history records")
    print(f"   + {len(evidence_requirements)} evidence requirements\n")

    # Resume capability for OpenAI output
    processed_claims = set()
    if output_csv.exists():
        try:
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "API call failed" not in row.get("evidence_standard_met_reason", ""):
                        processed_claims.add(row["user_claim"])
        except Exception as e:
            print(f"Could not read existing output file for resume: {e}")

    results = []
    
    if output_csv.exists() and processed_claims:
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["user_claim"] in processed_claims:
                    results.append(row)

    print(f"[*] Resuming: {len(processed_claims)} claims already processed successfully.")
    
    start_time = time.time()
    
    for i, claim in enumerate(claims, 1):
        if claim["user_claim"] in processed_claims:
            print(f"[{i}/{len(claims)}] Skipping {claim['user_id']} (already processed)")
            continue

        print(f"[{i}/{len(claims)}] Processing {claim['user_id']} ({claim['claim_object']})...", end="", flush=True)
        
        user_hist = user_history.get(claim["user_id"])
        claim_start = time.time()

        raw_result = process_claim_strategy_a(claim, user_hist, evidence_requirements)
        
        final_result = postprocess_result(claim, raw_result)
        formatted_row = format_output_row(claim, final_result)
        results.append(formatted_row)
        
        claim_time = time.time() - claim_start
        print(f" -> {final_result.get('claim_status', 'error')} ({claim_time:.1f}s)")
        
        # Write incremental results immediately
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(results)

        # Rate Limit Sleep: Enforce 60 seconds between calls to respect OpenAI free tier limits
        if i < len(claims):
            print(f"  [WAIT] Sleeping 60 seconds to respect 1 RPM limit...")
            time.sleep(60)

    total_time = time.time() - start_time
    usage = get_token_usage()
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Claims processed:  {len(results)}")
    print(f"Total time:        {total_time:.1f}s")
    print(f"API calls:         {usage['total_calls']}")
    print(f"Images processed:  {usage['total_images_processed']}")
    print(f"Input tokens:      {usage['total_input_tokens']:,}")
    print(f"Output tokens:     {usage['total_output_tokens']:,}")
    print(f"{'='*60}\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI Evidence Review")
    parser.add_argument("--mode", choices=["test", "sample"], default="test", help="Which dataset to process")
    args = parser.parse_args()

    if args.mode == "sample":
        input_csv = SAMPLE_CLAIMS_CSV
        output_csv = DATASET_DIR / "sample_openai_output.csv"
    else:
        input_csv = CLAIMS_CSV
        output_csv = DATASET_DIR / "openai_output.csv"

    process_claims(input_csv, output_csv)
