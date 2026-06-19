"""
Main orchestration pipeline for Multi-Modal Evidence Review.

Usage:
    python main.py                         # Process claims.csv -> output.csv
    python main.py --mode sample           # Process sample_claims.csv -> sample_output.csv
    python main.py --strategy b            # Use two-pass strategy
    python main.py --input custom.csv      # Custom input file
"""
import csv
import sys
import time
import argparse
from pathlib import Path

from config import (
    SAMPLE_CLAIMS_CSV, CLAIMS_CSV, OUTPUT_CSV, DATASET_DIR, OUTPUT_COLUMNS,
)
from data_loader import load_claims, load_user_history, load_evidence_requirements
from image_analyzer import (
    analyze_claim_strategy_a, analyze_claim_strategy_b,
    get_token_usage, reset_token_usage,
)
from postprocessor import postprocess_result, format_output_row


def process_claims(
    input_csv: Path,
    output_csv: Path,
    strategy: str = "a",
) -> list[dict]:
    """Process all claims from input CSV and write results to output CSV."""
    
    print(f"\n{'='*60}")
    print(f"Multi-Modal Evidence Review Agent")
    print(f"{'='*60}")
    print(f"Input:    {input_csv}")
    print(f"Output:   {output_csv}")
    print(f"Strategy: {'A (Single-pass)' if strategy == 'a' else 'B (Two-pass)'}")
    print(f"{'='*60}\n")
    
    # Load reference data
    print("[*] Loading reference data...")
    user_history = load_user_history()
    evidence_requirements = load_evidence_requirements()
    claims = load_claims(input_csv)
    print(f"   + {len(claims)} claims loaded")
    print(f"   + {len(user_history)} user history records")
    print(f"   + {len(evidence_requirements)} evidence requirements\n")
    
    # Reset token tracking
    reset_token_usage()
    
    # Process each claim
    results = []
    start_time = time.time()
    
    for i, claim in enumerate(claims, 1):
        user_id = claim["user_id"]
        claim_obj = claim["claim_object"]
        print(f"[{i}/{len(claims)}] Processing {user_id} ({claim_obj})...", end=" ", flush=True)
        
        # Lookup user history
        history = user_history.get(user_id)
        
        # Run analysis
        claim_start = time.time()
        if strategy == "a":
            raw_result = analyze_claim_strategy_a(claim, history, evidence_requirements)
        else:
            raw_result = analyze_claim_strategy_b(claim, history, evidence_requirements)
        
        # Post-process
        final_result = postprocess_result(raw_result, claim, history)
        output_row = format_output_row(final_result)
        results.append(output_row)
        
        elapsed = time.time() - claim_start
        status = output_row["claim_status"]
        print(f"-> {status} ({elapsed:.1f}s)")
        
        # Small delay between calls to respects free-tier 15 RPM
        if i < len(claims):
            time.sleep(5.0)
    
    total_time = time.time() - start_time
    
    # Write output CSV
    print(f"\n[*] Writing results to {output_csv}...")
    write_output_csv(results, output_csv)
    
    # Print summary
    usage = get_token_usage()
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Claims processed:  {len(results)}")
    print(f"Total time:        {total_time:.1f}s ({total_time/len(results):.1f}s/claim)")
    print(f"API calls:         {usage['total_calls']}")
    print(f"Images processed:  {usage['total_images_processed']}")
    print(f"Input tokens:      {usage['total_input_tokens']:,}")
    print(f"Output tokens:     {usage['total_output_tokens']:,}")
    
    # Cost estimate (Groq free tier - no cost)
    input_cost = (usage['total_input_tokens'] / 1_000_000) * 0.05
    output_cost = (usage['total_output_tokens'] / 1_000_000) * 0.08
    print(f"Estimated cost:    ${input_cost + output_cost:.4f} (Groq)")
    print(f"{'='*60}\n")
    
    return results


def write_output_csv(results: list[dict], output_path: Path):
    """Write results to a CSV file with the exact required schema."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"   + {len(results)} rows written")


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review Agent")
    parser.add_argument("--mode", choices=["sample", "test"], default="test",
                        help="Process sample_claims.csv (sample) or claims.csv (test)")
    parser.add_argument("--strategy", choices=["a", "b"], default="a",
                        help="Strategy A (single-pass) or B (two-pass)")
    parser.add_argument("--input", type=str, default=None,
                        help="Custom input CSV path")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output CSV path")
    
    args = parser.parse_args()
    
    # Determine input/output paths
    if args.input:
        input_csv = Path(args.input)
    elif args.mode == "sample":
        input_csv = SAMPLE_CLAIMS_CSV
    else:
        input_csv = CLAIMS_CSV
    
    if args.output:
        output_csv = Path(args.output)
    elif args.mode == "sample":
        output_csv = DATASET_DIR / "sample_output.csv"
    else:
        output_csv = OUTPUT_CSV
    
    process_claims(input_csv, output_csv, args.strategy)


if __name__ == "__main__":
    main()
