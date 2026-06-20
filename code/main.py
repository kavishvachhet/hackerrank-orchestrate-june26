"""
Main Entry Point for HackerRank Evidence Review.
Delegates to the OpenRouter V3 (Anti-Hallucination) pipeline.

Usage:
  python code/main.py --mode test
  python code/main.py --mode sample
"""
import sys
import os
import argparse
from pathlib import Path

# Ensure the code directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_CLAIMS_CSV, CLAIMS_CSV, DATASET_DIR
from openrouter_v3.main_v3 import process_claims

def main():
    parser = argparse.ArgumentParser(description="Run Evidence Review Pipeline (V3)")
    parser.add_argument("--mode", choices=["test", "sample"], default="test", help="Which dataset to process")
    args = parser.parse_args()

    # Determine input/output paths based on mode
    if args.mode == "sample":
        input_csv = SAMPLE_CLAIMS_CSV
        output_csv = DATASET_DIR / "sample_v3_output.csv"
    else:
        input_csv = CLAIMS_CSV
        output_csv = DATASET_DIR / "output.csv"  # Final output name

    # Execute the V3 pipeline
    process_claims(input_csv, output_csv)

if __name__ == "__main__":
    # Check for API Key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable is missing.", file=sys.stderr)
        print("Please set it before running. Example:", file=sys.stderr)
        print("  $env:OPENROUTER_API_KEY=\"your_key_here\"; python code/main.py --mode test", file=sys.stderr)
        sys.exit(1)

    main()
