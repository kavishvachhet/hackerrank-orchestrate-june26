# Multi-Modal Evidence Review Agent

AI-powered system that verifies damage claims (car, laptop, package) by analyzing submitted images against claim conversations, user history, and evidence requirements.

## Architecture

```
5-Stage Pipeline (per claim):

  1. Data Loading     → Parse CSV, load images as base64
  2. Image Analysis   → Single VLM call (Claude Sonnet 4) with all images
  3. Post-processing  → Validate enums, normalize values
  4. Risk Flagging    → Merge VLM flags + user history + injection detection
  5. Output Assembly  → Apply consistency rules, write CSV row
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
# Windows PowerShell:
$env:ANTHROPIC_API_KEY = "your-key-here"

# Linux/macOS:
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage

```bash
# Process test claims (claims.csv → output.csv)
python main.py

# Process sample claims for evaluation
python main.py --mode sample

# Use Strategy B (two-pass)
python main.py --strategy b

# Custom input/output
python main.py --input path/to/input.csv --output path/to/output.csv
```

## Evaluation

```bash
# Run evaluation on sample data (Strategy A only)
python evaluation/main.py

# Compare Strategy A vs Strategy B
python evaluation/main.py --compare
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Main pipeline entry point |
| `config.py` | Constants, allowed values, paths |
| `data_loader.py` | CSV and image loading utilities |
| `prompts.py` | VLM prompt templates (2 strategies) |
| `image_analyzer.py` | Anthropic Claude API integration |
| `postprocessor.py` | Output validation and consistency rules |
| `evaluation/main.py` | Evaluation pipeline and report generation |

## Strategies

- **Strategy A (Single-pass)**: One comprehensive VLM call per claim with all images + context. Lower cost, faster.
- **Strategy B (Two-pass)**: First describes images without claim context, then matches descriptions against claim. More objective but 2x API calls.

## Key Design Decisions

1. **Single VLM call per claim** — batches all images to reduce API calls
2. **Structured JSON output** — deterministic parsing from the model
3. **Rule-based post-processing** — enforces consistency (e.g., insufficient evidence → not_enough_information)
4. **Prompt injection defense** — detects and flags manipulation attempts in images and conversations
5. **User history integration** — adds risk context without overriding visual evidence
