"""
Evaluation module for comparing predictions against ground truth.
Computes per-field accuracy, F1 scores, and generates an evaluation report.
"""
import csv
import sys
import time
import json
from pathlib import Path
from collections import defaultdict

# Add parent to path so we can import code modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SAMPLE_CLAIMS_CSV, DATASET_DIR, OUTPUT_COLUMNS,
    CLAIM_STATUS_VALUES, ISSUE_TYPE_VALUES, SEVERITY_VALUES,
)
from data_loader import load_claims
from openrouter_v3.main_v3 import process_claims
from openrouter_v3.analyzer_v3 import get_token_usage, reset_token_usage


def evaluate_predictions(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Compare predictions against ground truth and compute metrics."""
    
    if len(predictions) != len(ground_truth):
        print(f"[WARN] Row count mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth")
    
    metrics = {}
    
    # Fields to evaluate with exact match accuracy
    exact_fields = [
        "claim_status", "issue_type", "object_part",
        "evidence_standard_met", "valid_image", "severity",
    ]
    
    for field in exact_fields:
        correct = 0
        total = 0
        confusion = defaultdict(lambda: defaultdict(int))
        errors = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            pred_val = str(pred.get(field, "")).strip().lower()
            gt_val = str(gt.get(field, "")).strip().lower()
            total += 1
            
            if pred_val == gt_val:
                correct += 1
            else:
                errors.append({
                    "row": i + 1,
                    "user_id": gt.get("user_id", ""),
                    "predicted": pred_val,
                    "expected": gt_val,
                })
            
            confusion[gt_val][pred_val] += 1
        
        metrics[field] = {
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "errors": errors,
            "confusion": dict(confusion),
        }
    
    # Set-based F1 for risk_flags and supporting_image_ids
    set_fields = ["risk_flags", "supporting_image_ids"]
    
    for field in set_fields:
        tp = fp = fn = 0
        errors = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            pred_set = _parse_set_field(pred.get(field, "none"))
            gt_set = _parse_set_field(gt.get(field, "none"))
            
            tp += len(pred_set & gt_set)
            fp += len(pred_set - gt_set)
            fn += len(gt_set - pred_set)
            
            if pred_set != gt_set:
                errors.append({
                    "row": i + 1,
                    "user_id": gt.get("user_id", ""),
                    "predicted": ";".join(sorted(pred_set)) or "none",
                    "expected": ";".join(sorted(gt_set)) or "none",
                })
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "errors": errors,
        }
    
    # Composite score (weighted)
    weights = {
        "claim_status": 0.30,
        "issue_type": 0.15,
        "object_part": 0.15,
        "evidence_standard_met": 0.10,
        "valid_image": 0.05,
        "severity": 0.10,
        "risk_flags": 0.10,
        "supporting_image_ids": 0.05,
    }
    
    composite = 0
    for field, weight in weights.items():
        if "accuracy" in metrics[field]:
            composite += weight * metrics[field]["accuracy"]
        elif "f1" in metrics[field]:
            composite += weight * metrics[field]["f1"]
    
    metrics["composite_score"] = composite
    
    return metrics


def _parse_set_field(value: str) -> set:
    """Parse a semicolon-separated field into a set."""
    v = str(value).strip().lower()
    if v in ("none", "n/a", ""):
        return {"none"}
    return set(p.strip() for p in v.split(";") if p.strip())


def generate_report(
    metrics: dict,
    usage: dict,
    strategy_name: str,
    runtime: float,
    num_claims: int,
    output_path: Path,
    strategy_b_metrics: dict = None,
    strategy_b_usage: dict = None,
    strategy_b_runtime: float = None,
):
    """Generate the evaluation_report.md file."""
    
    lines = []
    lines.append("# Evaluation Report -- Multi-Modal Evidence Review")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Claims evaluated**: {num_claims}")
    lines.append(f"- **Primary strategy**: Strategy A (Single-pass)")
    lines.append(f"- **Composite score**: {metrics['composite_score']:.2%}")
    lines.append("")
    
    # ── Per-field accuracy table ───────────────────────────────────────
    lines.append("## Per-Field Accuracy")
    lines.append("")
    lines.append("| Field | Metric | Score |")
    lines.append("|---|---|---|")
    
    exact_fields = ["claim_status", "issue_type", "object_part",
                    "evidence_standard_met", "valid_image", "severity"]
    for field in exact_fields:
        m = metrics[field]
        lines.append(f"| `{field}` | Accuracy | {m['accuracy']:.2%} ({m['correct']}/{m['total']}) |")
    
    set_fields = ["risk_flags", "supporting_image_ids"]
    for field in set_fields:
        m = metrics[field]
        lines.append(f"| `{field}` | F1 | {m['f1']:.2%} (P={m['precision']:.2%} R={m['recall']:.2%}) |")
    
    lines.append("")
    
    # ── Strategy comparison ────────────────────────────────────────────
    if strategy_b_metrics:
        lines.append("## Strategy Comparison")
        lines.append("")
        lines.append("| Metric | Strategy A (Single-pass) | Strategy B (Two-pass) |")
        lines.append("|---|---|---|")
        lines.append(f"| Composite Score | {metrics['composite_score']:.2%} | {strategy_b_metrics['composite_score']:.2%} |")
        lines.append(f"| Claim Status Accuracy | {metrics['claim_status']['accuracy']:.2%} | {strategy_b_metrics['claim_status']['accuracy']:.2%} |")
        lines.append(f"| Issue Type Accuracy | {metrics['issue_type']['accuracy']:.2%} | {strategy_b_metrics['issue_type']['accuracy']:.2%} |")
        lines.append(f"| API Calls | {usage['total_calls']} | {strategy_b_usage['total_calls']} |")
        lines.append(f"| Runtime | {runtime:.1f}s | {strategy_b_runtime:.1f}s |")
        
        a_cost = (usage['total_input_tokens'] / 1e6) * 0.05 + (usage['total_output_tokens'] / 1e6) * 0.08
        b_cost = (strategy_b_usage['total_input_tokens'] / 1e6) * 0.05 + (strategy_b_usage['total_output_tokens'] / 1e6) * 0.08
        lines.append(f"| Est. Cost | ${a_cost:.4f} | ${b_cost:.4f} |")
        lines.append("")
    
    # ── Error analysis ─────────────────────────────────────────────────
    lines.append("## Error Analysis (claim_status mismatches)")
    lines.append("")
    
    claim_errors = metrics["claim_status"].get("errors", [])
    if claim_errors:
        lines.append("| Row | User | Predicted | Expected |")
        lines.append("|---|---|---|---|")
        for err in claim_errors:
            lines.append(f"| {err['row']} | {err['user_id']} | {err['predicted']} | {err['expected']} |")
    else:
        lines.append("No claim_status errors!")
    lines.append("")
    
    # ── Operational analysis ───────────────────────────────────────────
    lines.append("## Operational Analysis")
    lines.append("")
    lines.append("### Model Calls")
    lines.append(f"- **Sample processing**: {usage['total_calls']} API calls")
    lines.append(f"- **Test processing (projected)**: ~45 API calls (Strategy A)")
    lines.append(f"- **Total for full run**: ~{usage['total_calls'] + 45} API calls")
    lines.append("")
    
    lines.append("### Token Usage")
    lines.append(f"- **Sample input tokens**: {usage['total_input_tokens']:,}")
    lines.append(f"- **Sample output tokens**: {usage['total_output_tokens']:,}")
    lines.append(f"- **Projected test input tokens**: ~{int(usage['total_input_tokens'] * 45 / max(num_claims, 1)):,}")
    lines.append(f"- **Projected test output tokens**: ~{int(usage['total_output_tokens'] * 45 / max(num_claims, 1)):,}")
    lines.append("")
    
    lines.append("### Images Processed")
    lines.append(f"- **Sample images**: {usage['total_images_processed']}")
    lines.append(f"- **Test images (projected)**: ~80")
    lines.append(f"- **Total**: ~{usage['total_images_processed'] + 80}")
    lines.append("")
    
    lines.append("### Cost Estimate")
    sample_cost = (usage['total_input_tokens'] / 1e6) * 0.05 + (usage['total_output_tokens'] / 1e6) * 0.08
    projected_test_cost = sample_cost * 45 / max(num_claims, 1)
    lines.append(f"- **Pricing**: Groq Llama 4 Scout @ $0.05/M input, $0.08/M output tokens")
    lines.append(f"- **Sample cost**: ${sample_cost:.4f}")
    lines.append(f"- **Projected test cost**: ${projected_test_cost:.4f}")
    lines.append(f"- **Total estimated cost**: ${sample_cost + projected_test_cost:.4f}")
    lines.append("")
    
    lines.append("### Latency & Runtime")
    lines.append(f"- **Sample runtime**: {runtime:.1f}s ({runtime/max(num_claims,1):.1f}s per claim)")
    lines.append(f"- **Projected test runtime**: ~{runtime * 45 / max(num_claims, 1):.0f}s")
    lines.append("")
    
    lines.append("### TPM/RPM & Rate Limit Strategy")
    lines.append("- **Approach**: Sequential processing with 0.5s delay between calls")
    lines.append("- **Retry strategy**: Exponential backoff (2s, 4s, 8s) on rate limit errors")
    lines.append("- **Max retries**: 3 per call")
    lines.append("- **Batching**: All images for one claim batched into a single API call")
    lines.append("- **Caching**: Not implemented (each claim is unique)")
    lines.append("- **Optimization**: Strategy A uses 1 call/claim; Strategy B uses 2 calls/claim")
    lines.append("")
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"   + Report written to {output_path}")


def run_evaluation(run_strategy_b: bool = False):
    """Run the full evaluation pipeline."""
    
    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE")
    print("=" * 60)
    
    # Load ground truth
    ground_truth = load_claims(SAMPLE_CLAIMS_CSV)
    num_claims = len(ground_truth)
    print(f"\n[*] Ground truth: {num_claims} labeled claims\n")
    
    # ── Strategy A ─────────────────────────────────────────────────────
    print("-" * 40)
    print("> Running Strategy A (Single-pass)...")
    print("-" * 40)
    
    reset_token_usage()
    sample_output_a = DATASET_DIR / "sample_output_a.csv"
    start_a = time.time()
    predictions_a = process_claims(SAMPLE_CLAIMS_CSV, sample_output_a)
    runtime_a = time.time() - start_a
    usage_a = get_token_usage()
    
    metrics_a = evaluate_predictions(predictions_a, ground_truth)
    
    print(f"\n[*] Strategy A Results:")
    print(f"   Composite: {metrics_a['composite_score']:.2%}")
    print(f"   Claim Status: {metrics_a['claim_status']['accuracy']:.2%}")
    print(f"   Issue Type: {metrics_a['issue_type']['accuracy']:.2%}")
    
    # ── Strategy B (optional) ──────────────────────────────────────────
    metrics_b = None
    usage_b = None
    runtime_b = None
    
    if run_strategy_b:
        print(f"\n{'-' * 40}")
        print("> Running Strategy B (Two-pass)...")
        print("-" * 40)
        
        reset_token_usage()
        sample_output_b = DATASET_DIR / "sample_output_b.csv"
        start_b = time.time()
        predictions_b = process_claims(SAMPLE_CLAIMS_CSV, sample_output_b, strategy="b")
        runtime_b = time.time() - start_b
        usage_b = get_token_usage()
        
        metrics_b = evaluate_predictions(predictions_b, ground_truth)
        
        print(f"\n[*] Strategy B Results:")
        print(f"   Composite: {metrics_b['composite_score']:.2%}")
        print(f"   Claim Status: {metrics_b['claim_status']['accuracy']:.2%}")
    
    # ── Generate report ────────────────────────────────────────────────
    report_path = Path(__file__).resolve().parent / "evaluation_report.md"
    generate_report(
        metrics=metrics_a,
        usage=usage_a,
        strategy_name="Strategy A",
        runtime=runtime_a,
        num_claims=num_claims,
        output_path=report_path,
        strategy_b_metrics=metrics_b,
        strategy_b_usage=usage_b,
        strategy_b_runtime=runtime_b,
    )
    
    print(f"\n{'='*60}")
    print(f"[OK] Evaluation complete! Report: {report_path}")
    print(f"{'='*60}\n")
    
    return metrics_a


if __name__ == "__main__":
    # Check if --compare flag is passed
    compare = "--compare" in sys.argv
    run_evaluation(run_strategy_b=compare)
