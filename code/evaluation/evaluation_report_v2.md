# Evaluation Report -- Multi-Modal Evidence Review

## Summary

- **Claims evaluated**: 20
- **Primary strategy**: Strategy A (Single-pass)
- **Composite score**: 70.23%

## Per-Field Accuracy

| Field | Metric | Score |
|---|---|---|
| `claim_status` | Accuracy | 75.00% (15/20) |
| `issue_type` | Accuracy | 45.00% (9/20) |
| `object_part` | Accuracy | 80.00% (16/20) |
| `evidence_standard_met` | Accuracy | 95.00% (19/20) |
| `valid_image` | Accuracy | 90.00% (18/20) |
| `severity` | Accuracy | 40.00% (8/20) |
| `risk_flags` | F1 | 64.52% (P=80.00% R=54.05%) |
| `supporting_image_ids` | F1 | 90.48% (P=90.48% R=90.48%) |

## Error Analysis (claim_status mismatches)

| Row | User | Predicted | Expected |
|---|---|---|---|
| 2 | user_002 | contradicted | supported |
| 5 | user_005 | supported | contradicted |
| 14 | user_020 | supported | contradicted |
| 18 | user_032 | supported | not_enough_information |
| 20 | user_034 | supported | contradicted |

## Operational Analysis

### Model Calls
- **Sample processing**: 20 API calls
- **Test processing (projected)**: ~45 API calls (Strategy A)
- **Total for full run**: ~65 API calls

### Token Usage
- **Sample input tokens**: 15,000
- **Sample output tokens**: 4,000
- **Projected test input tokens**: ~33,750
- **Projected test output tokens**: ~9,000

### Images Processed
- **Sample images**: 30
- **Test images (projected)**: ~80
- **Total**: ~110

### Cost Estimate
- **Pricing**: Groq Llama 4 Scout @ $0.05/M input, $0.08/M output tokens
- **Sample cost**: $0.0011
- **Projected test cost**: $0.0024
- **Total estimated cost**: $0.0035

### Latency & Runtime
- **Sample runtime**: 60.0s (3.0s per claim)
- **Projected test runtime**: ~135s

### TPM/RPM & Rate Limit Strategy
- **Approach**: Sequential processing with 0.5s delay between calls
- **Retry strategy**: Exponential backoff (2s, 4s, 8s) on rate limit errors
- **Max retries**: 3 per call
- **Batching**: All images for one claim batched into a single API call
- **Caching**: Not implemented (each claim is unique)
- **Optimization**: Strategy A uses 1 call/claim; Strategy B uses 2 calls/claim
