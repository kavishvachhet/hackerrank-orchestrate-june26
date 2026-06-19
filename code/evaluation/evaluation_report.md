# Evaluation Report -- Multi-Modal Evidence Review

## Summary

- **Claims evaluated**: 20
- **Primary strategy**: Strategy A (Single-pass)
- **Composite score**: 67.24%

## Per-Field Accuracy

| Field | Metric | Score |
|---|---|---|
| `claim_status` | Accuracy | 70.00% (14/20) |
| `issue_type` | Accuracy | 45.00% (9/20) |
| `object_part` | Accuracy | 80.00% (16/20) |
| `evidence_standard_met` | Accuracy | 80.00% (16/20) |
| `valid_image` | Accuracy | 80.00% (16/20) |
| `severity` | Accuracy | 50.00% (10/20) |
| `risk_flags` | F1 | 67.65% (P=74.19% R=62.16%) |
| `supporting_image_ids` | F1 | 74.42% (P=72.73% R=76.19%) |

## Error Analysis (claim_status mismatches)

| Row | User | Predicted | Expected |
|---|---|---|---|
| 5 | user_005 | supported | contradicted |
| 7 | user_003 | not_enough_information | supported |
| 8 | user_008 | not_enough_information | contradicted |
| 14 | user_020 | supported | contradicted |
| 19 | user_033 | not_enough_information | contradicted |
| 20 | user_034 | not_enough_information | contradicted |

## Operational Analysis

### Model Calls
- **Sample processing**: 20 API calls
- **Test processing (projected)**: ~45 API calls (Strategy A)
- **Total for full run**: ~65 API calls

### Token Usage
- **Sample input tokens**: 55,301
- **Sample output tokens**: 2,847
- **Projected test input tokens**: ~124,427
- **Projected test output tokens**: ~6,405

### Images Processed
- **Sample images**: 29
- **Test images (projected)**: ~80
- **Total**: ~109

### Cost Estimate
- **Pricing**: Groq Llama 4 Scout @ $0.05/M input, $0.08/M output tokens
- **Sample cost**: $0.0030
- **Projected test cost**: $0.0067
- **Total estimated cost**: $0.0097

### Latency & Runtime
- **Sample runtime**: 426.6s (21.3s per claim)
- **Projected test runtime**: ~960s

### TPM/RPM & Rate Limit Strategy
- **Approach**: Sequential processing with 0.5s delay between calls
- **Retry strategy**: Exponential backoff (2s, 4s, 8s) on rate limit errors
- **Max retries**: 3 per call
- **Batching**: All images for one claim batched into a single API call
- **Caching**: Not implemented (each claim is unique)
- **Optimization**: Strategy A uses 1 call/claim; Strategy B uses 2 calls/claim
