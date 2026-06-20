import csv
from pathlib import Path

gt_path = Path(__file__).parent.parent / "dataset" / "sample_claims.csv"
pred_path = Path(__file__).parent.parent / "dataset" / "sample_v2_output.csv"

gt = list(csv.DictReader(open(gt_path, encoding="utf-8")))
pred = list(csv.DictReader(open(pred_path, encoding="utf-8")))

fields = ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image", "risk_flags", "supporting_image_ids"]

for i, (g, p) in enumerate(zip(gt, pred)):
    uid = g["user_id"]
    has_error = False
    for f in fields:
        gv = g.get(f, "").strip().lower()
        pv = p.get(f, "").strip().lower()
        if gv != pv:
            has_error = True
    if has_error:
        print(f"\n=== {uid} ({g['claim_object']}) ===")
        for f in fields:
            gv = g.get(f, "").strip().lower()
            pv = p.get(f, "").strip().lower()
            mark = "X" if gv != pv else " "
            print(f"  [{mark}] {f:30s} GT={gv:30s} PRED={pv}")
