import csv
from pathlib import Path

gt = Path(__file__).parent.parent / "dataset" / "sample_claims.csv"
rows = list(csv.DictReader(open(gt, encoding="utf-8")))
for r in rows:
    print(f"{r['user_id']:10s} | {r['claim_object']:8s} | status={r.get('claim_status','?'):25s} | issue={r.get('issue_type','?'):20s} | part={r.get('object_part','?'):16s} | sev={r.get('severity','?'):8s} | evid={r.get('evidence_standard_met','?'):6s} | valid={r.get('valid_image','?'):6s} | flags={r.get('risk_flags','?')}")
