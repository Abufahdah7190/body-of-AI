"""Operational trial summaries, not a claim of standards certification."""
import json
from pathlib import Path
from datetime import datetime,timezone

def record_trial(path,case,event,attempt,profile,pad=None):
    outcome="accepted" if event=="LIVENESS_DEMO_PASSED" else "rejected" if event in ("CHALLENGE_TIMEOUT","PAD_REJECTED") else "inconclusive"
    record=dict(time=datetime.now(timezone.utc).isoformat(),kind=case["kind"],condition=case["condition"],outcome=outcome,event=event,attempt=attempt,profile=profile,pad=pad or {})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",encoding="utf-8") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")

def summarize(records):
    report={}
    for r in records:
        group="live" if r["kind"]=="live" else r["kind"]
        key=group+" | "+r["condition"]+" | "+r["profile"]
        counts=report.setdefault(key,dict(total=0,accepted=0,rejected=0,inconclusive=0))
        if r["outcome"] not in ("accepted","rejected","inconclusive"):raise ValueError("Invalid outcome")
        counts["total"]+=1;counts[r["outcome"]]+=1
    for key,c in report.items():
        decided=c["accepted"]+c["rejected"]
        errors=c["rejected"] if key.startswith("live |") else c["accepted"]
        c["error_rate_decided"]=errors/decided if decided else None
        c["inconclusive_rate"]=c["inconclusive"]/c["total"]
    return report
