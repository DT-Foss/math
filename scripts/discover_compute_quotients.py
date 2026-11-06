#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from wirkung_math.result_discovery import ingest_result_receipt


def main():
    p=argparse.ArgumentParser(
        description="Discover measured compute-quotient opportunities in Wirkung result receipts."
    )
    p.add_argument("paths",nargs="+",help="JSON result files or directories")
    p.add_argument("--top",type=int,default=50)
    args=p.parse_args()
    files=[]
    for raw in args.paths:
        path=Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    rows=[]
    for path in files:
        try:
            receipt=json.loads(path.read_text())
        except Exception as exc:
            rows.append({"file":str(path),"error":str(exc)})
            continue
        for route in ingest_result_receipt(receipt):
            rows.append({
                "file":str(path),
                "route":route.name,
                "evidence":route.evidence.name,
                "semantic_success":route.metadata.get("semantic_success"),
                "cost_win":route.metadata.get("cost_win"),
                "speedup":route.metadata.get("speedup"),
                "receipt_path":route.metadata.get("receipt_path"),
                "state_bytes":route.state_bytes,
            })
    good=[r for r in rows if r.get("semantic_success") and r.get("cost_win")]
    semantic_loss=[r for r in rows if r.get("semantic_success") and not r.get("cost_win")]
    failed=[r for r in rows if r.get("semantic_success") is False]
    key=lambda r:float(r.get("speedup") or 0.0)
    result={
        "schema":"compute-quotient-discovery/v1",
        "files_scanned":len(files),
        "measured_wins":sorted(good,key=key,reverse=True)[:args.top],
        "semantic_wins_hardware_losses":sorted(semantic_loss,key=key,reverse=True)[:args.top],
        "semantic_failures":sorted(failed,key=key,reverse=True)[:args.top],
        "parse_errors":[r for r in rows if "error" in r],
    }
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
