"""Create leak-free IEMOCAP leave-one-session-out manifests from real metadata.

Input CSV must contain ``utterance_id,session,speaker``. Each test session is
held out entirely; its speakers are excluded from both training and validation.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--metadata-csv",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
 with Path(a.metadata_csv).open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
 required={"utterance_id","session","speaker"}; missing=required-set(rows[0] if rows else [])
 if missing: raise ValueError("metadata CSV missing: "+", ".join(sorted(missing)))
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); sessions=sorted({r["session"] for r in rows})
 summary=[]
 for test_session in sessions:
  test=[r for r in rows if r["session"]==test_session]; forbidden={r["speaker"] for r in test}
  candidates=[r for r in rows if r["session"]!=test_session and r["speaker"] not in forbidden]
  validation_session=next((s for s in sessions if s!=test_session and not ({r["speaker"] for r in rows if r["session"]==s}&forbidden)),None)
  if validation_session is None: raise ValueError(f"cannot choose speaker-disjoint validation session for {test_session}")
  valid=[r for r in candidates if r["session"]==validation_session]; train=[r for r in candidates if r["session"]!=validation_session]
  payload={"test_session":test_session,"validation_session":validation_session,"train_ids":[r["utterance_id"] for r in train],"validation_ids":[r["utterance_id"] for r in valid],"test_ids":[r["utterance_id"] for r in test],"excluded_test_speakers":sorted(forbidden)}
  (out/f"fold_{test_session}.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); summary.append({"fold":test_session,"n_train":len(train),"n_validation":len(valid),"n_test":len(test),"test_speakers":";".join(sorted(forbidden))})
 with (out/"fold_summary.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
if __name__=="__main__": main()
