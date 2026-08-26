"""Audit available IEMOCAP split metadata without inferring speaker/session IDs."""
from __future__ import annotations
import argparse,json,pickle
from pathlib import Path
def main() -> int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset-pickle",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
 with Path(a.dataset_pickle).open("rb") as f: data=pickle.load(f)
 report={"dataset_pickle":a.dataset_pickle,"splits":{},"speaker_session_audit":"AUTHOR_INPUT_NEEDED"}
 for split,value in data.items():
  if not isinstance(value,dict): continue
  keys=sorted(value); count=None
  for key in ("labels","emotion","sentiment","text"):
   if key in value:
    try: count=len(value[key]); break
    except TypeError: pass
  report["splits"][split]={"keys":keys,"n_samples":count,"speaker_key_present":any("speaker" in key.lower() for key in keys),"session_key_present":any("session" in key.lower() for key in keys)}
 if not all(item["speaker_key_present"] and item["session_key_present"] for item in report["splits"].values()):
  report["speaker_session_audit"]="AUTHOR_INPUT_NEEDED: prepared pickle has no complete speaker/session metadata; provide original utterance IDs and official session split before claiming speaker-disjoint evaluation."
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"iemocap_split_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
 return 0
if __name__=="__main__": raise SystemExit(main())
