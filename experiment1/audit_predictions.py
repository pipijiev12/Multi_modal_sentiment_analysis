"""Audit that manuscript tables use identical saved test targets and runs."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

def digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", default="eval")
    p.add_argument("--batch-ids", nargs="+", default=["1","2","3","4","5"])
    p.add_argument("--datasets", nargs="+", default=["cmumosei","cmumosi","iemocap"])
    p.add_argument("--models", nargs="+", default=["qrsan"])
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(); root = Path(a.input_root); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for dataset in a.datasets:
        for model in a.models:
            for batch in a.batch_ids:
                path=root/f"matrix_{batch}"/dataset/f"{model}.predictions.npz"
                row={"dataset":dataset,"model":model,"run":f"matrix_{batch}","prediction_file":str(path),"exists":path.is_file()}
                if path.is_file():
                    with np.load(path) as z:
                        targets=np.asarray(z["targets"]); outputs=np.asarray(z["outputs"])
                    row.update({"n_test_instances":int(targets.shape[0]),"target_shape":list(targets.shape),"target_sha256":digest(targets),"output_shape":list(outputs.shape)})
                rows.append(row)
    fields=sorted({key for row in rows for key in row})
    with (out/"prediction_audit.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    grouped={}
    for row in rows:
        grouped.setdefault((row["dataset"],row["model"]),[]).append(row)
    consistency={f"{d}/{m}": len({r.get("target_sha256") for r in rs}) == 1 and all(r["exists"] for r in rs) for (d,m),rs in grouped.items()}
    (out/"prediction_audit.json").write_text(json.dumps({"target_consistent_across_runs":consistency,"rows":rows},indent=2)+"\n",encoding="utf-8")
    return 0 if all(consistency.values()) else 1
if __name__ == "__main__": raise SystemExit(main())
