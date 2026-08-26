"""Evaluate a fixed trained QRSAN under identity and random basis permutations."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from dataset import setup as setup_dataset
from utils.params import Params

def outputs(model, reader, device):
 values=[]; targets=[]
 with torch.no_grad():
  for batch in reader.get_data(iterable=True,shuffle=False,split="test"):
   values.append(model([x.to(device) for x in batch[:-1]]).detach().cpu()); targets.append(batch[-1].cpu())
 return torch.cat(values).numpy(),torch.cat(targets).numpy()
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True); p.add_argument("--model-file",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--permutations",type=int,default=10); p.add_argument("--seed",type=int,default=20260825); a=p.parse_args()
 params=Params(); params.parse_config(a.config); params.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); reader=setup_dataset(params); reader.read(params); params.reader=reader
 model=torch.load(a.model_file,weights_only=False).to(params.device).eval()
 if not hasattr(model,"set_basis_permutation"): raise TypeError("model lacks QRSAN basis-permutation interface")
 model.set_basis_permutation(); reference,target=outputs(model,reader,params.device); rng=np.random.default_rng(a.seed); rows=[]
 for index in range(a.permutations):
  model.set_basis_permutation(rng.permutation(model.measurement_dim)); value,_=outputs(model,reader,params.device)
  rows.append({"permutation":index+1,"mean_abs_output_change":float(np.mean(np.abs(value-reference))),"binary_accuracy":float(np.mean((value.reshape(-1)>=0)==(target.reshape(-1)>=0))) if params.label=="sentiment" else float(np.mean(value.reshape(target.shape).argmax(-1)==target.argmax(-1)))})
 model.set_basis_permutation(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 with (out/"basis_permutation_runs.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
 (out/"basis_permutation_summary.json").write_text(json.dumps({"permutations":a.permutations,"seed":a.seed,"mean_abs_output_change_mean":float(np.mean([r["mean_abs_output_change"] for r in rows])),"mean_abs_output_change_sd":float(np.std([r["mean_abs_output_change"] for r in rows],ddof=1))},indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
