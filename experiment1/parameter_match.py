"""Choose and write the closest parameter-matched RealQRSAN configuration."""
from __future__ import annotations
import argparse, configparser, json
from pathlib import Path
import torch
from dataset import setup as setup_dataset
from models import setup as setup_model
from utils.params import Params

ROOT = Path(__file__).resolve().parents[1]

def count(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
def parse_dims(value):
    dims=tuple(int(x) for x in value.split(","))
    if len(dims)!=3 or min(dims)<1: raise ValueError("candidate dimensions must be three positive integers, e.g. 10,10,10")
    return dims
def instantiate(config_path, network_type, dims):
    params=Params(); params.parse_config(str(config_path)); params.network_type=network_type; params.contracted_dims=",".join(map(str,dims)); params.device=torch.device("cpu")
    reader=setup_dataset(params); reader.read(params); params.reader=reader
    return count(setup_model(params)), params
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-config",required=True); p.add_argument("--candidate-dims",nargs="+",required=True,help="e.g. 8,8,8 9,9,9 10,10,10")
    p.add_argument("--output-config",required=True); p.add_argument("--report",required=True); p.add_argument("--run-id",default="parameter_matched_real"); p.add_argument("--max-relative-difference",type=float,default=0.01); a=p.parse_args()
    target_config=Path(a.target_config)
    parser=configparser.ConfigParser(); parser.read(target_config,encoding="utf-8"); target_dims=parse_dims(parser["COMMON"]["contracted_dims"])
    target_count,_=instantiate(target_config,"qrsan",target_dims)
    candidates=[]
    for raw in a.candidate_dims:
        dims=parse_dims(raw); value,_=instantiate(target_config,"real-qrsan",dims)
        candidates.append({"contracted_dims":raw,"trainable_parameters":value,"relative_difference":abs(value-target_count)/target_count})
    selected=min(candidates,key=lambda row: row["relative_difference"])
    parser["COMMON"]["network_type"]="real-qrsan"; parser["COMMON"]["contracted_dims"]=selected["contracted_dims"]
    parser["COMMON"]["dir_name"]=f"experiment1/{a.run_id}"
    parser["COMMON"]["output_file"]=f"eval/experiment1/{a.run_id}/metrics.csv"
    parser["COMMON"]["prediction_file"]=f"eval/experiment1/{a.run_id}/predictions.npz"
    output=Path(a.output_config); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",encoding="utf-8") as f: parser.write(f)
    report={"target_config":str(target_config),"target_trainable_parameters":target_count,"selected":selected,"all_candidates":candidates,"within_requested_tolerance":selected["relative_difference"]<=a.max_relative_difference}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if not report["within_requested_tolerance"]: raise SystemExit("No candidate met --max-relative-difference; widen the candidate grid and report the final mismatch.")
if __name__=="__main__": main()
