"""Measure parameter count, step time, inference latency and peak GPU memory.

All supplied INI files must use the same dataset, batch size and hardware for
their rows to be compared. Results are measurements, never estimates.
"""
from __future__ import annotations
import argparse, csv, json, platform, sys, time
from pathlib import Path

# These scripts are invoked as ``python experiment1/<script>.py``.  In that
# mode Python adds experiment1/ rather than the repository root to sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from dataset import setup as setup_dataset
from models import setup as setup_model
from utils.model import get_criterion, get_loss
from utils.params import Params

def sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)

def one_config(path: Path, warmup: int, repeats: int) -> dict:
    params=Params(); params.parse_config(str(path)); params.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader=setup_dataset(params); reader.read(params); params.reader=reader
    model=setup_model(params).to(params.device); model.train()
    batch=next(iter(reader.get_data(iterable=True,shuffle=False,split="train")))
    inputs=[x.to(params.device) for x in batch[:-1]]; target=batch[-1].to(params.device)
    optimizer=torch.optim.RMSprop(model.parameters(),lr=params.lr); criterion=get_criterion(params)
    def step():
        optimizer.zero_grad(); output=model(inputs)
        if output.shape != target.shape: output=output.reshape_as(target)
        get_loss(params,criterion,output,target).backward(); optimizer.step()
    for _ in range(warmup): step()
    if params.device.type=="cuda": torch.cuda.reset_peak_memory_stats(params.device)
    sync(params.device); started=time.perf_counter()
    for _ in range(repeats): step()
    sync(params.device); train_ms=1000*(time.perf_counter()-started)/repeats
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(inputs)
        sync(params.device); started=time.perf_counter()
        for _ in range(repeats): model(inputs)
        sync(params.device); infer_ms=1000*(time.perf_counter()-started)/repeats
    gpu_name = torch.cuda.get_device_name(params.device) if params.device.type == "cuda" else "CPU"
    return {"config_file":str(path),"dataset":params.dataset_name,"model":path.stem,"network_type":params.network_type,"batch_size":params.batch_size,"parameters":sum(p.numel() for p in model.parameters()),"train_step_ms":train_ms,"inference_batch_ms":infer_ms,"inference_example_ms":infer_ms/params.batch_size,"peak_gpu_mib":torch.cuda.max_memory_allocated(params.device)/2**20 if params.device.type=="cuda" else 0.0,"device":str(params.device),"gpu_name":gpu_name,"pytorch":torch.__version__,"python":platform.python_version(),"warmup_steps":warmup,"measurement_steps":repeats}

def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--configs",nargs="+",required=True); p.add_argument("--output",required=True); p.add_argument("--warmup",type=int,default=10); p.add_argument("--repeats",type=int,default=30); p.add_argument("--require-same-batch",action="store_true",help="Fail if models within a dataset use different batch sizes."); a=p.parse_args()
 rows=[one_config(Path(x),a.warmup,a.repeats) for x in a.configs]
 if a.require_same_batch:
  for dataset in sorted({row["dataset"] for row in rows}):
   sizes={row["batch_size"] for row in rows if row["dataset"]==dataset}
   if len(sizes)!=1: raise SystemExit(f"Incomparable benchmark: {dataset} batch sizes differ: {sorted(sizes)}")
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
if __name__=="__main__": main()
