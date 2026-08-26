"""Export auditable implementation metadata from real configs, logs and hardware."""
from __future__ import annotations
import argparse, configparser, csv, json, platform, sys
from pathlib import Path

FIELDS = ("dataset_name","network_type","seed","batch_size","lr","clip","epochs","text_hidden_dim","contracted_dims","measurement_size","output_cell_dim","subnet_dropout_rates","output_dropout_rate","optimizer","weight_decay","scheduler","early_stopping","validation_metric","max_seq_len")
def main() -> int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config-root",required=True); p.add_argument("--output-dir",required=True); a=p.parse_args()
 rows=[]
 for path in Path(a.config_root).rglob("*.ini"):
  c=configparser.ConfigParser(); c.read(path,encoding="utf-8"); common=c["COMMON"]
  row={"config_file":str(path),**{field:common.get(field,"") for field in FIELDS}}
  rows.append(row)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 with (out/"implementation_details.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=["config_file",*FIELDS]); w.writeheader(); w.writerows(rows)
 hardware={"python":sys.version,"platform":platform.platform()}
 try:
  import torch
  hardware.update({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda":torch.version.cuda,"gpu_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
 except ImportError: hardware["torch"]="AUTHOR_INPUT_NEEDED: torch unavailable"
 (out/"software_hardware.json").write_text(json.dumps(hardware,indent=2)+"\n",encoding="utf-8")
 (out/"missing_fields.json").write_text(json.dumps({"fields_requiring_real_logs_or_explicit_config": [field for field in ("optimizer","weight_decay","scheduler","early_stopping","validation_metric","max_seq_len") if any(not row[field] for row in rows)],"runtime_fields_requiring_benchmark": ["parameter_count","training_time_per_epoch","inference_latency","peak_gpu_memory"]},indent=2)+"\n",encoding="utf-8")
 return 0
if __name__=="__main__": raise SystemExit(main())
