"""Capture current machine, accelerator and software metadata without mutation."""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args: str) -> str:
    try:
        return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except FileNotFoundError:
        return "unavailable"


def cpu_model() -> str:
    if os.path.isfile("/proc/cpuinfo"):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def memory_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    if os.path.isfile("/proc/meminfo"):
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return {"total_kib": values.get("MemTotal", 0), "available_kib": values.get("MemAvailable", 0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval/experiment1/audits/hardware")
    args = parser.parse_args()
    try:
        import torch
        torch_info: dict[str, object] = {"pytorch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda}
    except ImportError:
        torch_info = {"pytorch": "unavailable", "cuda_available": False, "cuda_runtime": None}
    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version},
        "cpu": {"model": cpu_model(), "logical_cores": os.cpu_count(), "lscpu": command("lscpu")},
        "memory": memory_kib(),
        "software": torch_info,
        "gpu_query": command("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits"),
        "gpu_process_query": command("nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory", "--format=csv,noheader,nounits"),
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output / f"hardware_environment_{stamp}.json"
    markdown_path = output / f"hardware_environment_{stamp}.md"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "# Hardware environment capture\n\n"
        f"- Captured (UTC): {metadata['captured_at_utc']}\n"
        f"- Host: {metadata['host']['hostname']}\n"
        f"- CPU: {metadata['cpu']['model']} ({metadata['cpu']['logical_cores']} logical cores)\n"
        f"- RAM total: {metadata['memory']['total_kib'] / 1024 / 1024:.2f} GiB\n"
        f"- PyTorch: {torch_info['pytorch']}; CUDA runtime: {torch_info['cuda_runtime']}\n\n"
        "## GPU snapshot\n\n```text\n" + metadata["gpu_query"] + "\n```\n\n"
        "## Active GPU compute processes\n\n```text\n" + metadata["gpu_process_query"] + "\n```\n",
        encoding="utf-8",
    )
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
