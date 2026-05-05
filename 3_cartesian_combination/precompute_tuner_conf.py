"""Compute tuner_conf/conf.json without loading the full model into host RAM.

The upstream `tuner_conf.py` calls `AutoModelForCausalLM.from_pretrained(...)`
just to count parameters, which materializes the full 24+ GiB of Llama-2-13B
weights in host RAM and OOM-kills tight Slurm allocations. This helper
reproduces the same computation by reading:

  - model_path/config.json                          → max_seq_len, hidden, layers
  - model_path/model.safetensors.index.json         → total_size in bytes
  - nvidia-smi                                      → GPU VRAM (GiB)

and writes the same {max_sequence_length, min_world_size} JSON the upstream
tuner_conf.py would have produced. Same rounding rule as the original:
ceil(total_mem / (0.9 * VRAM)) then round up to the next power of 2 in {1,2,4,8}.

Usage:
    python precompute_tuner_conf.py <model_path> <output_conf_json>
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


def gpu_vram_gib() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(out.strip().split("\n")[0]) / 1024.0


def model_param_size_gib(model_dir: Path) -> float:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        total_bytes = d.get("metadata", {}).get("total_size", 0)
        if total_bytes:
            return total_bytes / 1024**3
    # Fallback: sum .safetensors file sizes on disk (roughly = weight bytes).
    sizes = [p.stat().st_size for p in model_dir.glob("*.safetensors")]
    if sizes:
        return sum(sizes) / 1024**3
    raise RuntimeError(f"Cannot determine model size from {model_dir}")


def round_up_world_size(min_ws: float) -> int:
    if min_ws <= 1:
        return 1
    if min_ws <= 2:
        return 2
    if min_ws <= 4:
        return 4
    return 8


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: precompute_tuner_conf.py <model_path> <output_conf_json>",
              file=sys.stderr)
        sys.exit(2)
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    cfg = json.loads((model_dir / "config.json").read_text())
    max_seq_len = cfg.get("max_position_embeddings",
                          cfg.get("max_sequence_length",
                                  cfg.get("model_max_length",
                                          cfg.get("seq_length", 4096))))
    hidden = cfg["hidden_size"]
    num_layers = cfg.get("num_hidden_layers", cfg.get("num_layers", 32))

    model_size_gib = model_param_size_gib(model_dir)
    # KV cache for one max-length sequence at fp16 (matches tuner_conf.py formula).
    kv_size_gib = max_seq_len * 2 * num_layers * hidden * 2 / 1024**3

    vram = gpu_vram_gib()
    capacity = 0.9 * vram
    total = model_size_gib + kv_size_gib
    raw_mws = math.ceil(total / capacity)
    mws = round_up_world_size(raw_mws)

    conf = {"max_sequence_length": int(max_seq_len), "min_world_size": int(mws)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(conf))

    print(f"model_size={model_size_gib:.2f} GiB  "
          f"kv_for_max_seq={kv_size_gib:.2f} GiB  "
          f"gpu_vram={vram:.2f} GiB  "
          f"capacity(0.9x)={capacity:.2f} GiB  "
          f"raw_min_world_size={raw_mws}  "
          f"rounded={mws}")
    print(f"Wrote {out_path}: {conf}")


if __name__ == "__main__":
    main()
