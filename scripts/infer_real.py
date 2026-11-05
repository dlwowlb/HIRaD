#!/usr/bin/env python3
"""Caption real radar recordings.

Each input is an .npz file with
    point_cloud       (T, P, D) float32, D >= 4
    num_radar_frames  () int, optional, number of valid frames
    action            () str, optional label kept in the output for reference

  python scripts/infer_real.py --config configs/default.yaml \
      --checkpoint checkpoints/hirad/stage3/best.pt \
      --input <npz file or directory> --output predictions_real.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hirad import Captioner
from hirad.checkpoint import load_captioner_checkpoint


def load_npz(path, max_frames, points_per_frame, point_dims):
    data = np.load(path, allow_pickle=True)
    pc = data["point_cloud"].astype(np.float32)
    T_src = int(data["num_radar_frames"]) if "num_radar_frames" in data.files else pc.shape[0]
    T_use = min(T_src, pc.shape[0], max_frames)
    P_use = min(pc.shape[1], points_per_frame)
    D_use = min(pc.shape[-1], point_dims)

    pc_p = np.zeros((max_frames, points_per_frame, point_dims), dtype=np.float32)
    pc_p[:T_use, :P_use, :D_use] = pc[:T_use, :P_use, :D_use]
    mask = np.zeros(max_frames, dtype=np.bool_)
    mask[:T_use] = True
    action = str(data["action"]) if "action" in data.files else ""
    return pc_p, mask, T_use, action


def resolve_inputs(arg):
    p = Path(arg)
    if p.is_dir():
        files = sorted(p.glob("*.npz"))
        if not files:
            raise SystemExit(f"no .npz files in {p}")
        return files
    if not p.exists():
        raise SystemExit(f"input not found: {p}")
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help="an .npz file or a directory of .npz files")
    ap.add_argument("--output", default="predictions_real.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--skip_done", action="store_true", help="skip files already present in --output")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    inputs = resolve_inputs(args.input)
    out_path = Path(args.output)
    results = []
    if args.skip_done and out_path.exists():
        results = json.loads(out_path.read_text()).get("predictions", [])
        done = {r["file"] for r in results}
        inputs = [p for p in inputs if str(p) not in done]

    model = Captioner(cfg).to(device)
    ck = load_captioner_checkpoint(model, args.checkpoint, device, lora_path=args.lora_path)
    print(f"checkpoint epoch {ck.get('epoch', '?')}")
    model.eval()

    ds = cfg["dataset"]
    lg = cfg["latent_graph"]
    max_frames = ds.get("max_seq_len", 100)
    ppf = ds.get("points_per_frame", lg.get("num_points", 128))
    pdims = lg.get("point_in_dim", 4)
    prompt = cfg.get("captioner", {}).get("prompt", "Describe the motion:")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(inputs, 1):
        pc_p, mask, T_use, action = load_npz(path, max_frames, ppf, pdims)
        pc_t = torch.from_numpy(pc_p).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

        caption = model.generate(
            pc_t, mask_t,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p)[0].strip()
        if caption.startswith(prompt):
            caption = caption[len(prompt):].strip()

        print(f"[{i}/{len(inputs)}] {path.name}  T={T_use}  {caption}")
        results.append({
            "file": str(path),
            "num_frames": int(T_use),
            "action": action,
            "prediction": caption,
        })
        out_path.write_text(json.dumps({
            "checkpoint": args.checkpoint,
            "config": args.config,
            "predictions": results,
        }, indent=2, ensure_ascii=False))

    print(f"{len(results)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
