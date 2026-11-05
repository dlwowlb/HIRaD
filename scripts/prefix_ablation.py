#!/usr/bin/env python3
"""Inference-time prefix ablations on the synthetic test split.

Modes
  normal          unchanged prefix (reproduces the test-split predictions)
  zero            zero both the semantic token and the node tokens
  zero_zsem       zero the semantic token only
  zero_nodes      zero the node tokens only
  shuffle_global  build each prefix from another test clip (global permutation)

For shuffle_global the output keeps both the clip's own caption and the
caption of the clip whose prefix was used.

  python scripts/prefix_ablation.py --mode zero_zsem \
      --checkpoint checkpoints/hirad/stage3/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hirad import Captioner
from hirad.checkpoint import load_captioner_checkpoint
from hirad.data import make_test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--mode", required=True,
                    choices=["normal", "zero", "zero_zsem", "zero_nodes", "shuffle_global"])
    ap.add_argument("--out_dir", default="test/hirad")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model = Captioner(cfg).to(device)
    load_captioner_checkpoint(model, args.checkpoint, device, lora_path=args.lora_path)
    model.eval()

    loader = make_test_loader(cfg, batch_size=args.batch_size)
    assert loader is not None, "no test split configured"

    results = []
    with torch.no_grad():
        if args.mode == "shuffle_global":
            z_all, n_all, m_all, gts = [], [], [], []
            for batch in tqdm(loader, desc="encode"):
                pc = batch["point_cloud"].to(device)
                mk = batch["radar_mask"].to(device)
                out = model.latent_graph.encode_full(pc, mk)
                z_all.append(out["z_sem"].cpu())
                n_all.append(out["node_emb"].cpu())
                m_all.append(mk.cpu())
                gts.extend(batch["texts"])
            z_all = torch.cat(z_all)
            n_all = torch.cat(n_all)
            m_all = torch.cat(m_all)
            N = len(gts)
            perm = torch.randperm(N, generator=torch.Generator().manual_seed(args.seed)).tolist()

            for s in tqdm(range(0, N, args.batch_size), desc="generate"):
                idx = list(range(s, min(s + args.batch_size, N)))
                src = [perm[i] for i in idx]
                fake = {"z_sem": z_all[src].to(device), "node_emb": n_all[src].to(device)}
                caps = model.generate_from_prefix(fake, m_all[idx].to(device))
                for i, j, cap in zip(idx, src, caps):
                    results.append({
                        "ground_truth_self": gts[i],
                        "ground_truth_source": gts[j],
                        "source_idx": int(j),
                        "prediction": cap,
                    })
        else:
            for batch in tqdm(loader, desc=args.mode):
                pc = batch["point_cloud"].to(device)
                mk = batch["radar_mask"].to(device)
                out = model.latent_graph.encode_full(pc, mk)
                if args.mode in ("zero", "zero_zsem"):
                    out["z_sem"] = torch.zeros_like(out["z_sem"])
                if args.mode in ("zero", "zero_nodes"):
                    out["node_emb"] = torch.zeros_like(out["node_emb"])
                caps = model.generate_from_prefix(out, mk)
                results.extend({"ground_truth": gt, "prediction": pr}
                               for gt, pr in zip(batch["texts"], caps))

    out_path = Path(args.out_dir) / f"predictions_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"{len(results)} predictions written to {out_path}")


if __name__ == "__main__":
    main()
