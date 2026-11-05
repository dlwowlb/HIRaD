#!/usr/bin/env python3
"""Three-stage training and test-split inference.

  python train.py --stage pretrain_lg --config configs/default.yaml
  python train.py --stage caption     --lg_ckpt      checkpoints/hirad/stage1/best.pt
  python train.py --stage joint       --caption_ckpt checkpoints/hirad/stage2/best.pt
  python train.py --stage test        --caption_ckpt checkpoints/hirad/stage3/best.pt
  python train.py --stage all

Stage 1 trains the latent graph with L_obs, L_KL, L_ctr and L_sem.
Stage 2 freezes the latent state and trains the semantic pathway, the prefix
projections and the QLoRA adapters with L_cap.
Stage 3 fine-tunes everything jointly with per-group learning rates.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from hirad import LatentGraph, Captioner
from hirad.data import make_loaders, make_test_loader
from hirad.checkpoint import load_captioner_checkpoint, captioner_state


def safe_device(dev):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.startswith("cuda:") and int(dev.split(":")[1]) >= torch.cuda.device_count():
        return torch.device("cuda:0")
    return torch.device(dev)


class LossTracker:
    def __init__(self, keys):
        self.keys = keys
        self.sums = {k: 0.0 for k in keys}
        self.count = 0

    def update(self, out):
        for k in self.keys:
            if k in out:
                v = out[k]
                self.sums[k] += v.item() if torch.is_tensor(v) else v
        self.count += 1

    def avg(self):
        n = max(self.count, 1)
        return {k: v / n for k, v in self.sums.items()}

    def summary(self):
        return " ".join(f"{k}={v:.4f}" for k, v in self.avg().items() if v != 0.0)


def to_device(batch, device):
    pc = batch["point_cloud"].to(device)
    mask = batch["radar_mask"].to(device)
    j = batch["motion_joints"]
    if j is not None:
        j = j.to(device)
    return pc, mask, j


@torch.no_grad()
def preview_captions(model, loader, device, n=5):
    model.eval()
    shown = 0
    for batch in loader:
        pc, mask, j = to_device(batch, device)
        caps = model.generate(pc, mask, motion_joints=j)
        for gt, pr in zip(batch["texts"], caps):
            if shown >= n:
                break
            print(f"  gt:   {gt}\n  pred: {pr}")
            shown += 1
        if shown >= n:
            break
    model.train()


def stage_cfg(config, name):
    return config.get("training", {}).get(name, {})


# ---------------------------------------------------------------- stage 1

def train_stage1(config, args):
    lg_cfg = config["latent_graph"]
    sc = stage_cfg(config, "stage1")
    device = safe_device(sc.get("device", args.device))
    print("\n== Stage 1: latent graph")

    model = LatentGraph(config).to(device)
    lr = float(sc.get("lr", lg_cfg.get("lr", 1e-4)))
    epochs = sc.get("epochs", 100)
    warmup = sc.get("warmup_epochs", lg_cfg.get("warmup_epochs", 5))

    opt = optim.AdamW(model.parameters(), lr=lr,
                      weight_decay=float(lg_cfg.get("weight_decay", 0.01)))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - warmup, 1))
    wsched = optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=warmup) if warmup else None

    start_epoch, best_val = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"], strict=False)
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("val_loss", float("inf"))
        if "optimizer_state_dict" in ck:
            opt.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            sched.load_state_dict(ck["scheduler_state_dict"])
        print(f"  resumed from epoch {start_epoch}")

    tr, va = make_loaders(config, sc.get("batch_size", 64))
    keys = ["loss", "loss_obs", "loss_kl", "loss_contrastive",
            "loss_semantic", "loss_conf", "metric_div"]
    save_dir = os.path.join(args.output_dir, "stage1")
    os.makedirs(save_dir, exist_ok=True)
    save_every = config.get("logging", {}).get("save_every", 10)
    grad_clip = lg_cfg.get("grad_clip", 1.0)

    for ep in range(start_epoch, epochs):
        model.train()
        tracker = LossTracker(keys)
        for batch in tqdm(tr, desc=f"S1 {ep + 1}/{epochs}", leave=False):
            pc, mask, j = to_device(batch, device)
            out = model(pc, motion_joints=j, temporal_mask=mask, texts=batch["texts"])
            if torch.isnan(out["loss"]):
                continue
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tracker.update(out)
        if ep < warmup and wsched:
            wsched.step()
        else:
            sched.step()

        model.eval()
        vt = LossTracker(keys)
        with torch.no_grad():
            for batch in va:
                pc, mask, j = to_device(batch, device)
                out = model(pc, motion_joints=j, temporal_mask=mask, texts=batch["texts"])
                if not torch.isnan(out["loss"]):
                    vt.update(out)
        val_loss = vt.avg().get("loss", float("inf"))
        print(f"ep {ep + 1}: {tracker.summary()} | val_loss={val_loss:.4f}")

        state = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "val_loss": val_loss,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(save_dir, "best.pt"))
            print(f"  best val_loss={best_val:.4f}")
        if (ep + 1) % save_every == 0:
            torch.save(state, os.path.join(save_dir, f"ep{ep + 1}.pt"))

    torch.save({"model_state_dict": model.state_dict(), "epoch": epochs - 1},
               os.path.join(save_dir, "final.pt"))
    return os.path.join(save_dir, "best.pt")


# ---------------------------------------------------------- stages 2 and 3

def _build_captioner(config, device):
    model = Captioner(config)
    model.latent_graph = model.latent_graph.to(device)
    model.semantic_proj = model.semantic_proj.to(device)
    model.node_proj = model.node_proj.to(device)
    return model


def _restore_opt(ck, opt, sched):
    if "optimizer_state_dict" in ck:
        try:
            opt.load_state_dict(ck["optimizer_state_dict"])
        except Exception as e:
            print(f"  optimizer not restored: {e}")
    if "scheduler_state_dict" in ck:
        try:
            sched.load_state_dict(ck["scheduler_state_dict"])
        except Exception as e:
            print(f"  scheduler not restored: {e}")


def _caption_epochs(model, opt, sched, tr, va, device, epochs, start_epoch,
                    best_val, save_dir, caption_every, save_every, patience,
                    desc, clip_params):
    keys = ["loss", "loss_caption", "loss_obs", "loss_kl",
            "loss_contrastive", "loss_semantic", "loss_conf"]
    os.makedirs(save_dir, exist_ok=True)
    patience_left = patience
    state = {}

    for ep in range(start_epoch, epochs):
        model.train()
        tracker = LossTracker(keys)
        for batch in tqdm(tr, desc=f"{desc} {ep + 1}/{epochs}", leave=False):
            pc, mask, j = to_device(batch, device)
            out = model(pc, mask, batch["texts"], motion_joints=j)
            if torch.isnan(out["loss"]):
                continue
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            tracker.update(out)
        sched.step()

        model.eval()
        vt = LossTracker(keys)
        with torch.no_grad():
            for batch in va:
                pc, mask, j = to_device(batch, device)
                out = model(pc, mask, batch["texts"], motion_joints=j)
                if not torch.isnan(out["loss"]):
                    vt.update(out)
        v = vt.avg()
        val_cap = v.get("loss_caption", v.get("loss", float("inf")))
        print(f"ep {ep + 1}: {tracker.summary()} | val={v.get('loss', 0):.4f} "
              f"val_caption={val_cap:.4f}")
        if (ep + 1) % caption_every == 0:
            preview_captions(model, va, device)

        state = captioner_state(model, ep, opt, sched, val_cap, best_val)
        lora_latest = os.path.join(save_dir, "qwen_lora_latest")
        model.qwen.save_pretrained(lora_latest)
        state["qwen_lora_path"] = lora_latest

        if val_cap < best_val:
            best_val = val_cap
            state["best_val"] = best_val
            lora_best = os.path.join(save_dir, "qwen_lora")
            model.qwen.save_pretrained(lora_best)
            state["qwen_lora_path"] = lora_best
            torch.save(state, os.path.join(save_dir, "best.pt"))
            print(f"  best val_caption={best_val:.4f}")
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {ep + 1}")
                break
        if (ep + 1) % save_every == 0:
            torch.save(state, os.path.join(save_dir, f"ep{ep + 1}.pt"))

    torch.save(state, os.path.join(save_dir, "final.pt"))
    return os.path.join(save_dir, "best.pt")


def train_stage2(config, args, lg_ckpt=None):
    sc = stage_cfg(config, "stage2")
    device = safe_device(sc.get("device", args.device))
    print("\n== Stage 2: captioner, latent state frozen")
    model = _build_captioner(config, device)

    start_epoch, best_val, ck = 0, float("inf"), None
    if args.resume and os.path.exists(args.resume):
        ck = load_captioner_checkpoint(model, args.resume, device, trainable=True)
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("val_caption", float("inf"))
        print(f"  resumed from epoch {start_epoch}")
    else:
        lg_ckpt = lg_ckpt or args.lg_ckpt
        if lg_ckpt and os.path.exists(lg_ckpt):
            load_captioner_checkpoint(model, lg_ckpt, device)
            print(f"  latent graph from {lg_ckpt}")
        else:
            print("  no stage 1 checkpoint given, latent graph starts from scratch")

    model.freeze_latent_graph()
    params = model.trainable_parameters()
    print(f"  trainable parameters: {sum(p.numel() for p in params):,}")

    epochs = sc.get("epochs", 50)
    opt = optim.AdamW(params, lr=float(sc.get("lr", 1e-4)),
                      weight_decay=float(sc.get("weight_decay", 0.01)))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - start_epoch, 1))
    if ck is not None:
        _restore_opt(ck, opt, sched)

    tr, va = make_loaders(config, sc.get("batch_size", 32))
    return _caption_epochs(
        model, opt, sched, tr, va, device, epochs, start_epoch, best_val,
        os.path.join(args.output_dir, "stage2"),
        sc.get("caption_every", 5),
        config.get("logging", {}).get("save_every", 10),
        sc.get("patience", 100), "S2", params)


def train_stage3(config, args, caption_ckpt=None):
    sc = stage_cfg(config, "stage3")
    device = safe_device(sc.get("device", args.device))
    print("\n== Stage 3: joint fine-tuning")
    model = _build_captioner(config, device)

    start_epoch, best_val, ck = 0, float("inf"), None
    if args.resume and os.path.exists(args.resume):
        ck = load_captioner_checkpoint(model, args.resume, device, trainable=True)
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("val_caption", float("inf"))
        print(f"  resumed from epoch {start_epoch}")
    else:
        caption_ckpt = caption_ckpt or args.caption_ckpt
        if caption_ckpt and os.path.exists(caption_ckpt):
            load_captioner_checkpoint(model, caption_ckpt, device, trainable=True)
            print(f"  stage 2 weights from {caption_ckpt}")
        else:
            print("  no stage 2 checkpoint given, training from scratch")

    model.unfreeze_latent_graph()
    epochs = sc.get("epochs", 30)
    opt = optim.AdamW([
        {"params": model.latent_graph.parameters(),
         "lr": float(sc.get("lg_lr", 1e-5))},
        {"params": list(model.semantic_proj.parameters()) + list(model.node_proj.parameters()),
         "lr": float(sc.get("proj_lr", 5e-5))},
        {"params": [p for p in model.qwen.parameters() if p.requires_grad],
         "lr": float(sc.get("qwen_lr", 3e-5))},
    ], weight_decay=float(sc.get("weight_decay", 0.01)))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs - start_epoch, 1))
    if ck is not None:
        _restore_opt(ck, opt, sched)

    tr, va = make_loaders(config, sc.get("batch_size", 16))
    return _caption_epochs(
        model, opt, sched, tr, va, device, epochs, start_epoch, best_val,
        os.path.join(args.output_dir, "stage3"),
        sc.get("caption_every", 3),
        config.get("logging", {}).get("save_every", 10),
        sc.get("patience", 10 ** 9), "S3", list(model.parameters()))


# ----------------------------------------------------------------- test

def run_test(config, args, ckpt_path=None):
    test_dir = args.test_dir or config.get("logging", {}).get("test_dir", "./test/hirad")
    os.makedirs(test_dir, exist_ok=True)

    if ckpt_path is None:
        for cand in ["stage3/best.pt", "stage3/final.pt", "stage2/best.pt", "stage2/final.pt"]:
            p = os.path.join(args.output_dir, cand)
            if os.path.exists(p):
                ckpt_path = p
                break
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("[test] no checkpoint found")
        return

    loader = make_test_loader(config)
    if loader is None:
        print("[test] no test split")
        return

    device = safe_device(args.device)
    print(f"\n== Test split inference\n  checkpoint: {ckpt_path}\n  output: {test_dir}")
    model = _build_captioner(config, device)
    load_captioner_checkpoint(model, ckpt_path, device, lora_path=args.lora_path)
    model.eval()

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            pc, mask, j = to_device(batch, device)
            caps = model.generate(pc, mask, motion_joints=j)
            results.extend({"ground_truth": gt, "prediction": pr}
                           for gt, pr in zip(batch["texts"], caps))

    out_path = os.path.join(test_dir, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(test_dir, "source_ckpt.txt"), "w") as f:
        f.write(ckpt_path + "\n")
    print(f"[test] {len(results)} predictions written to {out_path}")


# ----------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["pretrain_lg", "caption", "joint", "test", "all"])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output_dir", default=None, help="overrides logging.save_dir")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lg_ckpt", default=None, help="stage 1 checkpoint for --stage caption")
    p.add_argument("--caption_ckpt", default=None,
                   help="stage 2 checkpoint for --stage joint, or the checkpoint to evaluate for --stage test")
    p.add_argument("--resume", default=None, help="resume the given stage from this checkpoint")
    p.add_argument("--lora_path", default=None, help="--stage test: override the adapter directory")
    p.add_argument("--test_dir", default=None, help="--stage test: override logging.test_dir")
    return p.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.output_dir is None:
        args.output_dir = config.get("logging", {}).get("save_dir", "./checkpoints/hirad")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.stage == "pretrain_lg":
        train_stage1(config, args)
    elif args.stage == "caption":
        train_stage2(config, args)
    elif args.stage == "joint":
        train_stage3(config, args)
        run_test(config, args)
    elif args.stage == "test":
        run_test(config, args, ckpt_path=args.caption_ckpt)
    elif args.stage == "all":
        lg = train_stage1(config, args)
        args.resume = None
        cap = train_stage2(config, args, lg_ckpt=lg)
        train_stage3(config, args, caption_ckpt=cap)
        run_test(config, args)


if __name__ == "__main__":
    main()
