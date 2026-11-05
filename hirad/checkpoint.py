"""Checkpoint helpers.

Stage 1 saves a single `model_state_dict` of the latent graph. Stages 2 and 3
save the trainable parts under separate keys and keep the QLoRA adapter in a
PEFT directory next to the checkpoint:

    latent_graph_state, shared_dynamics_state, semantic_head_state,
    semantic_proj_state, node_proj_state, qwen_lora_path
"""

import os

import torch


def load_lora(qwen, lora_path, trainable=False):
    from peft import PeftModel
    if isinstance(qwen, PeftModel):
        if trainable:
            if "default" in qwen.peft_config:
                qwen.delete_adapter("default")
            qwen.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            qwen.set_adapter("default")
        else:
            qwen.load_adapter(lora_path, adapter_name="trained")
            qwen.set_adapter("trained")
        return qwen
    return PeftModel.from_pretrained(qwen, lora_path, is_trainable=trainable)


def resolve_lora_path(ck, ckpt_path, override=None):
    candidates = []
    if override:
        candidates.append(override)
    else:
        if ck.get("qwen_lora_path"):
            candidates.append(ck["qwen_lora_path"])
        d = os.path.dirname(ckpt_path)
        candidates += [os.path.join(d, "qwen_lora"), os.path.join(d, "qwen_lora_latest")]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_captioner_checkpoint(model, ckpt_path, device, lora_path=None,
                              trainable=False):
    """Loads a stage 2/3 checkpoint (or a stage 1 latent-graph checkpoint) into `model`."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "latent_graph_state" not in ck:
        state = ck.get("model_state_dict", ck)
        model.latent_graph.load_state_dict(state, strict=False)
        return ck

    model.latent_graph.load_state_dict(ck["latent_graph_state"], strict=False)
    for key, module in [
        ("shared_dynamics_state", model.latent_graph.shared_dynamics),
        ("semantic_head_state", model.latent_graph.semantic_head),
        ("semantic_proj_state", model.semantic_proj),
        ("node_proj_state", model.node_proj),
    ]:
        if key in ck:
            module.load_state_dict(ck[key])

    lp = resolve_lora_path(ck, ckpt_path, lora_path)
    if lp is None:
        print(f"[ckpt] no LoRA adapter found next to {ckpt_path}")
    else:
        model.qwen = load_lora(model.qwen, lp, trainable=trainable)
        print(f"[ckpt] LoRA adapter: {lp}")
    return ck


def captioner_state(model, epoch, opt, sched, val_caption, best_val):
    return {
        "epoch": epoch,
        "latent_graph_state": model.latent_graph.state_dict(),
        "shared_dynamics_state": model.latent_graph.shared_dynamics.state_dict(),
        "semantic_head_state": model.latent_graph.semantic_head.state_dict(),
        "semantic_proj_state": model.semantic_proj.state_dict(),
        "node_proj_state": model.node_proj.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "val_caption": val_caption,
        "best_val": best_val,
    }
