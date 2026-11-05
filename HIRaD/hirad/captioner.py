"""Structured-prefix conditioning of a frozen language model."""

import torch
import torch.nn as nn

from .latent_graph import LatentGraph


def load_qwen_4bit(path, gpu_id=0):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    return AutoModelForCausalLM.from_pretrained(
        path, device_map={"": gpu_id}, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True))


def setup_qlora(model, r=4, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[QLoRA] trainable {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


class Captioner(nn.Module):
    """Prefix = [prompt tokens; q; r^(1), ..., r^(M)].

    q is projected from the semantic bottleneck z_sem and the node tokens
    r^(m) from the node summaries. Node tokens are randomly dropped during
    training (node_drop_rate) and always kept at inference.
    """

    def __init__(self, config):
        super().__init__()
        lg_cfg = config.get("latent_graph", {})
        cap_cfg = config.get("captioner", {})
        from transformers import AutoTokenizer

        node_dim = lg_cfg.get("node_dim", 256)
        embed_dim = lg_cfg.get("world", {}).get("embed_dim", 256)

        self.latent_graph = LatentGraph(config)

        device_str = cap_cfg.get("device", "cuda:0")
        gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
        if torch.cuda.is_available() and gpu_id >= torch.cuda.device_count():
            gpu_id = 0

        qwen_path = cap_cfg.get("qwen_path", "Qwen/Qwen3-8B")
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path, padding_side="right", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cap_cfg.get("use_qlora", True):
            self.qwen = load_qwen_4bit(qwen_path, gpu_id)
            self.qwen = setup_qlora(
                self.qwen,
                r=cap_cfg.get("lora_r", 4),
                alpha=cap_cfg.get("lora_alpha", 32),
                dropout=cap_cfg.get("lora_dropout", 0.05))
        else:
            from transformers import AutoModelForCausalLM
            self.qwen = AutoModelForCausalLM.from_pretrained(
                qwen_path, dtype=torch.float16,
                device_map={"": gpu_id}, trust_remote_code=True)

        qd = self.qwen.config.hidden_size
        self.semantic_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, qd))
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_dim), nn.Linear(node_dim, qd),
            nn.GELU(), nn.Linear(qd, qd))

        self.prompt = cap_cfg.get("prompt", "Describe the motion:")
        self.lambda_caption = cap_cfg.get("lambda_caption", 1.0)
        self.node_drop_rate = cap_cfg.get("node_drop_rate", 0.1)

        gen = cap_cfg.get("generation", {})
        self.gen_kwargs = dict(
            max_new_tokens=gen.get("max_new_tokens", 48),
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.9),
            do_sample=gen.get("do_sample", True),
            repetition_penalty=gen.get("repetition_penalty", 1.3),
            no_repeat_ngram_size=gen.get("no_repeat_ngram_size", 4),
        )

    def build_prefix(self, lg_out, temporal_mask=None):
        B = lg_out["node_emb"].shape[0]
        device = lg_out["node_emb"].device

        prompt_enc = self.tokenizer(
            [self.prompt] * B, return_tensors="pt", padding=True,
            add_special_tokens=False, truncation=True).to(device)
        with torch.no_grad():
            prompt_emb = self.qwen.get_input_embeddings()(prompt_enc.input_ids)
        prompt_mask = prompt_enc.attention_mask

        z_sem = lg_out["z_sem"]
        if z_sem.dim() == 3:
            z_sem = z_sem.mean(dim=1)
        sem_token = self.semantic_proj(z_sem).unsqueeze(1)
        body_tokens = self.node_proj(lg_out["node_emb"])

        if self.training and self.node_drop_rate > 0:
            M = body_tokens.shape[1]
            keep = torch.bernoulli(
                torch.full((B, M, 1), 1 - self.node_drop_rate, device=device))
            body_tokens = body_tokens * keep / (1 - self.node_drop_rate)

        prefix = torch.cat([prompt_emb, sem_token, body_tokens], dim=1)
        K = sem_token.shape[1] + body_tokens.shape[1]
        sensor_mask = torch.ones(B, K, device=device, dtype=prompt_mask.dtype)
        prefix_mask = torch.cat([prompt_mask, sensor_mask], dim=1)
        return prefix, prefix_mask

    def forward(self, point_cloud, temporal_mask, texts, motion_joints=None):
        device = point_cloud.device
        lg_out = self.latent_graph(
            point_cloud, motion_joints, temporal_mask, texts=texts)
        prefix, prefix_mask = self.build_prefix(lg_out, temporal_mask)
        B, P, _ = prefix.shape

        enc = self.tokenizer(
            [t + self.tokenizer.eos_token for t in texts],
            padding=True, truncation=True, max_length=64,
            return_tensors="pt").to(device)
        with torch.no_grad():
            text_emb = self.qwen.get_input_embeddings()(enc.input_ids)

        inputs_embeds = torch.cat([prefix.to(text_emb.dtype), text_emb], dim=1)
        attn_mask = torch.cat(
            [prefix_mask.to(enc.attention_mask.dtype), enc.attention_mask], dim=1)

        text_labels = enc.input_ids.clone()
        text_labels[enc.attention_mask == 0] = -100
        labels = torch.cat([
            torch.full((B, P), -100, device=device, dtype=enc.input_ids.dtype),
            text_labels], dim=1)

        qwen_out = self.qwen(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask, labels=labels)
        loss = lg_out["loss"] + self.lambda_caption * qwen_out.loss

        return {
            "loss": loss,
            "loss_obs": lg_out["loss_obs"],
            "loss_kl": lg_out["loss_kl"],
            "loss_contrastive": lg_out["loss_contrastive"],
            "loss_semantic": lg_out["loss_semantic"],
            "loss_conf": lg_out["loss_conf"],
            "loss_caption": qwen_out.loss,
            "metric_div": lg_out["metric_div"],
        }

    @torch.no_grad()
    def generate_from_prefix(self, lg_out, temporal_mask=None, **overrides):
        prefix, prefix_mask = self.build_prefix(lg_out, temporal_mask)
        embed_weight = self.qwen.get_input_embeddings().weight
        prefix = prefix.to(dtype=embed_weight.dtype, device=embed_weight.device)
        kwargs = dict(self.gen_kwargs)
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        outputs = self.qwen.generate(
            inputs_embeds=prefix,
            attention_mask=prefix_mask.to(dtype=torch.long, device=embed_weight.device),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, point_cloud, temporal_mask, motion_joints=None, **overrides):
        self.eval()
        lg_out = self.latent_graph.encode_full(point_cloud, temporal_mask)
        return self.generate_from_prefix(lg_out, temporal_mask, **overrides)

    def freeze_latent_graph(self):
        """Stage 2: freeze the latent state, keep the semantic pathway trainable."""
        for p in self.latent_graph.parameters():
            p.requires_grad = False
        for mod in (self.latent_graph.shared_dynamics,
                    self.latent_graph.semantic_head):
            for p in mod.parameters():
                p.requires_grad = True

    def unfreeze_latent_graph(self):
        for p in self.latent_graph.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
