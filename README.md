# HIRaD: Hidden Interaction Inference from Predictive Radar Dynamics

Code for the NeurIPS 2026 paper. HIRaD captions human–object interactions
from sparse mmWave radar point clouds without detecting the object. It keeps a
predictive node-structured latent state, reads the corrective residual between
a motion-continuation prior and an observation-conditioned posterior,
compresses the residual history into a language-aligned semantic token, and
conditions a frozen Qwen3-8B decoder through a structured prefix.

Project page: https://dlwowlb.github.io/HIRaD

This repository contains the model, the three-stage training script and the
inference scripts. The radar simulation and caption generation pipeline is
described in Appendix G of the paper and is not part of this release.

## Layout

```
hirad/
  layers.py        point-to-node routing, transition, gated posterior update, decoders, readout
  motion.py        body-motion encoder used for the auxiliary contrastive loss
  semantic.py      deviation dynamics encoder and semantic bottleneck
  latent_graph.py  LatentGraph: latent state, residual trajectories, losses
  captioner.py     Captioner: structured prefix on the frozen language model
  data.py          dataset, splits, loaders
  augment.py       training-time point-cloud augmentation
  checkpoint.py    checkpoint and adapter loading
train.py           stage 1 / 2 / 3 training and test-split inference
scripts/
  infer_real.py        caption real radar recordings
  prefix_ablation.py   inference-time prefix ablations (zero / shuffle)
configs/default.yaml   hyperparameters used in the paper
```

## Setup

```bash
pip install -r requirements.txt
```

Set `captioner.qwen_path` in the config to a Hugging Face model id or a local
copy of Qwen3-8B. The decoder is loaded in 4-bit and trained with QLoRA
adapters on the attention projections.

## Data

Each sample is an `.npz` file:

| key | shape | note |
|---|---|---|
| `point_cloud` | `(T, P, D)` float32 | `D >= 4`, channels `[lateral, height, depth, doppler, ...]`, zero rows are empty slots |
| `text` | str | caption |
| `motion_joints` | `(T_m, 22, 3)` float32 | optional, paired body motion used only for the contrastive loss during training |

Point `dataset.train_dir`, `val_dir` and `test_dir` at the three split
directories. If all three are the same directory the files are split
deterministically with `train_val_test_split` and `split_seed`.

## Training

```bash
python train.py --stage pretrain_lg --config configs/default.yaml
python train.py --stage caption     --lg_ckpt      checkpoints/hirad/stage1/best.pt
python train.py --stage joint       --caption_ckpt checkpoints/hirad/stage2/best.pt
```

`--stage all` runs the three stages in sequence. `--stage joint` and
`--stage all` finish by writing test-split predictions to `logging.test_dir`.
`--resume <ckpt>` continues an interrupted stage. Checkpoints of stages 2 and
3 store the QLoRA adapter in a `qwen_lora/` directory next to the `.pt` file.

Stage 1 trains the latent graph with the reconstruction, KL, radar–motion
contrastive and semantic alignment losses. Stage 2 freezes the latent state
and trains the semantic pathway, the prefix projections and the adapters with
the captioning loss. Stage 3 fine-tunes all trainable modules jointly.

## Evaluation

Test split:

```bash
python train.py --stage test --caption_ckpt checkpoints/hirad/stage3/best.pt
```

Real recordings (`.npz` with `point_cloud` and optionally `num_radar_frames`):

```bash
python scripts/infer_real.py --checkpoint checkpoints/hirad/stage3/best.pt \
    --input <npz file or directory> --output predictions_real.json
```

Prefix ablations at inference time (semantic token or node tokens zeroed,
or prefixes permuted across clips):

```bash
python scripts/prefix_ablation.py --mode zero_zsem --checkpoint checkpoints/hirad/stage3/best.pt
```

The input of the semantic pathway can be switched with `ablation.mode` in the
config. `no_residue_strict` feeds the prior trajectories and `post_mu_input`
feeds the non-subtracted posterior trajectories, which are the controls
reported in the paper. These variants are retrained from scratch under the
same protocol.

## Citation

```bibtex
@inproceedings{lee2026hirad,
  title     = {{HIRaD}: Hidden Interaction Inference from Predictive Radar Dynamics},
  author    = {Lee, Jaekwon and Lee, Jaekoo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
