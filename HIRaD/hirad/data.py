"""Dataset, collate, splits and loaders.

Each sample is an .npz file with
    point_cloud   (T, P, D) float32, D >= 4, channels [lateral, height, depth, doppler, ...]
    text          str, the caption
    motion_joints (T_m, J, 3) float32, optional, paired body motion from the synthesis pipeline
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .augment import RadarAugment


class RadarCaptionDataset(Dataset):
    def __init__(self, paths, max_radar_frames=100, max_motion_frames=300,
                 points_per_frame=128, point_dims=4, load_joints=True):
        self.samples = [Path(p) for p in paths]
        self.mrf = max_radar_frames
        self.mmf = max_motion_frames
        self.ppf = points_per_frame
        self.pd = point_dims
        self.load_joints = load_joints

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)

        pc = data["point_cloud"].astype(np.float32)
        T_r = min(pc.shape[0], self.mrf)
        D = min(pc.shape[-1], self.pd)
        pc_p = np.zeros((self.mrf, self.ppf, self.pd), dtype=np.float32)
        P = min(pc.shape[1], self.ppf)
        pc_p[:T_r, :P, :D] = pc[:T_r, :P, :D]
        mask = np.zeros(self.mrf, dtype=np.bool_)
        mask[:T_r] = True

        joints = None
        if self.load_joints and "motion_joints" in data:
            j = data["motion_joints"].astype(np.float32)
            if j.ndim == 3 and j.shape[1] >= 22:
                T_j = min(j.shape[0], self.mmf)
                jp = np.zeros((self.mmf, 22, 3), dtype=np.float32)
                jp[:T_j] = j[:T_j, :22, :3]
                joints = torch.from_numpy(jp)

        return {
            "point_cloud": torch.from_numpy(pc_p),
            "radar_mask": torch.from_numpy(mask),
            "motion_joints": joints,
            "text": str(data["text"]),
        }


class AugmentedDataset(Dataset):
    def __init__(self, base, augment=None):
        self.base = base
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.augment is not None:
            item["point_cloud"] = torch.from_numpy(
                self.augment(item["point_cloud"].numpy()))
        return item


def collate_fn(batch):
    has = [b["motion_joints"] is not None for b in batch]
    if all(has):
        joints = torch.stack([b["motion_joints"] for b in batch])
    elif any(has):
        T = max(b["motion_joints"].shape[0] for b in batch if b["motion_joints"] is not None)
        joints = torch.stack([
            b["motion_joints"] if b["motion_joints"] is not None
            else torch.zeros(T, 22, 3) for b in batch])
    else:
        joints = None
    return {
        "point_cloud": torch.stack([b["point_cloud"] for b in batch]),
        "radar_mask": torch.stack([b["radar_mask"] for b in batch]),
        "motion_joints": joints,
        "texts": [b["text"] for b in batch],
    }


def _list_npz(d):
    d = Path(d)
    files = sorted(d.glob("*.npz"))
    if not files:
        files = sorted(d.rglob("sample_*.npz"))
    return files


def split_samples(config):
    """Returns (train, val, test) file lists.

    If train_dir, val_dir and test_dir are three different directories they
    are used as they are. If they all point to the same directory, the files
    are shuffled with `split_seed` and split by `train_val_test_split`.
    """
    ds = config["dataset"]
    train_dir = ds["train_dir"]
    val_dir = ds.get("val_dir", train_dir)
    test_dir = ds.get("test_dir", train_dir)

    if train_dir == val_dir == test_dir:
        paths = _list_npz(train_dir)
        seed = ds.get("split_seed", 42)
        if seed is not None:
            random.Random(seed).shuffle(paths)
        r = [float(x) for x in ds.get("train_val_test_split", [0.8, 0.1, 0.1])]
        r = [x / sum(r) for x in r]
        n = len(paths)
        n_tr, n_va = int(n * r[0]), int(n * r[1])
        return paths[:n_tr], paths[n_tr:n_tr + n_va], paths[n_tr + n_va:]

    return _list_npz(train_dir), _list_npz(val_dir), _list_npz(test_dir)


def _dataset_kwargs(config):
    ds = config["dataset"]
    lg = config.get("latent_graph", {})
    return dict(
        max_radar_frames=ds.get("max_seq_len", 100),
        max_motion_frames=ds.get("max_motion_frames", 300),
        points_per_frame=ds.get("points_per_frame", lg.get("num_points", 128)),
        point_dims=lg.get("point_in_dim", 4),
        load_joints=lg.get("use_motion_encoder", True),
    )


def make_loaders(config, batch_size):
    """Train loader (with augmentation) and validation loader."""
    tr_paths, va_paths, _ = split_samples(config)
    kw = _dataset_kwargs(config)
    aug = RadarAugment.from_config(config.get("augment"))
    tr = AugmentedDataset(RadarCaptionDataset(tr_paths, **kw), aug)
    va = AugmentedDataset(RadarCaptionDataset(va_paths, **kw), None)
    print(f"[data] train {len(tr)}  val {len(va)}  "
          f"augmentation {'on' if aug else 'off'}")
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True,
                   collate_fn=collate_fn, num_workers=4, pin_memory=True),
        DataLoader(va, batch_size=batch_size, shuffle=False,
                   collate_fn=collate_fn, num_workers=2, pin_memory=True),
    )


def make_test_loader(config, batch_size=8):
    _, _, te_paths = split_samples(config)
    if not te_paths:
        return None
    te = RadarCaptionDataset(te_paths, **_dataset_kwargs(config))
    return DataLoader(te, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn, num_workers=2, pin_memory=True)
