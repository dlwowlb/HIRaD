"""On-the-fly point-cloud augmentation applied to the training split only.

Input and output are (T, N, D) arrays with channel order
[lateral, height, depth, doppler, (snr)]. Zero rows are empty slots and a
point is valid when its xyz norm is above 1e-6.
"""

import numpy as np


class RadarAugment:
    def __init__(
        self,
        p_doppler_scale=0.9, doppler_scale_range=(0.2, 1.0),
        p_doppler_jitter=0.8, doppler_sigma=0.05,
        p_ghost=0.5, ghost_rate=0.10,
        ghost_depth_shift=(1.0, 2.5), ghost_doppler_atten=(0.3, 0.7),
        p_upper_body_bias=0.3, upper_body_drop=0.3, lower_threshold=0.3,
        p_spatial_jitter=0.9, spatial_sigma=0.03,
        p_dropout=0.8, dropout_rate=0.2,
        p_frame_dropout=0.2, frame_dropout_rate=0.05,
        p_depth_shift=0.3, depth_shift_sigma=0.1,
        p_azimuth_rot=0.3, azimuth_rot_deg=20.0,
        always_shuffle=True, rng_seed=None,
    ):
        self.p_doppler_scale = p_doppler_scale
        self.doppler_scale_range = doppler_scale_range
        self.p_doppler_jitter = p_doppler_jitter
        self.doppler_sigma = doppler_sigma
        self.p_ghost = p_ghost
        self.ghost_rate = ghost_rate
        self.ghost_depth_shift = ghost_depth_shift
        self.ghost_doppler_atten = ghost_doppler_atten
        self.p_upper_body_bias = p_upper_body_bias
        self.upper_body_drop = upper_body_drop
        self.lower_threshold = lower_threshold
        self.p_spatial_jitter = p_spatial_jitter
        self.spatial_sigma = spatial_sigma
        self.p_dropout = p_dropout
        self.dropout_rate = dropout_rate
        self.p_frame_dropout = p_frame_dropout
        self.frame_dropout_rate = frame_dropout_rate
        self.p_depth_shift = p_depth_shift
        self.depth_shift_sigma = depth_shift_sigma
        self.p_azimuth_rot = p_azimuth_rot
        self.azimuth_rot_deg = azimuth_rot_deg
        self.always_shuffle = always_shuffle
        self.rng = np.random.default_rng(rng_seed)

    @classmethod
    def from_config(cls, aug_cfg):
        if not aug_cfg or not aug_cfg.get("enabled", False):
            return None
        keys = [
            "p_doppler_scale", "doppler_scale_range", "p_doppler_jitter",
            "doppler_sigma", "p_ghost", "ghost_rate", "ghost_depth_shift",
            "ghost_doppler_atten", "p_upper_body_bias", "upper_body_drop",
            "lower_threshold", "p_spatial_jitter", "spatial_sigma",
            "p_dropout", "dropout_rate", "p_frame_dropout",
            "frame_dropout_rate", "p_depth_shift", "depth_shift_sigma",
            "p_azimuth_rot", "azimuth_rot_deg", "always_shuffle",
        ]
        kwargs = {k: aug_cfg[k] for k in keys if k in aug_cfg}
        for k in ("doppler_scale_range", "ghost_depth_shift", "ghost_doppler_atten"):
            if k in kwargs:
                kwargs[k] = tuple(kwargs[k])
        return cls(**kwargs)

    @staticmethod
    def _valid_mask(pc):
        return np.linalg.norm(pc[..., :3], axis=-1) > 1e-6

    def __call__(self, pc):
        pc = pc.astype(np.float32, copy=True)
        D = pc.shape[-1]
        valid = self._valid_mask(pc)

        # whole-clip azimuth rotation
        if D >= 3 and self.rng.random() < self.p_azimuth_rot:
            ang = np.deg2rad(self.rng.uniform(-self.azimuth_rot_deg, self.azimuth_rot_deg))
            c, s = np.cos(ang), np.sin(ang)
            x, z = pc[..., 0].copy(), pc[..., 2].copy()
            pc[..., 0] = np.where(valid, c * x + s * z, pc[..., 0])
            pc[..., 2] = np.where(valid, -s * x + c * z, pc[..., 2])

        # subject distance
        if D >= 3 and self.rng.random() < self.p_depth_shift:
            shift = self.rng.normal(0, self.depth_shift_sigma)
            pc[..., 2] = np.where(valid, pc[..., 2] + shift, pc[..., 2])

        # range/angle resolution noise
        if D >= 3 and self.rng.random() < self.p_spatial_jitter:
            noise = self.rng.normal(0, self.spatial_sigma, size=pc[..., :3].shape).astype(np.float32)
            pc[..., :3] = np.where(valid[..., None], pc[..., :3] + noise, pc[..., :3])

        # doppler scale and quantization jitter
        if D >= 4 and self.rng.random() < self.p_doppler_scale:
            scale = self.rng.uniform(*self.doppler_scale_range)
            pc[..., 3] = np.where(valid, pc[..., 3] * scale, pc[..., 3])
        if D >= 4 and self.rng.random() < self.p_doppler_jitter:
            dn = self.rng.normal(0, self.doppler_sigma, size=pc[..., 3].shape).astype(np.float32)
            pc[..., 3] = np.where(valid, pc[..., 3] + dn, pc[..., 3])

        # lower-body point loss
        if D >= 2 and self.rng.random() < self.p_upper_body_bias:
            for t in range(pc.shape[0]):
                v = valid[t]
                if v.sum() == 0:
                    continue
                h_mean = float(pc[t, v, 1].mean())
                low = (pc[t, :, 1] < h_mean - self.lower_threshold) & v
                drop = self.rng.random(low.shape) < self.upper_body_drop
                pc[t, low & drop] = 0.0
            valid = self._valid_mask(pc)

        # CFAR misses
        if self.rng.random() < self.p_dropout:
            drop_mask = (self.rng.random(valid.shape) < self.dropout_rate) & valid
            pc[drop_mask] = 0.0
            valid = self._valid_mask(pc)

        # multipath ghosts
        if D >= 3 and self.rng.random() < self.p_ghost:
            for t in range(pc.shape[0]):
                v = valid[t]
                idx_valid = np.where(v)[0]
                idx_empty = np.where(~v)[0]
                if len(idx_valid) < 2 or len(idx_empty) == 0:
                    continue
                n_ghost = min(int(len(idx_valid) * self.ghost_rate), len(idx_empty))
                if n_ghost <= 0:
                    continue
                src = self.rng.choice(idx_valid, size=n_ghost, replace=False)
                dst = self.rng.choice(idx_empty, size=n_ghost, replace=False)
                ghosts = pc[t, src].copy()
                ghosts[:, 2] += self.rng.uniform(*self.ghost_depth_shift, size=n_ghost).astype(np.float32)
                if D >= 4:
                    ghosts[:, 3] *= self.rng.uniform(*self.ghost_doppler_atten, size=n_ghost).astype(np.float32)
                pc[t, dst] = ghosts
            valid = self._valid_mask(pc)

        # tracker gaps
        if self.rng.random() < self.p_frame_dropout:
            n_drop = int(pc.shape[0] * self.frame_dropout_rate)
            if n_drop > 0:
                drop_t = self.rng.choice(pc.shape[0], size=n_drop, replace=False)
                pc[drop_t] = 0.0

        # slot order invariance
        if self.always_shuffle:
            for t in range(pc.shape[0]):
                pc[t] = pc[t, self.rng.permutation(pc.shape[1])]

        return pc
