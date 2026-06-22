"""Pi3 / Pi3X as a drop-in replacement for the DA3 inference call.

Why bother: the runtime budget (T <= 0.0198 s/frame, 0.0028 spare at v008) is
what has vetoed every rotation fix we measured -- rotation averaging lowers
RPE_rot by 19% but its +0.006 s/f raises W_t enough to lose more score than it
gains. A cheaper backbone buys that budget back. Pi3 reports 57.4 FPS against
VGGT's 43.2 while also scoring better on Sintel rotation (RPE_rot 0.282), so it
is the one candidate that moves both constraints in the right direction.

Pi3X additionally accepts `intrinsics=`. That matters here: DA3 ignores the
intrinsics argument unless extrinsics are supplied too, and self-estimates a
focal 1.6-2.5x the true one on our rectified frames -- a focal error does not
merely rescale, it shears rotation into translation, which is exactly the
signature we measured (pitch/yaw error 1.2-4.4x the roll error). We already know
the true pinhole K after kb4 rectification, so being able to hand it over
directly replaces the `DA3_RECT_FSCALE` workaround with the real thing.

The adapter presents DA3's `.inference()` contract (extrinsics w2c, depth, conf,
intrinsics, processed_images) so predict.py's chaining, map building and
submission writers are untouched.
"""
import os
from types import SimpleNamespace

import numpy as np
import torch

PIXEL_LIMIT = int(os.environ.get("PI3_PIXEL_LIMIT", 255000))
PATCH = 14


def _target_hw(h, w, limit):
    """Pi3's own sizing rule: keep aspect, land on a multiple of the patch size."""
    import math
    s = math.sqrt(limit / max(h * w, 1))
    k, m = max(1, round(w * s / PATCH)), max(1, round(h * s / PATCH))
    while (k * PATCH) * (m * PATCH) > limit and (k > 1 or m > 1):
        if (k / max(m, 1)) > (w / max(h, 1)):
            k -= 1
        else:
            m -= 1
    return m * PATCH, k * PATCH


class Pi3Backend:
    def __init__(self, model_id=None, device="cuda", log=print):
        model_id = model_id or os.environ.get("PI3_MODEL", "yyfz233/Pi3X")
        self.is_x = "Pi3X" in model_id
        if self.is_x:
            from pi3.models.pi3x import Pi3X as _M
        else:
            from pi3.models.pi3 import Pi3 as _M
        self.model = _M.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.log = log
        # bf16 is emulated (and 4x slower) on Turing; the eval host is Blackwell.
        want = os.environ.get("DA3_PRECISION", "auto")
        if want == "fp16":
            self.dtype = torch.float16
        elif want == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
                          else torch.float16)
        log(f"Pi3 backend: {model_id} dtype={self.dtype}")

    def inference(self, frames, process_res=None, export_format=None, K=None):
        """frames: list of HxWx3 uint8 RGB. Returns a DA3-shaped result."""
        h0, w0 = frames[0].shape[:2]
        th, tw = _target_hw(h0, w0, PIXEL_LIMIT)
        import cv2
        batch = np.stack([cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
                          for f in frames])
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255).to(self.device)

        kwargs = {}
        if self.is_x and K is not None:
            Ks = np.tile(np.asarray(K, np.float32)[None], (len(frames), 1, 1)).copy()
            Ks[:, 0, :] *= tw / float(w0)          # rescale to the fed resolution
            Ks[:, 1, :] *= th / float(h0)
            kwargs["intrinsics"] = torch.from_numpy(Ks).to(self.device)[None]

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            res = self.model(t[None], **kwargs)

        c2w = res["camera_poses"][0].float()                     # (N,4,4) camera->world
        w2c = torch.linalg.inv(c2w).cpu().numpy()
        lp = res["local_points"][0].float().cpu().numpy()        # (N,H,W,3)
        depth = lp[..., 2].copy()
        conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()

        # Pi3 is a pinhole model, so the intrinsics are exact from the local
        # points: u = fx * X/Z + cx. Solve fx, cx by least squares on the first
        # view's valid pixels rather than assuming a principal point.
        intr = np.tile(self._fit_K(lp[0], th, tw)[None], (len(frames), 1, 1))
        return SimpleNamespace(extrinsics=w2c, depth=depth, conf=conf,
                               intrinsics=intr, processed_images=batch)

    @staticmethod
    def _fit_K(lp, h, w):
        Z = lp[..., 2]
        m = np.isfinite(Z) & (Z > 1e-6)
        v, u = np.nonzero(m)
        x, y = lp[..., 0][m] / Z[m], lp[..., 1][m] / Z[m]
        K = np.eye(3)
        for ax, coord, px in ((0, x, u), (1, y, v)):
            A = np.stack([coord, np.ones_like(coord)], 1)
            f, c = np.linalg.lstsq(A, px.astype(np.float64), rcond=None)[0]
            K[ax, ax], K[ax, 2] = f, c
        return K
