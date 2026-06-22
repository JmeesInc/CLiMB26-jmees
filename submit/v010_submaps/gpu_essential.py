"""Batched GPU essential-matrix estimation, replacing cv2.USAC_MAGSAC.

Why: the rotation-averaging tail is the only part of the pipeline that does not
get cheaper on the eval host. ALIKED and LightGlue are GPU work and shrink ~4x
there; `cv2.findEssentialMat(USAC_MAGSAC)` + `cv2.recoverPose` are single-thread
CPU and shrink by nothing. That asymmetry is what made v006/v007 pay W_t.

What is actually unexportable about RANSAC is its *adaptive stopping* -- the
sampling itself is embarrassingly parallel. So the estimator here is a fixed
hypothesis budget (branch-free, one batched eigendecomposition) followed by an
IRLS polish on the Sampson error. Both stages are dense linear algebra over a
(edges x hypotheses) batch with no data-dependent control flow, which is the
property that makes a TensorRT export possible at all.

Everything is (B, ...) batched over edges with a validity mask, so edges with
different match counts ride in the same call.
"""
import os

import numpy as np
import torch

HYPO = int(os.environ.get("DA3_DLT_HYPO", 256))
CHUNK_HYPO = int(os.environ.get("DA3_DLT_CHUNK", 32))
GNC_START = float(os.environ.get("DA3_DLT_GNC", 100.0))
_W = {}


def _dlt(x0, x1, w, f64=True):
    """Smallest right-singular vector of the 9-column epipolar design matrix.

    Solved as the smallest eigenvector of A^T A (9x9) rather than an SVD of the
    (N,9) matrix: same answer, but the cost stops depending on the match count.
    """
    A = torch.cat([x1[..., :1] * x0, x1[..., 1:2] * x0, x0], -1)   # (B,N,9)
    M = torch.einsum("bni,bn,bnj->bij", A, w, A)
    M = M + 1e-12 * torch.eye(9, device=A.device, dtype=A.dtype)
    _, V = torch.linalg.eigh(M.double() if f64 else M)
    return V[..., 0].to(A.dtype).view(-1, 3, 3)


def _sampson(E, x0, x1):
    Ex0 = torch.einsum("bij,bnj->bni", E, x0)
    Etx1 = torch.einsum("bji,bnj->bni", E, x1)
    num = (x1 * Ex0).sum(-1) ** 2
    den = Ex0[..., 0] ** 2 + Ex0[..., 1] ** 2 + Etx1[..., 0] ** 2 + Etx1[..., 1] ** 2
    return num / den.clamp(min=1e-12)


def _seed(x0, x1, mask, thr, gen):
    """Fixed-budget minimal-sample search, batched over edges AND hypotheses.

    IRLS alone cannot be trusted past ~5% contamination (measured: 1.4 deg
    median against MAGSAC's 0.14) because it is seeded from an unweighted fit
    that the outliers already own. Sampling fixes the seed.

    fp32 and chunked over hypotheses: cuSOLVER's batched workspace for 46k 9x9
    fp64 systems asks for 11.8 GB in one call, which the scored run does not
    have spare next to a GIANT backbone.
    """
    B, N, _ = x0.shape
    dev = x0.device
    n = mask.sum(1).clamp(min=8).long()
    xs0, xs1, ms = x0.float(), x1.float(), mask.float()
    thr_f = float(thr)
    best_c = torch.full((B,), float("inf"), device=dev)
    best_E = torch.zeros(B, 3, 3, device=dev)
    for lo in range(0, HYPO, max(1, CHUNK_HYPO)):
        S = min(max(1, CHUNK_HYPO), HYPO - lo)
        idx = (torch.rand(B, S, 8, device=dev, generator=gen) * n[:, None, None]).long()
        gi = idx[..., None].expand(-1, -1, -1, 3)
        g0 = torch.gather(xs0[:, None].expand(-1, S, -1, -1), 2, gi).reshape(B * S, 8, 3)
        g1 = torch.gather(xs1[:, None].expand(-1, S, -1, -1), 2, gi).reshape(B * S, 8, 3)
        Eh = _dlt(g0, g1, torch.ones(B * S, 8, device=dev), f64=False)
        r = _sampson(Eh,
                     xs0[:, None].expand(-1, S, -1, -1).reshape(B * S, N, 3),
                     xs1[:, None].expand(-1, S, -1, -1).reshape(B * S, N, 3)).view(B, S, N)
        # MSAC (truncated quadratic) -- what MAGSAC approximates without its
        # marginalisation over the noise scale.
        cost = (torch.minimum(r / thr_f, torch.ones_like(r)) * ms[:, None]).sum(-1)
        c, a = cost.min(1)
        upd = c < best_c
        best_c = torch.where(upd, c, best_c)
        best_E = torch.where(upd[:, None, None],
                             Eh.view(B, S, 3, 3)[torch.arange(B, device=dev), a], best_E)
    return best_E.to(x0.dtype)


def _skew(v):
    z = torch.zeros_like(v[..., 0])
    return torch.stack([z, -v[..., 2], v[..., 1],
                        v[..., 2], z, -v[..., 0],
                        -v[..., 1], v[..., 0], z], -1).view(*v.shape[:-1], 3, 3)


def _expm(w):
    """Rodrigues, batched."""
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    a = w / th
    K = _skew(a)
    th = th[..., None]
    return (torch.eye(3, device=w.device, dtype=w.dtype)
            + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K))


def _basis(t):
    """Two unit vectors spanning the tangent plane of the sphere at t."""
    a = torch.zeros_like(t)
    a[..., 0] = 1.0
    a = torch.where((t[..., :1].abs() > 0.9).expand_as(a),
                    torch.nn.functional.pad(torch.ones_like(t[..., :1]), (1, 1)), a)
    u = torch.cross(t, a, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    v = torch.cross(t, u, dim=-1)
    return u, v


def _sampson_signed(E, x0, x1):
    Ex0 = torch.einsum("bij,bnj->bni", E, x0)
    Etx1 = torch.einsum("bji,bnj->bni", E, x1)
    num = (x1 * Ex0).sum(-1)
    den = (Ex0[..., 0] ** 2 + Ex0[..., 1] ** 2
           + Etx1[..., 0] ** 2 + Etx1[..., 1] ** 2).clamp(min=1e-12).sqrt()
    return num / den


def _refine_manifold(R, t, x0, x1, w, iters=5, eps=1e-5, lam=1e-6):
    """Gauss-Newton on the 5 real degrees of freedom of an essential matrix.

    The DLT estimates a 9-vector and only then projects onto sigma = (1,1,0), so
    it never uses the essential constraint while fitting -- which is exactly why
    the 8-point family is statistically less efficient than a 5-point solver
    (measured: 0.80 deg against 0.29 at 0.7 px noise). Optimising E = [t]_x R
    directly in (rotvec, sphere-tangent) coordinates removes that gap while
    staying a fixed, branch-free sequence of dense ops.

    The Jacobian is finite-differenced over 5 parameters -- 5 extra residual
    evaluations per step, each O(B*N), against the 256*N the hypothesis search
    already costs, so it is free in context and avoids hand-derived derivatives.
    """
    for _ in range(iters):
        u, v = _basis(t)
        E0 = _skew(t) @ R
        r0 = _sampson_signed(E0, x0, x1)
        cols = []
        for d in range(5):
            if d < 3:
                dw = torch.zeros_like(t)
                dw[..., d] = eps
                Rp, tp = R @ _expm(dw), t
            else:
                tp = t + eps * (u if d == 3 else v)
                tp = tp / tp.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                Rp = R
            cols.append((_sampson_signed(_skew(tp) @ Rp, x0, x1) - r0) / eps)
        J = torch.stack(cols, -1)                              # (B,N,5)
        JT = J * w[..., None]
        H = torch.einsum("bnk,bnl->bkl", JT, J)
        g = torch.einsum("bnk,bn->bk", JT, r0)
        H = H + lam * torch.diagonal(H, dim1=1, dim2=2).mean(-1)[:, None, None]             * torch.eye(5, device=H.device, dtype=H.dtype) + 1e-12 * torch.eye(
                5, device=H.device, dtype=H.dtype)
        dx = -torch.linalg.solve(H, g[..., None])[..., 0]
        dx = dx.clamp(-0.1, 0.1)                               # keep GN local
        R = R @ _expm(dx[..., :3])
        t = t + dx[..., 3:4] * u + dx[..., 4:5] * v
        t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return R, t


def _project(E):
    """Nearest matrix with singular values (1,1,0) -- the essential manifold."""
    U, S, Vh = torch.linalg.svd(E.double())
    s = torch.zeros_like(S)
    s[:, 0] = s[:, 1] = 1.0
    return (U @ torch.diag_embed(s) @ Vh).to(E.dtype), U.to(E.dtype), Vh.to(E.dtype)


def _cheirality(U, Vh, x0, x1, w):
    """Pick the (R,t) among the four decompositions with the most points in
    front of both cameras. Linear triangulation, batched over the candidates."""
    key = (U.device,)
    if key not in _W:
        _W[key] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]],
                               device=U.device, dtype=torch.float64)
    Wm = _W[key]
    Ud, Vd = U.double(), Vh.double()
    # det<0 turns a rotation into a reflection; absorb the sign first.
    Ud = Ud * torch.sign(torch.linalg.det(Ud))[:, None, None]
    Vd = Vd * torch.sign(torch.linalg.det(Vd))[:, None, None]
    R1, R2 = Ud @ Wm @ Vd, Ud @ Wm.T @ Vd
    t = Ud[..., 2]
    R = torch.stack([R1, R1, R2, R2], 1)                      # (B,4,3,3)
    T = torch.stack([t, -t, t, -t], 1)                        # (B,4,3)

    p0 = x0.double()[:, None].expand(-1, 4, -1, -1)
    p1 = x1.double()[:, None].expand(-1, 4, -1, -1)
    # X = d0*p0 in cam0 and d1*p1 in cam1, with X1 = R X0 + t, so
    # d0*p0 - d1*(R^T p1) = -R^T t. Two-unknown least squares per point.
    r1 = torch.einsum("bkij,bkni->bknj", R, p1)
    Rt = torch.einsum("bkij,bki->bkj", R, T)[:, :, None, :]
    a11 = (p0 * p0).sum(-1)
    a12 = -(p0 * r1).sum(-1)
    a22 = (r1 * r1).sum(-1)
    b1 = -(p0 * Rt).sum(-1)
    b2 = (r1 * Rt).sum(-1)
    det = (a11 * a22 - a12 * a12).clamp(min=1e-12)
    d0 = (a22 * b1 - a12 * b2) / det
    d1 = (a11 * b2 - a12 * b1) / det
    ok = ((d0 > 0) & (d1 > 0)).to(torch.float64) * w[:, None]
    best = ok.sum(-1).argmax(1)
    b = torch.arange(len(R), device=R.device)
    return R[b, best], T[b, best], ok[b, best].sum(-1)


def solve(pairs, K, device="cuda", iters=3, thr_px=1.0, min_inl=15, seed=0):
    """pairs: list of (p0, p1) pixel-coordinate arrays. -> [(R, n_inl) | None].

    R follows cv2.recoverPose: the rotation taking camera 0 into camera 1.
    """
    if not pairs:
        return []
    B = len(pairs)
    N = max(len(p[0]) for p in pairs)
    P0 = np.zeros((B, N, 2), np.float64)
    P1 = np.zeros((B, N, 2), np.float64)
    M = np.zeros((B, N), np.float64)
    for i, (a, b) in enumerate(pairs):
        n = len(a)
        P0[i, :n], P1[i, :n], M[i, :n] = a, b, 1.0
    dt = torch.float64
    p0 = torch.as_tensor(P0, device=device, dtype=dt)
    p1 = torch.as_tensor(P1, device=device, dtype=dt)
    mask = torch.as_tensor(M, device=device, dtype=dt)

    Kt = torch.as_tensor(np.asarray(K, np.float64), device=device, dtype=dt)
    Ki = torch.linalg.inv(Kt)

    def to_cam(p):                                   # pixels -> normalised rays
        h = torch.cat([p, torch.ones_like(p[..., :1])], -1)
        return torch.einsum("ij,bnj->bni", Ki, h)

    c0, c1 = to_cam(p0), to_cam(p1)

    # No Hartley normalisation. It conditions the DLT, but it is a similarity,
    # so the normalised E is no longer on the essential manifold and the sigma =
    # (1,1,0) constraint -- worth ~2x in accuracy here -- cannot be imposed
    # between iterations. Camera-normalised rays are already O(1) for this
    # optic (f ~= 400 px over a 720x540 frame), so the conditioning argument
    # does not bite.
    x0, x1 = c0, c1
    thr = (thr_px / max(float(Kt[0, 0]), 1e-9)) ** 2
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    E = _seed(x0, x1, mask, thr, gen)

    # No GNC annealing. The hypothesis search already lands within 10% of the
    # ground-truth MSAC cost (110.7 against 100.9 on a 300-match edge), and a
    # wide-scale IRLS restarted from there DRIFTS -- it reweights the outliers
    # back in and ends at 246. Only tight reweighting is useful here.
    for _ in range(iters):
        r = _sampson(E, x0, x1)
        hard = mask * (r < 5.9 * thr).to(dt)         # chi2(2) @ 95%
        hard = torch.where(hard.sum(1, keepdim=True) < min_inl, mask, hard)
        E = _dlt(x0, x1, hard / (1.0 + r / thr))
        # The 9-vector DLT solves an 8-DOF fundamental matrix; an essential
        # matrix has 5, so re-impose sigma = (1,1,0) every iteration.
        E, U, Vh = _project(E)

    tol = (thr_px / max(float(Kt[0, 0]), 1e-9)) ** 2
    inl = mask * (_sampson(E, c0, c1) < tol).to(dt)
    R, tvec, ncheir = _cheirality(U, Vh, c0, c1, inl)
    wr = mask / (1.0 + _sampson(_skew(tvec) @ R, c0, c1) / tol)
    R, _ = _refine_manifold(R, tvec, c0, c1, wr)
    inl = mask * (_sampson(_skew(tvec) @ R, c0, c1) < tol).to(dt)
    R = R.cpu().numpy()
    n = torch.minimum(inl.sum(1), ncheir).cpu().numpy()
    return [(R[i], float(n[i])) if n[i] >= min_inl else None for i in range(B)]
