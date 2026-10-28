"""Per-page key summaries (one page = 128 consecutive tokens of one KV head), computed once when a page is written:
  mu      the page's mean key
  U, lam  its top-RK covariance directions and their variances (subspace iteration on the 128 centered keys)
  res     the isotropic remainder: (trace - sum lam) / (D - RK)
  mx, mn  per-channel max / min (the Quest bound)
The page score for a query q (pre-scaled by the softmax scale) is a log-mass estimate of the page:
  est = q.mu + sig/2 (sum_r lam_r (q.U_r)^2 + res (|q|^2 - sum_r (q.U_r)^2))        (second cumulant, Gaussian keys)
  ub  = sum_d max(q_d mx_d, q_d mn_d)                                                (no key in the page scores more)
  score = min(est, ub), max over the query heads of the GQA group; page 0 (attention sinks) always opens."""
import math

import torch

PAGE, RK, SIG = 128, 2, 0.5


_INIT = {}


def _orth(A):
    """(B, D, rk) -> orthonormal columns spanning the same subspace. rk = 2: Gram-Schmidt in plain tensor ops (no
    cuSOLVER call, so it is CUDA-graph capturable and cheap for tiny batches); equal to torch.linalg.qr's Q up to
    column signs and rounding. Summaries only enter scores through (q . U_r)^2, so signs do not matter."""
    if A.shape[-1] != 2:
        return torch.linalg.qr(A)[0]
    a, b = A[..., 0], A[..., 1]
    u1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b = b - (u1 * b).sum(-1, keepdim=True) * u1
    return torch.stack([u1, b / b.norm(dim=-1, keepdim=True).clamp_min(1e-30)], -1)


def top_dirs(X, rk=RK, iters=6):
    """X (B, n, D) centered rows -> U (B, D, rk) top covariance directions, lam (B, rk), total variance (B,).
    Subspace iteration from a fixed random start (seed 0, cached per shape so no generator is created per call)."""
    B, n, D = X.shape
    key = (B, D, rk, str(X.device))
    if key not in _INIT:
        g = torch.Generator(device=X.device).manual_seed(0)
        _INIT[key] = _orth(torch.randn(B, D, rk, device=X.device, generator=g))
    U = _INIT[key]
    for _ in range(iters):
        U = _orth(X.transpose(1, 2) @ (X @ U) / n)
    XU = X @ U
    return U, (XU * XU).sum(1) / n, (X * X).sum((1, 2)) / n


def summarize(k):
    """k (P*PAGE, Hkv, D) keys of whole pages -> dict of (Hkv, P, ...) summaries (fp32)."""
    T, Hkv, D = k.shape
    P = T // PAGE
    x = k.float().view(P, PAGE, Hkv, D).permute(2, 0, 1, 3)                  # (Hkv, P, PAGE, D)
    mu = x.mean(2)
    U, lam, tr = top_dirs((x - mu[:, :, None]).reshape(-1, PAGE, D))
    res = ((tr - lam.sum(-1)) / (D - RK)).clamp_min(0)
    return dict(mu=mu, U=U.transpose(1, 2).reshape(Hkv, P, RK, D), lamres=torch.cat([lam, res[:, None]], -1).view(Hkv, P, RK + 1),
                mx=x.amax(2), mn=x.amin(2))


def page_scores(q_s, mu, U, lamres, mx, mn, cov=True, bound=True, sig=SIG, q_b=None):
    """PyTorch reference. q_s (Hkv, G, D) query in the summaries' domain (scaled), q_b the same in the bound's domain
    (defaults to q_s); summaries (Hkv, P, ...). Returns (Hkv, P) with page 0 = +inf."""
    q_b = q_s if q_b is None else q_b
    est = torch.einsum("hgd,hpd->hgp", q_s, mu.float())
    if cov:
        z = torch.einsum("hgd,hprd->hgpr", q_s, U.float())
        lr = lamres.float()
        qn2 = (q_s * q_s).sum(-1)[..., None]
        quad = (lr[:, None, :, :RK] * z * z).sum(-1) + lr[:, None, :, RK] * (qn2 - (z * z).sum(-1))
        est = est + sig / 2 * quad
    if bound:
        ub = (torch.einsum("hgd,hpd->hgp", q_b.clamp_min(0), mx.float())
              + torch.einsum("hgd,hpd->hgp", q_b.clamp_max(0), mn.float()))
        est = torch.minimum(est, ub)
    sc = est.amax(1)
    sc[:, 0] = math.inf
    return sc
