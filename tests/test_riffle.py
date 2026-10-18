"""Riffle: summaries, scores and the reference path on CPU; the CUDA kernels against the reference (marked gpu)."""
import math

import pytest
import torch

from riffle import PAGE, RK, RiffleCache, page_scores, summarize, top_dirs

H, D = 2, 128


def _kv(T, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    nt = -(-T // PAGE)
    mu = torch.randn(nt, 1, H, D, generator=g) * 0.8
    K = (mu.expand(nt, PAGE, H, D).reshape(-1, H, D)[:T] + 0.6 * torch.randn(T, H, D, generator=g))
    V = torch.randn(T, H, D, generator=g)
    q = torch.randn(H * 4, D, generator=g) * 1.5
    return K.to(dev), V.to(dev), q.to(dev)


def test_top_dirs_recovers_planted_subspace():
    g = torch.Generator().manual_seed(1)
    B, n = 3, PAGE
    basis = torch.linalg.qr(torch.randn(B, D, RK, generator=g))[0]
    X = (torch.randn(B, n, RK, generator=g) * torch.tensor([5.0, 3.0])) @ basis.transpose(1, 2)
    X = X + 0.01 * torch.randn(B, n, D, generator=g)
    X = X - X.mean(1, keepdim=True)
    U, lam, tr = top_dirs(X)
    overlap = (basis.transpose(1, 2) @ U).square().sum((1, 2))           # = RK when the subspaces coincide
    assert torch.allclose(overlap, torch.full((B,), float(RK)), atol=1e-3)
    assert torch.all(lam.sum(-1) <= tr + 1e-4)


def test_mean_score_and_bound():
    """cov off: the estimate is the page's mean logit; with a huge sigma the min() returns the bound, which no key in
    the page exceeds."""
    K, V, q = _kv(8 * PAGE)
    s = summarize(K)
    qs = (q * D ** -0.5).view(H, 4, D)
    logits = torch.einsum("hgd,thd->hgt", qs, K).view(H, 4, 8, PAGE)
    mean = page_scores(qs, s["mu"], s["U"], s["lamres"], s["mx"], s["mn"], cov=False, bound=False)
    assert torch.allclose(mean[:, 1:], logits.mean(-1).amax(1)[:, 1:], atol=1e-4)
    ub = page_scores(qs, s["mu"], s["U"], s["lamres"], s["mx"], s["mn"], cov=True, bound=True, sig=1e9)
    assert torch.all(ub[:, 1:] >= logits.amax(-1).amax(1)[:, 1:] - 1e-4) and torch.isinf(ub[:, 0]).all()


@pytest.mark.parametrize("T", [5, 300, 9 * PAGE + 17])
def test_reference_with_all_pages_is_dense_attention(T):
    K, V, q = _kv(T)
    c = RiffleCache(H, T + 64, head_dim=D, rho=1.0, device="cpu")
    c.prefill(K, V)
    qs = (q * D ** -0.5).view(H, 4, D)
    Kb, Vb = K.to(torch.bfloat16).float(), V.to(torch.bfloat16).float()
    dense = torch.cat([torch.softmax(qs[h] @ Kb[:, h].T, -1) @ Vb[:, h] for h in range(H)])
    assert torch.allclose(c.attend(q), dense, atol=1e-5)


def test_append_matches_prefill():
    K, V, q = _kv(6 * PAGE + 40)
    a = RiffleCache(H, 2048, head_dim=D, rho=0.3, device="cpu")
    a.prefill(K, V)
    b = RiffleCache(H, 2048, head_dim=D, rho=0.3, device="cpu")
    b.prefill(K[:200], V[:200])
    for t in range(200, len(K)):
        b.append(K[t:t + 1], V[t:t + 1])
    assert a.npg == b.npg == (len(K) - 128) // PAGE
    assert torch.allclose(a.mu.float(), b.mu.float()) and torch.allclose(a.attend(q), b.attend(q), atol=1e-6)


def test_window_and_open_count():
    K, V, q = _kv(10 * PAGE + 5)
    c = RiffleCache(H, 4096, head_dim=D, rho=0.1, device="cpu")
    c.prefill(K, V)
    assert c.npg == 9 and c.n_open() == 1                    # window = last 133 tokens; ceil(0.1 * 9) = 1 page (the sink)
    c.attend(q)
    assert torch.equal(c.last_sel, torch.zeros(H, 1, dtype=torch.long))
    assert 0 < c.read_bits() < 16


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("T,rho", [(64, 0.1), (5 * PAGE + 3, 0.5), (60 * PAGE + 77, 0.1), (60 * PAGE, 1.0)])
def test_kernels_match_reference(head_dim, T, rho):
    global D
    D0, D = D, head_dim
    try:
        K, V, q = _kv(T, dev="cuda")
        c = RiffleCache(H, T + 256, head_dim=head_dim, rho=rho, device="cuda")
        c.prefill(K, V)
        ref = c.attend_ref(q)
        sel_ref = c.last_sel
        out = c.attend_kernel(q)
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=2e-4, rtol=1e-3), float((out - ref).abs().max())
        if c.npg:
            fin = torch.isfinite(c.page_scores(q))
            assert torch.allclose(c.scores[:, :c.npg][fin], c.page_scores(q)[fin], atol=1e-3, rtol=1e-4)
            m = c.n_open()
            for h in range(H):
                assert set(c.sel[h, :m].tolist()) == set(sel_ref[h].tolist())
    finally:
        D = D0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_kernel_path_is_graph_capturable():
    K, V, q = _kv(40 * PAGE, dev="cuda")
    c = RiffleCache(H, 48 * PAGE, head_dim=D, rho=0.2, device="cuda")
    c.prefill(K, V)
    eager = c.attend_kernel(q).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        c.attend_kernel(q)
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = c.attend_kernel(q)
    gr.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, eager)


def test_research_tree_opening_everything_is_exact():
    from riffle.research.tree import Tree
    K, V, q = _kv(64 + 128 + 8 * 64 + 20, seed=3)
    t = Tree(H, len(K) + 8, device="cpu", rho=1.0, head_dim=D)
    t.prefill(K, V)
    qm = torch.stack([torch.eye(D)] * H)
    t.build_tree(qm)
    qh = q.view(H, 4, D)
    Kb, Vb = K.to(torch.bfloat16).float(), V.to(torch.bfloat16).float()
    dense = torch.cat([torch.softmax(D ** -0.5 * qh[h] @ Kb[:, h].T, -1) @ Vb[:, h] for h in range(H)])
    assert torch.allclose(t.attend(q), dense, atol=1e-5)
