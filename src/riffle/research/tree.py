"""Barnes-Hut tree over the keys (research reference in PyTorch, not a kernel): the design study that led to Riffle.

Originally developed in the FoveaKV repository's compute track (E24-E26); moved here with the method.
At prefill, per KV head, the region [64, T_prefill - 128) is clustered into nodes of ~64 keys by k-means in the query
metric E[q q^T]^(1/2) (from the prefill's own queries); parents = k-means of the node centroids (~sqrt(#nodes)). Each
node keeps the second-cumulant summary: mean, rank-2 covariance + isotropic rest (scaled by sigma = 0.5), per-channel
max/min bound, value mean and the value-key cross covariance on the two directions (tilted value mean). Sinks (first
64), the last 128 prefill tokens and every decoded token stay exact.

Per decode step and KV head (shared by its GQA group, max of the query heads' estimates): estimate every node's log mass
(flat) or the parents' first and only the children of the Pp heaviest parents (tree); open nodes by estimated mass until
rho of the region's tokens; opened tokens are read exactly, unopened nodes contribute their summary, unopened parents
their parent summary. 'drop' = the same opening, unopened nodes dropped (Riffle's choice, on contiguous pages).
compute() = (exact tokens + node evaluations x (4 + 2r)/2 token-equivalents) / context, averaged over steps."""
import math
import torch

G = 4
RK, SIG, NODE, SINK, W_ = 2, 0.5, 64, 64, 128


def kmeans(X, k, iters=12, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        lab = ((X * X).sum(-1, keepdim=True) - 2 * X @ C.T + (C * C).sum(-1)[None]).argmin(-1)
        S = torch.zeros_like(C).index_add_(0, lab, X)
        cnt = torch.bincount(lab, minlength=k).float()
        keep = cnt > 0
        C[keep] = S[keep] / cnt[keep, None]
    return lab


def node_stats(K, V, lab, n):
    """K, V (N, D) fp32, lab (N,) -> dict of per-node summaries (n nodes)."""
    D = K.shape[1]
    cnt = torch.bincount(lab, minlength=n).float().clamp_min(1)
    mu = torch.zeros(n, D, device=K.device).index_add_(0, lab, K) / cnt[:, None]
    muv = torch.zeros(n, D, device=K.device).index_add_(0, lab, V) / cnt[:, None]
    X, Y = K - mu[lab], V - muv[lab]
    Ck = torch.zeros(n, D, D, device=K.device)
    Cvk = torch.zeros(n, D, D, device=K.device)
    for a in range(0, len(K), 1024):                     # bounded memory: outer products in chunks
        sl = slice(a, a + 1024)
        Ck.index_add_(0, lab[sl], X[sl, :, None] * X[sl, None, :])
        Cvk.index_add_(0, lab[sl], Y[sl, :, None] * X[sl, None, :])
    Ck /= cnt[:, None, None]; Cvk /= cnt[:, None, None]
    lam, U = torch.linalg.eigh(Ck)                       # ascending
    lam, U = lam[:, -RK:].clamp_min(0), U[:, :, -RK:]
    tr = torch.diagonal(Ck, dim1=1, dim2=2).sum(-1)
    res = ((tr - lam.sum(-1)) / (D - RK)).clamp_min(0)
    Wm = Cvk @ U                                         # (n, D, RK)
    mx = torch.full((n, D), -1e30, device=K.device).scatter_reduce(0, lab[:, None].expand(-1, D), K, "amax")
    mn = torch.full((n, D), 1e30, device=K.device).scatter_reduce(0, lab[:, None].expand(-1, D), K, "amin")
    logn = torch.log(torch.bincount(lab, minlength=n).float())          # -inf for empty nodes
    return dict(mu=mu, muv=muv, U=U, lam=lam, res=res, W=Wm, mx=mx, mn=mn, logn=logn)


def summarize(st, q, s):
    """q (G, D) for one KV head -> log mass (G, n), tilted values (G, n, D)."""
    a = s * q @ st["mu"].T
    z = torch.einsum("gd,ndr->gnr", q, st["U"])
    qn2 = (q * q).sum(-1, keepdim=True)
    quad = s * s * ((st["lam"][None] * z * z).sum(-1) + st["res"][None] * (qn2 - (z * z).sum(-1)))
    ub = s * (q.clamp_min(0) @ st["mx"].T + q.clamp_max(0) @ st["mn"].T)
    ln = st["logn"][None]
    lm = torch.minimum(ln + a + SIG * quad / 2, ln + ub)
    val = st["muv"][None] + SIG * s * torch.einsum("gnr,ndr->gnd", z, st["W"])
    return lm, val


class Tree:
    def __init__(self, Hkv, Tmax, device="cuda", rho=0.2, mode="flat", pp=6, drop=False, head_dim=256):
        self.Hkv, self.dev = Hkv, device
        self.kraw = torch.zeros(Tmax, Hkv, head_dim, dtype=torch.bfloat16, device=device)
        self.vraw = torch.zeros_like(self.kraw)
        self.n = 0
        self.scale = head_dim ** -0.5
        self.rho, self.mode, self.pp, self.drop = rho, mode, pp, drop
        self.name = f"tree {mode} rho={rho}" + (" drop" if drop else "")
        self.cost, self.steps = 0.0, 0

    def _store(self, k, v):
        t = k.shape[0]
        self.kraw[self.n:self.n + t] = k.to(torch.bfloat16)
        self.vraw[self.n:self.n + t] = v.to(torch.bfloat16)
        self.n += t

    prefill = append = _store

    def build_tree(self, qm):
        """qm (Hkv, D, D) = E[q q^T] per KV head from the prefill."""
        T0 = self.n
        self.r0, self.r1 = SINK, SINK + ((T0 - W_ - SINK) // NODE) * NODE
        n_tok = self.r1 - self.r0
        self.nb = n_tok // NODE
        self.npar = int(math.ceil(math.sqrt(self.nb)))
        self.heads = []
        D = self.kraw.shape[-1]
        for h in range(self.Hkv):
            K = self.kraw[self.r0:self.r1, h].float()
            V = self.vraw[self.r0:self.r1, h].float()
            w, U = torch.linalg.eigh(qm[h].float())
            QM = U @ torch.diag(w.clamp_min(0).sqrt()) @ U.T
            Xq = K @ QM
            lab = kmeans(Xq, self.nb, seed=h)
            cnt = torch.bincount(lab, minlength=self.nb)
            cent = torch.zeros(self.nb, D, device=K.device).index_add_(0, lab, Xq) / cnt.clamp_min(1)[:, None].float()
            cent[cnt == 0] = 1e6
            plab = kmeans(cent, self.npar, seed=100 + h)                    # parent of each node
            self.heads.append(dict(lab=lab, plab=plab, size=cnt.float(),
                                   c=node_stats(K, V, lab, self.nb), p=node_stats(K, V, plab[lab], self.npar)))

    def attend(self, q):
        n, s = self.n, self.scale
        D = self.kraw.shape[-1]
        qh = q.float().view(self.Hkv, G, D)
        out, cost = [], 0.0
        n_tok = self.r1 - self.r0
        node_cost = (4 + 2 * RK) / 2
        for h in range(self.Hkv):
            H_ = self.heads[h]
            K, V = self.kraw[:n, h].float(), self.vraw[:n, h].float()
            sc = s * qh[h] @ K.T                                                # (G, n) exact logits (reference)
            lmc, valc = summarize(H_["c"], qh[h], s)
            if self.mode == "flat":
                elig = torch.ones(self.nb, dtype=torch.bool, device=q.device)
                popen = torch.ones(self.npar, dtype=torch.bool, device=q.device)
                neval = self.nb
            else:
                lmp, valp = summarize(H_["p"], qh[h], s)
                popen = torch.zeros(self.npar, dtype=torch.bool, device=q.device)
                popen[lmp.max(0).values.topk(min(self.pp, self.npar)).indices] = True
                elig = popen[H_["plab"]]
                neval = self.npar + int(elig.sum())
            score = torch.where(elig, lmc.max(0).values, torch.tensor(-math.inf, device=q.device))
            order = score.argsort(descending=True)
            csz = H_["size"][order].cumsum(0)
            nopen = min(int((csz <= self.rho * n_tok).sum()), int(elig.sum()))
            copen = torch.zeros(self.nb, dtype=torch.bool, device=q.device)
            copen[order[:nopen]] = True
            tok = torch.ones(n, dtype=torch.bool, device=q.device)
            tok[self.r0:self.r1] = copen[H_["lab"]]
            lg = [torch.where(tok[None], sc, torch.tensor(-math.inf, device=q.device))]
            vals = [None]
            if not self.drop:
                lg.append(torch.where((elig & ~copen)[None], lmc, torch.tensor(-math.inf, device=q.device)))
                vals.append(valc)
                if self.mode != "flat":
                    lg.append(torch.where((~popen)[None], lmp, torch.tensor(-math.inf, device=q.device)))
                    vals.append(valp)
            M = torch.stack([x.max(-1).values for x in lg]).max(0).values[:, None]
            num, den = 0, 0
            for x, v in zip(lg, vals):
                p = torch.exp(x - M)
                den = den + p.sum(-1)
                num = num + (p @ V if v is None else torch.einsum("gn,gnd->gd", p, v))
            out.append(num / den[:, None])
            cost += (float(tok.sum()) + neval * node_cost) / n
        self.cost += cost / self.Hkv
        self.steps += 1
        return torch.cat(out, 0)

    def compute(self):
        return self.cost / max(self.steps, 1)

    def read_bits(self):
        return 16.0
