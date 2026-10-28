"""SparsePageCache: decode attention that reads only the pages a query needs.

Layout per attention layer: bf16 K/V (Hkv, Tcap, D); pages of 128 tokens. Every page older than the recent window
gets summaries (summaries.py) when it is written. Per decode step and KV head: score all summarized pages, open the top
ceil(rho * pages) (page 0 always), read them exactly, drop the rest (their mass is neither read nor estimated), add the
exact recent window (the last `window`..`window + 127` tokens), merge. The kernel path (attend_kernel) runs without host
synchronization, so it can be captured in a CUDA graph; it passes the page and token counts from the host, so a graph
captured here is valid for the cache length at capture (the kernels' device-side counts, used by HgRT, lift that).
attend_ref is the PyTorch reference.

Shapes: keys / values (T, Hkv, D) for prefill and (1, Hkv, D) for append; queries (Hkv * G, D) post-RoPE, unscaled."""
import math

import torch

from .summaries import PAGE, RK, SIG, page_scores, summarize

G = 4


def tokens_per_block(n, Hkv, target_blocks=160):
    """split-K block size: a multiple of 128 (whole pages; 4 warps x 32-token tiles), about target_blocks blocks."""
    t = max(PAGE, -(-n * Hkv // target_blocks))
    return -(-t // PAGE) * PAGE


class SparsePageCache:
    def __init__(self, Hkv, Tmax, head_dim=256, rho=0.1, cov=True, bound=True, sig=SIG, window=128, device="cuda",
                 use_kernel=None):
        assert window % PAGE == 0
        self.Hkv, self.D, self.rho, self.cov, self.bound, self.sig, self.window = Hkv, head_dim, rho, cov, bound, sig, window
        self.dev = device
        self.cap = -(-Tmax // PAGE) * PAGE
        npg = self.cap // PAGE
        self.k = torch.zeros(Hkv, self.cap, head_dim, dtype=torch.bfloat16, device=device)
        self.v = torch.zeros_like(self.k)
        f16 = lambda *s: torch.zeros(*s, dtype=torch.float16, device=device)
        self.mu, self.U = f16(Hkv, npg, head_dim), f16(Hkv, npg, RK, head_dim)
        self.lamres = torch.zeros(Hkv, npg, RK + 1, device=device)
        self.mx, self.mn = f16(Hkv, npg, head_dim), f16(Hkv, npg, head_dim)
        self.scores = torch.empty(Hkv, npg, device=device)
        self.sel = torch.zeros(Hkv, npg, dtype=torch.int32, device=device)
        self.n = 0              # tokens stored
        self.npg = 0            # pages summarized (selectable); the rest [npg * PAGE, n) is the exact window
        self.scale = head_dim ** -0.5
        self.use_kernel = (str(device).startswith("cuda")) if use_kernel is None else use_kernel
        self.last_sel = None

    # ---------------------------------------------------------------------------------------------- writes
    def _store(self, k, v):
        t = k.shape[0]
        self.k[:, self.n:self.n + t] = k.transpose(0, 1).to(torch.bfloat16)
        self.v[:, self.n:self.n + t] = v.transpose(0, 1).to(torch.bfloat16)
        self.n += t
        target = max(0, (self.n - self.window) // PAGE)
        if target > self.npg:
            a, b = self.npg, target
            s = summarize(self.k[:, a * PAGE:b * PAGE].transpose(0, 1))
            self.mu[:, a:b], self.U[:, a:b] = s["mu"].half(), s["U"].half()
            self.lamres[:, a:b] = s["lamres"]
            self.mx[:, a:b], self.mn[:, a:b] = s["mx"].half(), s["mn"].half()
            self.npg = target

    def prefill(self, k, v):
        self._store(k, v)

    def append(self, k, v):
        self._store(k, v)

    # ---------------------------------------------------------------------------------------------- reads
    def n_open(self):
        return min(self.npg, max(1, math.ceil(self.rho * self.npg))) if self.npg else 0

    def _q(self, q):
        return (q.float() * self.scale).view(self.Hkv, G, self.D).contiguous()

    def page_scores(self, q):
        p = self.npg
        return page_scores(self._q(q), self.mu[:, :p], self.U[:, :p], self.lamres[:, :p], self.mx[:, :p],
                           self.mn[:, :p], self.cov, self.bound, self.sig)

    def attend_ref(self, q):
        """PyTorch reference: same selection, exact softmax over the opened pages + window."""
        qs = self._q(q)
        if self.npg:
            sel = torch.topk(self.page_scores(q), self.n_open(), dim=-1).indices
        else:
            sel = torch.empty(self.Hkv, 0, dtype=torch.long, device=q.device)
        self.last_sel = sel
        win = torch.arange(self.npg * PAGE, self.n, device=q.device)
        out = []
        for h in range(self.Hkv):
            tok = torch.cat([(sel[h][:, None] * PAGE + torch.arange(PAGE, device=q.device)[None]).reshape(-1), win])
            Kh, Vh = self.k[h, tok].float(), self.v[h, tok].float()
            out.append(torch.softmax(qs[h] @ Kh.T, -1) @ Vh)
        return torch.cat(out)

    def _partials(self, nb):
        e = lambda *s: torch.empty(*s, device=self.dev)
        return e(nb, self.Hkv, G), e(nb, self.Hkv, G), e(nb, self.Hkv, G, self.D)

    def attend_kernel(self, q):
        from .ext import ext
        e = ext(self.D)
        qs = self._q(q)
        m = self.n_open()
        if m:
            e.score(qs, qs, self.mu, self.U, self.lamres, self.mx, self.mn, self.npg, int(self.cov), int(self.bound),
                    self.sig, self.scores, None)
            e.select(self.scores, self.npg, m, self.sel, None, 0.0, None)
        n1 = m * PAGE
        tpb1 = tokens_per_block(max(n1, 1), self.Hkv)
        nb1 = -(-n1 // tpb1)
        p1 = self._partials(max(nb1, 1))
        if nb1:
            e.pass_(qs, self.k, self.v, 0, n1, self.sel, tpb1, *p1)
        w0, nw = self.npg * PAGE, self.n - self.npg * PAGE
        nb2 = -(-nw // 64)
        p2 = self._partials(max(nb2, 1))
        if nb2:
            e.pass_(qs, self.k, self.v, w0, nw, None, 64, *p2)
        out = torch.empty(self.Hkv * G, self.D, device=self.dev)
        e.combine(*p1, nb1, *p2, nb2, None, out)
        return out

    def attend(self, q):
        return self.attend_kernel(q) if self.use_kernel else self.attend_ref(q)

    def read_bits(self):
        """bits read per cached token per step: opened pages + window in bf16, all summaries."""
        if self.n == 0:
            return 16.0
        summ = 16 * self.D * ((RK if self.cov else 0) + 1 + (2 if self.bound else 0)) + 32 * (RK + 1)
        per_head = (self.n_open() * PAGE + self.n - self.npg * PAGE) * 2 * 16 * self.D + self.npg * summ
        return per_head / (2 * self.D) / self.n
