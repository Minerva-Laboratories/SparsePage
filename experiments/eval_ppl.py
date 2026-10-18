"""End-to-end perplexity on Qwen3.5 with Riffle (and the research tree) in the decode loop, against exact attention.
Protocol: exact prefill of L-N tokens, N tokens teacher-forced through the cache (src/riffle/qwen.py).
Env: RIFFLE_MODEL (default Qwen/Qwen3.5-9B), RIFFLE_LENS (16384,32768), RIFFLE_NDOC (3), RIFFLE_N (256),
RIFFLE_SCHEMES, RIFFLE_SRCS (prose,code), RIFFLE_OUT (results/<model>/ppl.jsonl, resumable)."""
import json, math, os, time

import torch

from riffle import RiffleCache
from riffle.qwen import DEV, Runner, code_text, doc_windows, prose_text
from riffle.research.tree import Tree

MODEL = os.environ.get("RIFFLE_MODEL", "Qwen/Qwen3.5-9B")
LENS = [int(x) for x in os.environ.get("RIFFLE_LENS", "16384,32768").split(",")]
NDOC, N = int(os.environ.get("RIFFLE_NDOC", 3)), int(os.environ.get("RIFFLE_N", 256))
SRCS = os.environ.get("RIFFLE_SRCS", "prose,code").split(",")


def riffle(rho, **kw):
    return lambda Hkv, Tmax: RiffleCache(Hkv, Tmax, head_dim=R.hd, rho=rho, **kw)


def tree(**kw):
    class T(Tree):
        def after_prefill(self, qm):
            self.build_tree(qm)
    return lambda Hkv, Tmax: T(Hkv, Tmax, head_dim=R.hd, **kw)


SCHEMES = {"exact": lambda: riffle(1.0), "riffle5": lambda: riffle(0.05), "riffle10": lambda: riffle(0.10),
           "riffle20": lambda: riffle(0.20), "riffle10-bound": lambda: riffle(0.10, cov=False),
           "riffle10-mean": lambda: riffle(0.10, cov=False, bound=False),
           "tree-flat-r10": lambda: tree(rho=0.1), "tree-drop-r20": lambda: tree(rho=0.2, drop=True)}
ORDER = os.environ.get("RIFFLE_SCHEMES", "exact,riffle5,riffle10,riffle20,riffle10-bound,riffle10-mean").split(",")
tag = MODEL.rstrip("/").split("/")[-1].lower()
out_p = os.environ.get("RIFFLE_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", tag, "ppl.jsonl"))
os.makedirs(os.path.dirname(out_p), exist_ok=True)
done = {(r["src"], r["L"], r["doc"], r["scheme"]) for r in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()

R = Runner(MODEL)
t0 = time.time()
for src in SRCS:
    text = prose_text() if src == "prose" else code_text()
    for L in LENS:
        for d, ids in enumerate(doc_windows(R.tok, text, L, NDOC, seed=L)):
            todo = [s for s in ORDER if (src, L, d, s) not in done]
            if not todo:
                continue
            x = torch.tensor(ids, device=DEV)
            snap = R.prefill(x[:L - N])
            for s in todo:
                ts = time.time()
                nll, st = R.decode_teacher(SCHEMES[s](), snap, x[L - N:])
                rec = dict(src=src, L=L, doc=d, scheme=s, nll=float(nll.mean()), nll_tok=[float(v) for v in nll], **st)
                with open(out_p, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{time.time() - t0:6.0f}s] {src} L={L} doc{d} {s:15s} PPL {math.exp(nll.mean()):8.4f} {st} "
                      f"({time.time() - ts:.0f}s)", flush=True)
            del snap
            torch.cuda.empty_cache()
print("RIFFLE_PPL_DONE")
