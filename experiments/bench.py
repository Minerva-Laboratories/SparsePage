"""Latency per attention call of Riffle against exact attention (same kernel, all pages), eager and inside a CUDA graph
of 8 calls (the 9B's attention layers). Synthetic caches with drifting topic means (batch 1, G = 4).
Env: BENCH_HKV (4 = Qwen3.5-9B), BENCH_D (256), BENCH_CTX (32768,131072,262144), BENCH_RHO (0.1,0.2), BENCH_OUT."""
import json, os

import numpy as np
import torch

from riffle import RiffleCache

torch.manual_seed(0)
HKV, D = int(os.environ.get("BENCH_HKV", 4)), int(os.environ.get("BENCH_D", 256))
CTX = [int(x) for x in os.environ.get("BENCH_CTX", "32768,131072,262144").split(",")]
RHOS = [float(x) for x in os.environ.get("BENCH_RHO", "0.1,0.2").split(",")]
N = 30


def synth(T):
    K = torch.empty(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    for a in range(0, T, 512 * 256):                                   # topic means drifting every 512 tokens
        b = min(T, a + 512 * 256)
        nt = -(-(b - a) // 512)
        mu = torch.randn(nt, HKV, D, device="cuda") * 0.8
        K[a:b] = (mu.repeat_interleave(512, 0)[: b - a] + torch.randn(b - a, HKV, D, device="cuda") * 0.6).to(torch.bfloat16)
    V = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(HKV * 4, D, device="cuda") * 1.5
    return K, V, q


def timed(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def graph_ms(f, calls=8):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            f()
    return timed(g.replay) / calls


rows = []
for T in CTX:
    K, V, q = synth(T)
    for rho in [1.0] + RHOS:
        c = RiffleCache(HKV, T + 256, head_dim=D, rho=rho)
        c.prefill(K, V)
        row = dict(T=T, Hkv=HKV, D=D, rho=rho, read_bits=c.read_bits())
        if rho < 1:
            ref, out = c.attend_ref(q), c.attend_kernel(q)
            row["out_rel_diff_vs_ref"] = float((out - ref).norm() / ref.norm())
        row["ms_eager"] = timed(lambda: c.attend_kernel(q))
        row["ms_graph_per_call"] = graph_ms(lambda: c.attend_kernel(q))
        rows.append(row)
        print(f"T={T:8d} rho={rho:4.2f} eager {row['ms_eager'] * 1e3:8.1f} us  graph/call {row['ms_graph_per_call'] * 1e3:8.1f} us"
              f"  bits/step {row['read_bits']:.2f}" + (f"  out vs ref {row['out_rel_diff_vs_ref']:.1e}" if rho < 1 else ""),
              flush=True)
        del c
        torch.cuda.empty_cache()
    del K, V
    torch.cuda.empty_cache()
out = os.environ.get("BENCH_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"bench_h{HKV}_d{D}.json"))
json.dump(rows, open(out, "w"), indent=1)
print("RIFFLE_BENCH_DONE")
