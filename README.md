# SparsePage

Decode attention for long contexts that reads only the pages a query needs. Every 128-token page of the KV cache
gets a small key summary when it is written; at each decode step SparsePage scores all summaries, reads the top
`rho` of the pages exactly (page 0, the attention sinks, always), **drops the rest** (their mass is neither read nor
estimated) and adds the exact recent window. Scoring, top-m selection, the paged attention pass and the merge are CUDA
kernels with no host synchronization, so a whole decode step can be captured in a CUDA graph.

SparsePage is a method repository: bf16 pages, a PyTorch reference for every kernel, an end-to-end harness for Qwen3.5.
It composes with quantized pages (the scoring and selection kernels take the query in the summaries' domain and in the
bound's domain separately, and the merge takes an optional unrotation): the combined stack with FoveaKV 2+2-bit pages
and DepthSkip prefill lives in [HgRT](https://github.com/Minerva-Laboratories/hgrt).

## How a page is scored

Summaries per page and KV head (`src/sparsepage/summaries.py`): mean key `mu`, top-2 covariance directions `U` with
variances `lam` and the isotropic rest `res`, per-channel `max`/`min`. For a query `q` (scaled by 1/sqrt(d)):

    est   = q.mu + sigma/2 * (sum_r lam_r (q.U_r)^2 + res (|q|^2 - sum_r (q.U_r)^2))   # log-mass of a Gaussian page
    bound = sum_d max(q_d max_d, q_d min_d)                                          # no key in the page scores more
    score = max over the GQA group of min(est, bound)

`est` is the second-cumulant estimate of the page's log-sum-exp (sigma = 0.5 by default); the bound is Quest's. The
top `ceil(rho * pages)` pages per KV head are found by a 4-round radix select in shared memory.

## Usage

```python
from sparsepage import SparsePageCache
c = SparsePageCache(Hkv=4, Tmax=262144, head_dim=256, rho=0.1)       # one per full-attention layer
c.prefill(k, v)            # (T, Hkv, D) post-RoPE keys, values
c.append(k1, v1)           # (1, Hkv, D) per decoded token
out = c.attend(q)          # (Hkv * 4, D) post-RoPE query -> (Hkv * 4, D); CUDA kernels, attend_ref = PyTorch
```

Limits today: batch 1, GQA group 4 (the Qwen3.5 family), head_dim 128 or 256.

## Results

Qwen3.5-9B on the Jetson AGX Orin (MODE_50W). **SparsePage on bf16 pages: running** (`experiments/eval_ppl.py`,
`experiments/bench.py`; tables will be filled from `results/`).

Measured during development, in FoveaKV's harness (`results/research/`), on Qwen3.5-9B, protocol: exact prefill, 256
tokens teacher-forced through the cache, prose (WikiText-2) and code (transformers sources), 3 documents per length;
ΔPPL against exact attention:

| scheme (selection over 64-token k-means nodes) | compute vs dense | 16K | 32K | 64K (2 docs) |
|---|---|---|---|---|
| tree, open 10%, unopened nodes estimated | 0.17 | +0.29% | -0.10% | -0.19% |
| tree, open 20%, unopened estimated | 0.27 | +0.25% | +0.06% | +0.02% |
| **open 20%, unopened dropped** | 0.27 | +0.02% | -0.31% | -0.02% |
| Quest 5% pages (min/max bound only) | 0.18 | +1.15% | +0.60% | +0.32% |
| Quest 15% pages | 0.28 | +0.39% | -0.34% | -0.05% |

Estimating the unopened mass (Barnes-Hut style) buys nothing over dropping it, which is why SparsePage drops. With the
same scoring on contiguous FoveaKV 4-bit pages (HgRT's stack), the kernels give 0.36 / 0.79 / 1.45 ms per attention
call at 32K / 128K / 256K with 4 KV heads (exact bf16 kernel: 5.5 / 21.5 / 42.3 ms) and ΔPPL +0.21% at 20% /
+0.63% at 10% on the same protocol.

## Related work and what is new

Page selection by summaries is established: Quest (arXiv:2406.10774; min/max bound, top pages), ArkVale (page digests,
NeurIPS 2024), InfLLM, LServe (arXiv:2502.14866; hierarchical query-centric page selection with fused kernels), and
cluster-based variants that also approximate the unselected keys: Multipole Attention (arXiv:2506.13059), RetroInfer
(arXiv:2505.02922), ClusterKV, MagicPIG. SparsePage's differences: the score combines a second-cumulant (covariance)
estimate of each page's log-mass with Quest's bound; the unselected pages are dropped (measured above to lose nothing
against estimating them); the whole step is graph-capturable with on-GPU selection; and the kernels compose with
quantized (rotated) pages. At 10% the covariance + bound score is slightly better than the bound or the mean alone,
but the confidence intervals overlap.

## Layout

```
src/sparsepage/csrc/sparsepage.cu   score, select, paged pass, combine
src/sparsepage/summaries.py     page summaries and the reference score
src/sparsepage/cache.py         SparsePageCache (kernel path + PyTorch reference)
src/sparsepage/qwen.py          Qwen3.5 decode harness (any cache in the loop)
src/sparsepage/research/tree.py Barnes-Hut tree over keys (the design study; PyTorch)
experiments/                bench.py (latency), eval_ppl.py (end-to-end perplexity)
tests/                      CPU tests; -m gpu: kernels vs reference, graph capture
```

```bash
pip install -e .            # kernels build on first use for the local GPU (nvcc matching torch's CUDA)
python -m pytest -q         # add -m "not gpu" without CUDA
```

License: Apache-2.0. The kernels are adapted from the FoveaKV decode kernels.
