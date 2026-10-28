<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/sparsepage-dark.png">
  <img alt="SparsePage" src="docs/assets/sparsepage-light.png" width="520">
</picture>

### Decode attention that reads only the pages a query needs

Every 128-token page of the KV cache gets a small key summary. Each decode step scores all of them on the GPU,<br>
reads the top 10-20% of pages exactly and drops the rest. No host synchronization, so the step fits in a CUDA graph.

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-1d9bd1.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-1d9bd1.svg)](pyproject.toml)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-1d9bd1.svg)](pyproject.toml)
[![CUDA kernels](https://img.shields.io/badge/CUDA-score%20%C2%B7%20select%20%C2%B7%20pass%20%C2%B7%20combine-1d9bd1.svg)](src/sparsepage/csrc/sparsepage.cu)
[![Tests: 8 CPU + 9 GPU](https://img.shields.io/badge/tests-8%20CPU%20%2B%209%20GPU-1baf7a.svg)](tests/test_sparsepage.py)
[![Part of: HgRT](https://img.shields.io/badge/part%20of-HgRT-2a78d6.svg)](https://github.com/Minerva-Laboratories/HgRT)

[**How a page is scored**](#how-a-page-is-scored) ·
[**Results**](#results) ·
[**Quickstart**](#quickstart) ·
[**Kernels**](#how-the-kernels-work) ·
[**Raw results**](results/README.md) ·
[**HgRT**](https://github.com/Minerva-Laboratories/HgRT) ·
[**FoveaKV**](https://github.com/Minerva-Laboratories/FoveaKV) ·
[**DepthSkip**](https://github.com/Minerva-Laboratories/DepthSkip)

</div>

---

At long context, decode attention is a memory-bandwidth problem: every step reads the whole KV cache for one query,
although most of the softmax mass sits on a small part of it. SparsePage keeps, for every 128-token page and KV head, a
summary of its keys (mean, the two main covariance directions, per-channel max and min) that is computed once when the
page is written. At each decode step it:

1. **scores** every page from its summary: an estimate of the page's attention mass, capped by an upper bound on its
   best logit;
2. **selects** the top `ceil(rho * pages)` pages per KV head on the GPU (page 0, the attention sinks, always);
3. **reads** those pages exactly, together with the exact recent window;
4. **drops** the rest. Their mass is neither read nor estimated.

All four steps are CUDA kernels that never read a value back to the host, so a decode step can be captured in a CUDA
graph. Every kernel has a PyTorch reference, and an end-to-end harness runs Qwen3.5 with any cache in the loop.

SparsePage is the decode-selection component of [HgRT](https://github.com/Minerva-Laboratories/HgRT), where it runs on
[FoveaKV](https://github.com/Minerva-Laboratories/FoveaKV)'s 2+2-bit pages after a
[DepthSkip](https://github.com/Minerva-Laboratories/DepthSkip) prefill. This repository holds the method on bf16
pages: the kernels, the reference, the harness, and the design study that led to dropping the unread pages.

> The project was first called **Riffle**. The raw result files in `results/` predate the rename and are kept as they
> were written; [`results/README.md`](results/README.md) maps their scheme names to the rows below.

## Highlights

<table>
<tr>
<td width="50%" valign="top">

**About 5x less time per attention call at 10%.**
With 4 KV heads and head_dim 256, inside a CUDA graph: 0.36 / 1.13 / 2.11 ms per call at 32K / 128K / 256K context,
against 1.94 / 5.57 / 11.1 ms when the same kernels open every page. About 2 bits read per cached value per step
instead of 16.

</td>
<td width="50%" valign="top">

**Dropping loses nothing measurable.**
In the design study on Qwen3.5-9B (16K / 32K / 64K), opening 20% of the nodes and dropping the rest gives ΔPPL of
+0.02% / -0.31% / -0.02%. Estimating the unopened mass instead (Barnes-Hut style) is no better: +0.25% / +0.06% /
+0.02%.

</td>
</tr>
<tr>
<td valign="top">

**A score with two terms.**
A second-cumulant estimate of each page's log-mass (mean + rank-2 covariance), capped by Quest's min/max bound. The
cap stops a page with a large spread from claiming mass that none of its keys can reach.

</td>
<td valign="top">

**Graph-capturable and composable.**
On-GPU radix select, no host sync, optional device-side page counts. The score and combine kernels accept a query in a
transformed domain and an unrotation matrix, so they run unchanged on FoveaKV's rotated, quantized pages.

</td>
</tr>
</table>

## How a page is scored

Per page (128 consecutive tokens of one KV head), computed once when the page leaves the exact window
(`src/sparsepage/summaries.py`):

| summary | per page | meaning |
|---|---|---|
| `mu` | D | mean key |
| `U`, `lam` | 2 x D, 2 | top-2 covariance directions and their variances (6 steps of subspace iteration on the 128 centered keys) |
| `res` | 1 | isotropic remainder of the variance: `(trace - sum lam) / (D - 2)` |
| `mx`, `mn` | D, D | per-channel max and min (the Quest bound) |

For a query `q`, already multiplied by the softmax scale `1/sqrt(D)`:

```text
est   = q.mu + sig/2 * ( sum_r lam_r (q.U_r)^2 + res * (|q|^2 - sum_r (q.U_r)^2) )    # second cumulant, Gaussian keys
ub    = sum_d max(q_d * mx_d, q_d * mn_d)                                               # no key in the page scores more
score = max over the query heads of the GQA group of min(est, ub);   page 0 (attention sinks) = +inf
```

- `est` is the second-order cumulant expansion of `log mean_k exp(q.k)` for keys with mean `mu` and covariance
  `U diag(lam) U^T + res (I - U U^T)`. All pages have the same size, so ranking by `est` ranks by estimated attention
  mass. `sig` (default 0.5) damps the covariance term; at `sig = 1` it is the exact value for Gaussian keys.
- `ub` is Quest's bound: no key in the page can have a larger logit. Taking the minimum keeps a page with a large spread
  from outranking pages whose keys really score high.
- The query heads of a GQA group share one selection (max over the group), so each KV head reads each page once.
- `cov=False` drops the covariance term (mean + bound, the "bound" ablation); `cov=False, bound=False` is the mean alone.

The top `m = min(pages, max(1, ceil(rho * pages)))` pages per KV head are opened (page 0 counts toward `m`), read in
bf16 and merged in one softmax with the exact recent window (between `window` and `window + 127` tokens, depending on
where the last page boundary falls). Everything else is skipped.

## Results

All measurements on a **Jetson AGX Orin 64 GB, power mode MODE_50W**, batch 1. Absolute times are specific to this
board and power mode.

### Latency per attention call (bf16 pages)

`experiments/bench.py`: synthetic caches with topic means drifting every 512 tokens, 4 KV heads x 4 query heads (the
Qwen3.5-9B layout), head_dim 256, median of 30 runs. "All pages" is `rho = 1`: the same score, select, pass and combine
kernels with every page opened, so it includes the scoring overhead. "Graph" is the time per call inside a CUDA graph
of 8 calls (the 9B's 8 full-attention layers).

| context | all pages (graph) | **10% (graph)** | 20% (graph) | all pages (eager) | 10% (eager) | 20% (eager) | bits / value / step at 10% / 20% / all |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32K | 1.94 ms | **0.36 ms** (5.4x) | 0.65 ms (3.0x) | 2.00 ms | 0.49 ms | 0.65 ms | 2.00 / 3.56 / 16.31 |
| 128K | 5.57 ms | **1.13 ms** (4.9x) | 1.89 ms (3.0x) | 8.16 ms | 1.15 ms | 1.81 ms | 1.94 / 3.53 / 16.31 |
| 256K | 11.12 ms | **2.11 ms** (5.3x) | 2.97 ms (3.7x) | 15.90 ms | 2.06 ms | 3.43 ms | 1.92 / 3.52 / 16.31 |

<sub>Source: [`results/bench_h4_d256.json`](results/bench_h4_d256.json) (fields `ms_graph_per_call`, `ms_eager`,
`read_bits`; speedups are against "all pages" in the graph). Bits count K and V of the opened pages and the window in
bf16, plus every page's summaries, per cached value. Kernel output against the PyTorch reference: relative difference
1.2e-6 to 1.3e-6 on every row (`out_rel_diff_vs_ref`). The file was written before the rename; it has no scheme
names.</sub>

At 10% the call costs about a fifth of the all-pages call, not a tenth: every page is still scored, and the sink page
and the recent window are always read.

### Quality: the design study (drop vs estimate)

Before the kernels, a Barnes-Hut-style tree over the keys (`src/sparsepage/research/tree.py`) tested whether the mass of
the unread keys has to be estimated. Protocol: Qwen3.5-9B, exact prefill, 256 tokens teacher-forced through the cache,
prose (WikiText-2) and code (`transformers` sources), 3 documents per length (64K: 2 prose documents); ΔPPL against
exact attention. Selection here is over **64-token k-means nodes** (clustered in the query metric), not contiguous
pages. "Compute vs dense" counts exact tokens plus node evaluations in token equivalents.

| scheme | compute vs dense | 16K | 32K | 64K |
|---|---:|---:|---:|---:|
| tree, open 10%, unopened nodes estimated | 0.17 | +0.29% | -0.10% | -0.19% |
| tree, open 20%, unopened estimated | 0.27 | +0.25% | +0.06% | +0.02% |
| tree, two-level (children of the 6 heaviest parents), open 20%, estimated | 0.23 | +0.46% | -0.07% | - |
| **open 20%, unopened dropped** | 0.27 | **+0.02%** | **-0.31%** | **-0.02%** |
| Quest, 5% of pages (min/max bound only) | 0.18 | +1.15% | +0.60% | +0.32% |
| Quest, 15% of pages | 0.28 | +0.39% | -0.34% | -0.05% |
| FoveaKV, 5% hot (reads every token at 2 bits; reference) | 1.00 | +0.34% | +0.18% | +0.39% |

<sub>Source: [`results/research/tree_ppl_qwen3.5-9b.jsonl`](results/research/tree_ppl_qwen3.5-9b.jsonl), mean over
documents of `exp(nll - nll_bf16) - 1`. The file was produced in FoveaKV's harness before the rename and uses that
harness's names (`tree-flat-r10`, `tree-flat-r20`, `tree-pp6-r20`, `drop-flat-r20`, `quest5`, `quest15`, `dyn5`); see
[`results/README.md`](results/README.md). With 2-3 documents per cell, differences of a few tenths of a percent are
within noise.</sub>

Estimating the unopened mass buys nothing over dropping it, so SparsePage drops it. That removes the value summaries
and the work on unopened nodes, and leaves one exact softmax over the opened pages and the window.

### In HgRT: the same scoring on FoveaKV pages

With the same score / select / combine kernels on contiguous FoveaKV 4-bit pages (HgRT's stack, synthetic caches, 4 KV
heads, CUDA graph of 8 calls), an attention call takes **0.36 / 0.79 / 1.45 ms** at 32K / 128K / 256K at 10% (HgRT
reports 2.57 ms at 256K in a rerun), against 5.5 / 21.5 / 42.3 ms for an exact bf16 kernel. On Qwen3.5-9B (16K and
32K, prose and code, 3 documents each) the stack gives ΔPPL **+0.21% at 20%** (1.27 bits read per value per step) and
**+0.63% at 10%** (0.86 bits). At 10%, the covariance + bound score is slightly better than the bound alone (+0.85%),
but the confidence intervals overlap.

<sub>Source: the previous version of this README and the [HgRT README](https://github.com/Minerva-Laboratories/HgRT)
(`experiments/stack_bench.py`, `experiments/stack_ppl.py`, `results/qwen3.5-9b/stack_ppl.jsonl` in that repository).
The exact bf16 kernel there is a different kernel from the "all pages" column above.</sub>

### Not measured yet

End-to-end perplexity of SparsePage on **bf16 pages** (`experiments/eval_ppl.py`: exact, 5 / 10 / 20%, and the
bound-only and mean-only ablations at 10%) has not been recorded. Its output will go to `results/qwen3.5-9b/ppl.jsonl`.
Until then, the quality evidence is the design study above (k-means nodes, not pages) and the HgRT stack (FoveaKV
pages, not bf16).

## Quickstart

```bash
git clone https://github.com/Minerva-Laboratories/SparsePage && cd SparsePage
pip install -e .                 # torch, numpy; the kernels build on first use (nvcc matching torch's CUDA)
pip install -e ".[eval,test]"    # + transformers, pyarrow (Qwen3.5 harness) and pytest
```

One `SparsePageCache` per full-attention layer:

```python
import torch
from sparsepage import SparsePageCache

Hkv, D = 4, 256                                         # Qwen3.5-9B: 4 KV heads, 16 query heads, head_dim 256
c = SparsePageCache(Hkv, Tmax=262144, head_dim=D, rho=0.1)

c.prefill(k, v)          # k, v: (T, Hkv, D) post-RoPE keys and values of the prompt
c.append(k1, v1)         # (1, Hkv, D) per decoded token
out = c.attend(q)        # q: (Hkv * 4, D) post-RoPE, unscaled -> (Hkv * 4, D) float32
```

The full signature (`src/sparsepage/cache.py`):

```python
SparsePageCache(Hkv, Tmax, head_dim=256, rho=0.1, cov=True, bound=True, sig=0.5, window=128,
                device="cuda", use_kernel=None)
```

| argument | meaning |
|---|---|
| `Hkv`, `Tmax`, `head_dim` | KV heads, capacity in tokens (rounded up to whole pages), head dimension (kernels: 128 or 256) |
| `rho` | fraction of summarized pages opened per step; `1.0` opens all pages (exact attention) |
| `cov`, `bound`, `sig` | the score's covariance term, the Quest bound, and the damping of the covariance term |
| `window` | minimum exact recent window in tokens (a multiple of 128) |
| `device`, `use_kernel` | `use_kernel=None` uses the CUDA kernels on a CUDA device and the PyTorch reference elsewhere |

Other entry points: `attend_kernel(q)` and `attend_ref(q)` (force one path), `page_scores(q)` (the `(Hkv, pages)`
scores), `n_open()`, `read_bits()` (bits read per cached value per step), `last_sel` (pages chosen by the last
reference call). The package also exports `summarize`, `page_scores`, `top_dirs`, `tokens_per_block` and the constants
`PAGE = 128`, `RK = 2`, `SIG = 0.5`, `G = 4`.

**Qwen3.5 end to end.** `sparsepage.qwen.Runner` runs Qwen3.5's layers directly (Gated DeltaNet + full attention), with
any object that has `prefill(k, v)`, `append(k, v)` and `attend(q)` as the attention cache. The model must be a local
directory or already in the Hugging Face hub cache (it is not downloaded); WikiText-2 is read from the hub cache too.

```bash
python experiments/bench.py       # latency, writes results/bench_h4_d256.json
python experiments/eval_ppl.py    # perplexity vs exact attention, writes results/qwen3.5-9b/ppl.jsonl (resumable)
```

## How the kernels work

`src/sparsepage/csrc/sparsepage.cu`, built once per head_dim by `torch.utils.cpp_extension.load`
(`src/sparsepage/ext.py`).

| kernel | launch | what it does |
|---|---|---|
| `score` | (≤ 48, Hkv) blocks of 4 warps | warps grid-stride over pages; each warp keeps its KV head's 4 queries and their norms in registers, so the work per page is five vector loads per lane (`mu`, `U_1`, `U_2`, `mx`, `mn`, fp16) and warp reductions. Writes one fp32 score per page, `+inf` for page 0. |
| `select` | Hkv blocks of 1024 threads | top-m per KV head by radix select: scores mapped to order-preserving 32-bit keys in shared memory, four rounds of 8-bit histograms find the m-th key, then a compaction writes exactly m page indices (ties included up to m). No sort, no host sync. |
| `pass` | (blocks, Hkv) of 4 warps | split-K flash-decoding over bf16 K/V, over a token range (the window) or a page list (the selection). Lane-per-token logits over 32-token tiles for the 4 query heads at once, online softmax, values accumulated across dimensions in parallel; writes partial (max, sum, acc) per block. |
| `combine` | Hkv·4 blocks of D threads | merges the two partial sets (opened pages, window) into the output; the first set may be in a transformed domain and is rotated back by an optional `R` (`acc @ R`). |

**Graph capture.** No kernel reads a value back to the host. `SparsePageCache.attend_kernel` passes the page count and
`m` from the host, so a graph captured from it is valid for the cache length at capture time (this is what
`experiments/bench.py` and the graph test replay). For a graph that stays valid while the cache grows, `score` and
`select` also take the page count as a device tensor (`npg_dev`); `select` then derives `m = ceil(rho * npg)` on the
GPU and publishes the `m * 128` selected tokens (`ntok_dev`) for the paged pass. HgRT's decode step uses this path.

**Quantized pages.** `score` takes the query twice: `q_s` in the domain of the summaries (`mu`, `U`) and `q_b` in the
domain of the bound (`mx`, `mn`). They are the same tensor on bf16 pages; on rotated pages (FoveaKV) they differ, and
`combine`'s `R` rotates the paged result back.

**Summaries.** Pages are summarized in PyTorch when they leave the window (`summarize`). The covariance directions use
subspace iteration from a cached seeded start with a rank-2 Gram-Schmidt instead of `torch.linalg.qr` (same subspace;
column signs may differ, and scores never see them), so no cuSOLVER call is needed and the summaries can be captured in
a graph too.

## Environment variables

| variable | used by | default |
|---|---|---|
| `SPARSEPAGE_BUILD_DIR` | `ext.py`: where the JIT-built kernels are cached | `~/.cache/sparsepage` |
| `TORCH_CUDA_ARCH_LIST` | `ext.py`: target architecture; if unset, set to the local GPU's compute capability | local GPU |
| `HF_HUB_CACHE` | `qwen.py`: where models and WikiText-2 are looked up | `~/.cache/huggingface/hub` |
| `SPARSEPAGE_MODEL` | `eval_ppl.py` | `Qwen/Qwen3.5-9B` |
| `SPARSEPAGE_LENS`, `SPARSEPAGE_NDOC`, `SPARSEPAGE_N` | `eval_ppl.py`: context lengths, documents per length, decoded tokens | `16384,32768`, `3`, `256` |
| `SPARSEPAGE_SRCS` | `eval_ppl.py`: text sources | `prose,code` |
| `SPARSEPAGE_SCHEMES` | `eval_ppl.py`: schemes to run (also available: `tree-flat-r10`, `tree-drop-r20`) | `exact,sparsepage5,sparsepage10,sparsepage20,sparsepage10-bound,sparsepage10-mean` |
| `SPARSEPAGE_OUT` | `eval_ppl.py`: output file (appended to, resumable) | `results/<model>/ppl.jsonl` |
| `BENCH_HKV`, `BENCH_D` | `bench.py`: KV heads, head_dim | `4`, `256` |
| `BENCH_CTX`, `BENCH_RHO` | `bench.py`: context lengths, opened fractions | `32768,131072,262144`, `0.1,0.2` |
| `BENCH_OUT` | `bench.py`: output file | `results/bench_h<HKV>_d<D>.json` |
| `TITLE_CHROME` | `scripts/make_title.py`: headless Chromium for the PNG title art | Playwright's, else `google-chrome` |

## Repository layout

```text
src/sparsepage/
  csrc/sparsepage.cu     score, select, paged pass, combine (CUDA)
  ext.py                 JIT build of the kernels, once per head_dim
  summaries.py           page summaries and the reference score
  cache.py               SparsePageCache: kernel path + PyTorch reference
  qwen.py                Qwen3.5 decode harness (any cache in the loop), prose and code texts
  research/tree.py       Barnes-Hut tree over the keys: the design study (PyTorch)
experiments/
  bench.py               latency per attention call, eager and in a CUDA graph
  eval_ppl.py            end-to-end perplexity on Qwen3.5 against exact attention
results/                 raw outputs (see results/README.md)
tests/test_sparsepage.py CPU tests; -m gpu: kernels vs reference, graph capture
scripts/make_title.py    the title artwork in docs/assets/
```

## Testing

```bash
python -m pytest -q -m "not gpu"   # 8 CPU tests
python -m pytest -q                # + 9 GPU tests (build the kernels)
```

The CPU tests check that subspace iteration recovers a planted subspace, that the mean-only score equals the page's
mean logit and the bound is never exceeded, that `rho = 1` equals dense attention, that token-by-token appends match a
single prefill, the window and open-page counts, and that the research tree with everything opened is exact. The GPU
tests compare the kernels' scores, selected page sets and outputs with the reference (head_dim 128 and 256), and replay
a captured CUDA graph.

## Related work and what is new

Page selection by summaries is established: Quest (arXiv:2406.10774; min/max bound, top pages), ArkVale (page digests,
NeurIPS 2024), InfLLM, LServe (arXiv:2502.14866; hierarchical query-centric page selection with fused kernels), and
cluster-based variants that also approximate the unselected keys: Multipole Attention (arXiv:2506.13059), RetroInfer
(arXiv:2505.02922), ClusterKV, MagicPIG.

What SparsePage adds:

- the score combines a second-cumulant (covariance) estimate of each page's log-mass with Quest's bound;
- the unselected pages are dropped, which the design study above measured to lose nothing against estimating them;
- the whole step is graph-capturable, with selection on the GPU and optional device-side counts;
- the kernels compose with quantized, rotated pages (FoveaKV in HgRT).

The gain of the covariance term over the bound or the mean alone is small: at 10% it is slightly better, but the
confidence intervals overlap.

## Related repositories

| repository | role |
|---|---|
| [HgRT](https://github.com/Minerva-Laboratories/HgRT) | the long-context stack: DepthSkip prefill, then FoveaKV pages + SparsePage selection in one CUDA-graph decode step; includes this repository as `third_party/sparsepage` |
| [FoveaKV](https://github.com/Minerva-Laboratories/FoveaKV) | 2-bit KV cache with 2-bit residuals for the attended tokens; the SparsePage kernels are adapted from its decode kernels, and the tree study ran in its harness |
| [DepthSkip](https://github.com/Minerva-Laboratories/DepthSkip) | training-free layer skipping for the prompt pass |

## Limitations

- **Shapes.** Batch 1, GQA group 4 (the Qwen3.5 family; `G = 4` is fixed in the kernels and the cache), head_dim 128 or
  256, bf16 pages of 128 tokens.
- **Quality on bf16 pages** is not measured end to end yet (see [Not measured yet](#not-measured-yet)).
- **One board.** All numbers are from a Jetson AGX Orin at MODE_50W; latencies are single attention calls on synthetic
  caches, not full decode steps.
- **Dropped mass is gone.** Pages outside the selection contribute nothing. This was harmless in the tests above
  (perplexity on prose and code); tasks that need many scattered tokens at once may need a larger `rho`.
- **Summaries run in PyTorch**, not in a fused kernel, and the cache's write path decides on the host when a page is
  summarized.

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The page-selection and paged-attention CUDA kernels are
adapted from the FoveaKV decode kernels (Minerva Laboratories), and the Qwen3.5 harness from FoveaKV's evaluation
harness.
