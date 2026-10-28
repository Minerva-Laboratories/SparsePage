# Results

Raw outputs of the scripts in `experiments/` and of the design study. Every number in the top-level README comes from
one of these files (or, where marked, from the HgRT repository). Measured on a Jetson AGX Orin 64 GB, power mode
MODE_50W.

| file | produced by | what it holds |
|---|---|---|
| [`bench_h4_d256.json`](bench_h4_d256.json) | `experiments/bench.py` (defaults) | latency per attention call, 4 KV heads, head_dim 256, synthetic caches at 32K / 128K / 256K; `rho` 1.0 (all pages, same kernels) / 0.1 / 0.2; eager and inside a CUDA graph of 8 calls; bits read per cached token per step; kernel output vs the PyTorch reference |
| [`research/tree_ppl_qwen3.5-9b.jsonl`](research/tree_ppl_qwen3.5-9b.jsonl) | the Barnes-Hut tree study (`src/sparsepage/research/tree.py`), run in FoveaKV's evaluation harness | end-to-end NLL on Qwen3.5-9B, one row per (source, length, document, scheme), with per-token NLL, `bits` and `compute` |

`experiments/eval_ppl.py` writes `results/<model>/ppl.jsonl` (for example `results/qwen3.5-9b/ppl.jsonl`). That run, SparsePage
on bf16 pages end to end, has not been recorded yet.

## Names in the raw files

Both files predate the rename of the project (it was first called Riffle) and are kept byte for byte as they were
written. `bench_h4_d256.json` has no scheme names (rows are keyed by `rho`). The tree study uses the names of FoveaKV's
harness, not the `sparsepage*` names of `experiments/eval_ppl.py`:

| raw `scheme` | meaning | README row |
|---|---|---|
| `bf16` | exact attention (the reference for every ΔPPL) | - |
| `tree-flat-r10`, `tree-flat-r20` | 64-token k-means nodes, every node scored, open 10% / 20% of the region's tokens, unopened nodes estimated from their summaries | tree, open 10% / 20%, unopened estimated |
| `drop-flat-r20` | same opening as `tree-flat-r20`, unopened nodes dropped (the choice SparsePage makes, on contiguous pages) | open 20%, unopened dropped |
| `tree-pp6-r20` | two-level tree: parents scored first, only the children of the 6 heaviest parents scored, open 20% | tree, two-level, open 20% |
| `quest5`, `quest15` | Quest: pages ranked by the min/max bound only, 5% / 15% opened | Quest 5% / 15% pages |
| `dyn5` | FoveaKV, 5% hot tokens (2-bit cache, 2-bit residuals for the hot tokens): a reference row from that harness | FoveaKV 5% hot |

ΔPPL in the README is the mean over documents of `exp(nll_scheme - nll_bf16) - 1`, per context length (prose and code
pooled; 64K has 2 prose documents only); `compute` is averaged over the same rows.
