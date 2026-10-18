"""End-to-end decode on Qwen3.5 (hybrid Gated DeltaNet + full attention) with any attention cache in the loop.

Protocol: exact (bf16) chunked prefill of the context -> snapshot raw K/V of the full-attention layers, the DeltaNet
states and the prefill's query metric -> per cache type: build the cache from the snapshot, then decode token by token
(teacher-forced, for perplexity). A cache is any object with prefill(k, v) [(T, Hkv, D)], append(k, v) [(1, Hkv, D)]
and attend(q) [(Hq, D) post-RoPE, unscaled -> (Hq, D)]; optional after_prefill(qm) receives E[q q^T] per KV head.

Adapted from the FoveaKV evaluation harness (gpu/eval_e2e.py, Minerva Laboratories)."""
import glob
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

DEV = "cuda"


class LinCache:
    """Minimal cache for the Gated DeltaNet layers (the API their forward uses)."""

    def __init__(self, layer_ids):
        self.layers = {i: SimpleNamespace(conv_states=[None], recurrent_states=[None], record_past=False) for i in layer_ids}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return self.layers[layer_idx].recurrent_states[0] is not None

    def update_conv_state(self, s, layer_idx, conv_kernel_size=None, **kw):
        lay = self.layers[layer_idx]
        if conv_kernel_size is None:                               # older transformers API
            lay.conv_states[0] = s
            return s
        full = s if lay.conv_states[0] is None else torch.cat([lay.conv_states[0], s], -1)
        if full.shape[-1] < conv_kernel_size:
            full = F.pad(full, (conv_kernel_size - full.shape[-1], 0))
        lay.conv_states[0] = full[..., -conv_kernel_size:].contiguous()
        return full

    def update_recurrent_state(self, s, layer_idx, **kw):
        self.layers[layer_idx].recurrent_states[0] = s

    def clone(self):
        c = LinCache(list(self.layers))
        for i, lay in self.layers.items():
            c.layers[i].conv_states[0] = lay.conv_states[0].clone()
            c.layers[i].recurrent_states[0] = lay.recurrent_states[0].clone()
        return c


class _Recorder:
    """prefill-only: raw K/V and E[q q^T] per KV head (every 4th query)."""

    def __init__(self, Hkv, Tmax, D):
        self.kraw = torch.zeros(Tmax, Hkv, D, dtype=torch.bfloat16, device=DEV)
        self.vraw = torch.zeros_like(self.kraw)
        self.n, self.Hkv, self.qq, self.nq = 0, Hkv, None, 0

    def store(self, k, v, q):
        t = k.shape[0]
        self.kraw[self.n:self.n + t], self.vraw[self.n:self.n + t] = k, v
        self.n += t
        x = q[:, ::4].float().reshape(self.Hkv, -1, q.shape[-1])
        a = x.transpose(1, 2) @ x
        self.qq = a if self.qq is None else self.qq + a
        self.nq += x.shape[1]


def resolve_model(path_or_id):
    """a local directory, or a Hugging Face repo id already in the local hub cache."""
    if os.path.isdir(path_or_id):
        return path_or_id
    hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    d = glob.glob(os.path.join(hub, "models--" + path_or_id.replace("/", "--"), "snapshots", "*"))
    if not d:
        raise FileNotFoundError(path_or_id)
    return d[0]


class Runner:
    def __init__(self, model="Qwen/Qwen3.5-9B"):
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        import transformers.models.qwen3_5.modeling_qwen3_5 as M
        path = resolve_model(model)
        self.m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.rope = M.apply_rotary_pos_emb
        cfg = self.m.config
        self.types = cfg.layer_types
        self.attn_ids = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.lin_ids = [i for i, t in enumerate(self.types) if t != "full_attention"]
        self.hd, self.Hkv = cfg.head_dim, cfg.num_key_value_heads

    def _attn(self, a, x, cos, sin, cache, prefill):
        B, T, _ = x.shape
        hd = self.hd
        q, gate = torch.chunk(a.q_proj(x).view(B, T, -1, 2 * hd), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        kk, vv = k[0].transpose(0, 1), v[0].transpose(0, 1)                    # (T, Hkv, D)
        if prefill:
            cache.store(kk.to(torch.bfloat16), vv.to(torch.bfloat16), q[0])
            K = cache.kraw[:cache.n].transpose(0, 1)[None]
            V = cache.vraw[:cache.n].transpose(0, 1)[None]
            o = F.scaled_dot_product_attention(q.to(torch.bfloat16), K, V, attn_mask=causal_lower_right(T, cache.n),
                                               enable_gqa=True)
            o = o.transpose(1, 2).reshape(B, T, -1)
        else:
            outs = []
            with torch.autocast("cuda", enabled=False):
                for t in range(T):
                    cache.append(kk[t:t + 1].float(), vv[t:t + 1].float())
                    outs.append(cache.attend(q[0, :, t].float()).reshape(1, -1))
            o = torch.stack(outs, 1).to(x.dtype)
        return a.o_proj(o * torch.sigmoid(gate))

    @torch.no_grad()
    def forward(self, ids, pos0, caches, lin, prefill):
        mm = self.m.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = mm.embed_tokens(ids[None])
            pos = torch.arange(pos0, pos0 + len(ids), device=DEV)[None]
            try:
                cos, sin = mm.rotary_emb(x, pos)
            except IndexError:                          # newer transformers: (3, B, T) multimodal position ids
                cos, sin = mm.rotary_emb(x, pos[None].expand(3, *pos.shape))
            h = x
            for i, layer in enumerate(mm.layers):
                y = layer.input_layernorm(h)
                if self.types[i] == "full_attention":
                    y = self._attn(layer.self_attn, y, cos, sin, caches[i], prefill)
                else:
                    y = layer.linear_attn(y, cache_params=lin)
                h = h + y
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            return mm.norm(h)[0]

    def logits(self, h):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.lm_head(h).float()

    @torch.no_grad()
    def prefill(self, ids, chunk=512):
        T = len(ids)
        rec = {i: _Recorder(self.Hkv, T, self.hd) for i in self.attn_ids}
        lin = LinCache(self.lin_ids)
        h = None
        for s in range(0, T, chunk):
            h = self.forward(ids[s:s + chunk], s, rec, lin, prefill=True)
        return dict(kv={i: (c.kraw[:T], c.vraw[:T]) for i, c in rec.items()}, lin=lin, T=T,
                    qm={i: c.qq / max(c.nq, 1) for i, c in rec.items()}, last_logits=self.logits(h[-1:])[0])

    def build(self, make_cache, snap, extra=512):
        caches = {}
        for i, (k, v) in snap["kv"].items():
            c = make_cache(self.Hkv, snap["T"] + extra)
            c.prefill(k, v)
            if hasattr(c, "after_prefill"):
                c.after_prefill(snap["qm"][i])
            caches[i] = c
        return caches

    @torch.no_grad()
    def decode_teacher(self, make_cache, snap, targets):
        """targets[:-1] fed one at a time after the prefix; NLL of targets (the first from the prefill's logits)."""
        caches = self.build(make_cache, snap)
        lin = snap["lin"].clone()
        nll = [float(-torch.log_softmax(snap["last_logits"], -1)[targets[0]])]
        pos = snap["T"]
        for t in range(len(targets) - 1):
            h = self.forward(targets[t:t + 1], pos, caches, lin, prefill=False)
            pos += 1
            nll.append(float(-torch.log_softmax(self.logits(h[-1:])[0], -1)[targets[t + 1]]))
        stats = {k: float(np.mean([getattr(c, k)() for c in caches.values()])) for k in ("read_bits", "compute")
                 if all(hasattr(c, k) for c in caches.values())}
        del caches
        return np.array(nll), stats


# ------------------------------------------------------------------------------------------------- texts
def prose_text():
    """WikiText-2 train (raw) from the local Hugging Face datasets cache."""
    out = os.path.join(os.path.expanduser("~/.cache/riffle"), "wikitext2_train.txt")
    if not os.path.exists(out):
        import pyarrow.parquet as pq
        hub = os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
        f = glob.glob(os.path.join(hub, "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-2-raw-v1", "train-*.parquet"))
        if not f:
            raise FileNotFoundError("huggingface-cli download --repo-type dataset Salesforce/wikitext")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w").write("".join(pq.read_table(f[0]).column("text").to_pylist()))
    return open(out).read()


def code_text():
    """Python source of 120 transformers model files (~2M tokens)."""
    import transformers
    base = os.path.join(os.path.dirname(transformers.__file__), "models")
    files = sorted(glob.glob(os.path.join(base, "*", "modeling_*.py")))
    random.Random(0).shuffle(files)
    return "".join(f"# file: {os.path.relpath(f, base)}\n" + open(f).read() + "\n" for f in files[:120])


def doc_windows(tok, text, L, n_docs, seed):
    ids = tok(text[: 5 * L * (n_docs + 2)], add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(ids) - L), n_docs))
    return [ids[s:s + L] for s in starts]
