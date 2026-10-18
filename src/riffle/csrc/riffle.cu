// Riffle decode kernels (batch 1, GQA group G = 4, head_dim D = RF_D in {128, 256}).
//
//   score   : one warp per page; score of a page for a KV head = max over its G query heads of
//             min(q_s . mu + sig/2 (sum_r lam_r (q_s . U_r)^2 + res (|q_s|^2 - sum_r (q_s . U_r)^2)),   [cov]
//                 sum_d max(q_b,d mx_d, q_b,d mn_d))                                                  [bound]
//             q_s lives in the summaries' domain (mu, U), q_b in the bound's domain (mx, mn); they are the same tensor
//             unless the pages are stored in a transformed (e.g. rotated) domain. Page 0 (attention sinks) = +inf.
//   select  : top-m pages per KV head, radix select (4 x 8-bit digits) in shared memory + compaction; no host sync.
//   pass    : split-K flash-decoding over bf16 K/V (Hkv, Tcap, D): token range [t0, t0+n) or a page list
//             pages[h][i] (n = 128 m). Lane-per-token logits over 32-token tiles, dims-parallel values.
//             Writes partials (m, l, acc) per (block, h, g).
//   combine : merges two partial sets (pages, window); the first may be in a transformed domain (acc @ R), R optional.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

#ifndef RF_D
#define RF_D 256
#endif
constexpr int D = RF_D;
static_assert(D == 128 || D == 256, "head_dim must be 128 or 256");
constexpr int G = 4;
constexpr int LD = D / 32;
constexpr int WARPS = 4;
constexpr int TILE = 32;
constexpr int PAGE = 128;
constexpr int RK = 2;

template <typename T> __device__ __forceinline__ void load_lane16(const T* src, T* dst) {
  if constexpr (LD == 8) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
  else *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(src);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o);
  return x;
}

// ------------------------------------------------------------------------------------------- page scores
__global__ void score_k(const float* qs, const float* qb, const __half* mu, const __half* U, const float* lamres,
                        const __half* kmx, const __half* kmn, int npg, int npg_cap, int cov, int bound, float sig,
                        float* scores, const int* npg_dev) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int pg = blockIdx.x * WARPS + w;
  if (npg_dev) npg = *npg_dev;
  if (pg >= npg) return;
  const size_t prow = (size_t)h * npg_cap + pg;
  float ql[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) ql[g][i] = qs[(h * G + g) * D + lane * LD + i];
  float a[G], z0[G], z1[G], qn[G], ub[G];
  __half m[LD];
  load_lane16(mu + prow * D + lane * LD, m);
#pragma unroll
  for (int g = 0; g < G; g++) {
    a[g] = 0.f; z0[g] = 0.f; z1[g] = 0.f; qn[g] = 0.f; ub[g] = 0.f;
#pragma unroll
    for (int i = 0; i < LD; i++) a[g] += ql[g][i] * __half2float(m[i]);
  }
  if (cov) {
    __half u0[LD], u1[LD];
    load_lane16(U + (prow * RK + 0) * D + lane * LD, u0);
    load_lane16(U + (prow * RK + 1) * D + lane * LD, u1);
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int i = 0; i < LD; i++) {
        z0[g] += ql[g][i] * __half2float(u0[i]); z1[g] += ql[g][i] * __half2float(u1[i]); qn[g] += ql[g][i] * ql[g][i];
      }
  }
  if (bound) {
    __half mx[LD], mn[LD];
    load_lane16(kmx + prow * D + lane * LD, mx);
    load_lane16(kmn + prow * D + lane * LD, mn);
#pragma unroll
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int i = 0; i < LD; i++) {
        const float qv = qb[(h * G + g) * D + lane * LD + i];
        ub[g] += fmaxf(qv * __half2float(mx[i]), qv * __half2float(mn[i]));
      }
  }
  float best = -INFINITY;
#pragma unroll
  for (int g = 0; g < G; g++) {
    float est = warp_sum(a[g]);
    if (cov) {
      const float y0 = warp_sum(z0[g]), y1 = warp_sum(z1[g]), nn = warp_sum(qn[g]);
      const float* lr = lamres + prow * (RK + 1);
      est += 0.5f * sig * (lr[0] * y0 * y0 + lr[1] * y1 * y1 + lr[2] * (nn - y0 * y0 - y1 * y1));
    }
    if (bound) est = fminf(est, warp_sum(ub[g]));
    best = fmaxf(best, est);
  }
  if (lane == 0) scores[prow] = (pg == 0) ? INFINITY : best;
}

// ------------------------------------------------------------------------------------------- top-m select
__device__ __forceinline__ unsigned fkey(float f) {
  const unsigned u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void select_k(const float* scores, int npg, int npg_cap, int m, int* sel, int sel_stride,
                         const int* npg_dev, double rho, int* ntok_dev) {
  extern __shared__ unsigned keys[];
  __shared__ unsigned hist[256];
  __shared__ unsigned s_prefix, s_mask, s_rem, s_cnt, s_eq;
  const int h = blockIdx.x;
  if (npg_dev) {                         // counts on the GPU: m = clamp(ceil(rho npg), 1, npg); publish m * PAGE tokens
    npg = *npg_dev;
    m = min(npg, max(1, (int)ceil(rho * (double)npg)));   // same as Python ceil(rho * npg)
    if (h == 0 && threadIdx.x == 0 && ntok_dev) *ntok_dev = m * PAGE;
  }
  for (int i = threadIdx.x; i < npg; i += blockDim.x) keys[i] = fkey(scores[(size_t)h * npg_cap + i]);
  if (threadIdx.x == 0) { s_prefix = 0u; s_mask = 0u; s_rem = (unsigned)m; s_cnt = 0u; s_eq = 0u; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0u;
    __syncthreads();
    const unsigned pre = s_prefix, msk = s_mask;
    for (int i = threadIdx.x; i < npg; i += blockDim.x)
      if ((keys[i] & msk) == pre) atomicAdd(&hist[(keys[i] >> shift) & 255u], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned above = 0u, rem = s_rem;
      int d = 255;
      for (; d > 0; d--) { if (above + hist[d] >= rem) break; above += hist[d]; }
      s_rem = rem - above;
      s_prefix = pre | ((unsigned)d << shift);
      s_mask = msk | (255u << shift);
    }
    __syncthreads();
  }
  const unsigned thr = s_prefix, ties = s_rem;
  for (int i = threadIdx.x; i < npg; i += blockDim.x) {
    const unsigned k = keys[i];
    if (k > thr) sel[h * sel_stride + atomicAdd(&s_cnt, 1u)] = i;
    else if (k == thr && atomicAdd(&s_eq, 1u) < ties) sel[h * sel_stride + atomicAdd(&s_cnt, 1u)] = i;
  }
}

// ------------------------------------------------------------------------------------------- bf16 pass
struct PassArgs {
  int Hkv, Tcap, t0, n, tokens_per_block;
  const int* pages; int pages_stride;
  const float* q;                              // (Hkv, G, D), pre-multiplied by the softmax scale
  const __nv_bfloat16 *kb, *vb;                // (Hkv, Tcap, D)
  float *pm, *pl, *pacc;                       // (nb, Hkv, G), (nb, Hkv, G, D)
};

__global__ void __launch_bounds__(32 * WARPS) pass_k(PassArgs p) {
  const int h = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  constexpr int QBYTES = D * 16, PBYTES = WARPS * TILE * 16, ACCBYTES = WARPS * G * D * 4;
  constexpr int SMEM = (QBYTES + PBYTES) > ACCBYTES ? (QBYTES + PBYTES) : ACCBYTES;
  __shared__ __align__(16) unsigned char smem[SMEM];
  float4* qw = reinterpret_cast<float4*>(smem);                                   // q of the 4 heads per dim
  float4 (*ps)[TILE] = reinterpret_cast<float4(*)[TILE]>(smem + QBYTES);           // probabilities per token
  if (w == 0)
#pragma unroll
    for (int i = 0; i < LD; i++) {
      const int d = lane * LD + i;
      qw[d] = make_float4(p.q[(h * G + 0) * D + d], p.q[(h * G + 1) * D + d], p.q[(h * G + 2) * D + d], p.q[(h * G + 3) * D + d]);
    }
  __syncthreads();
  float m_run[G], l_run[G], acc[G][LD];
#pragma unroll
  for (int g = 0; g < G; g++) { m_run[g] = -INFINITY; l_run[g] = 0.f;
#pragma unroll
    for (int i = 0; i < LD; i++) acc[g][i] = 0.f; }
  const int per_warp = (p.tokens_per_block + WARPS - 1) / WARPS;
  const int b0 = blockIdx.x * p.tokens_per_block + w * per_warp;
  const int b1 = min(min(b0 + per_warp, (int)(blockIdx.x + 1) * p.tokens_per_block), p.n);
  auto tok_of = [&](int it) -> int {
    if (p.pages) return p.pages[h * p.pages_stride + it / PAGE] * PAGE + (it % PAGE);
    return p.t0 + it;
  };
  for (int it0 = b0; it0 < b1; it0 += TILE) {
    const int it = it0 + lane;
    float s[G];
    if (it < b1) {
      const size_t row = (size_t)h * p.Tcap + tok_of(it);
      const uint4* kp = reinterpret_cast<const uint4*>(p.kb + row * D);
#pragma unroll
      for (int g = 0; g < G; g++) s[g] = 0.f;
#pragma unroll 4
      for (int c = 0; c < D / 8; c++) {
        const uint4 r = kp[c];
        const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&r);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float kv = __bfloat162float(bf[i]);
          const float4 qq = qw[c * 8 + i];
          s[0] += qq.x * kv; s[1] += qq.y * kv; s[2] += qq.z * kv; s[3] += qq.w * kv;
        }
      }
    } else {
#pragma unroll
      for (int g = 0; g < G; g++) s[g] = -INFINITY;
    }
    float pr[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
      float mt = s[g];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, o));
      const float mn = fmaxf(m_run[g], mt);
      const float sc = (mn == -INFINITY) ? 1.f : __expf(m_run[g] - mn);
      pr[g] = (s[g] == -INFINITY) ? 0.f : __expf(s[g] - mn);
      l_run[g] = l_run[g] * sc + warp_sum(pr[g]);
      m_run[g] = mn;
#pragma unroll
      for (int i = 0; i < LD; i++) acc[g][i] *= sc;
    }
    ps[w][lane] = make_float4(pr[0], pr[1], pr[2], pr[3]);
    __syncwarp();
    const int nt = min(TILE, b1 - it0);
    for (int j = 0; j < nt; j++) {
      const float4 pj = ps[w][j];
      __nv_bfloat16 b[LD];
      load_lane16(p.vb + ((size_t)h * p.Tcap + tok_of(it0 + j)) * D + lane * LD, b);
#pragma unroll
      for (int i = 0; i < LD; i++) {
        const float x = __bfloat162float(b[i]);
        acc[0][i] += pj.x * x; acc[1][i] += pj.y * x; acc[2][i] += pj.z * x; acc[3][i] += pj.w * x;
      }
    }
    __syncwarp();
  }
  __shared__ float sm[WARPS][G], sl[WARPS][G];
  __syncthreads();
  float (*sacc)[G][D] = reinterpret_cast<float(*)[G][D]>(smem);
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) { sm[w][g] = m_run[g]; sl[w][g] = l_run[g]; }
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int i = 0; i < LD; i++) sacc[w][g][lane * LD + i] = acc[g][i];
  __syncthreads();
  const size_t ob = (size_t)blockIdx.x * p.Hkv + h;
  for (int e = threadIdx.x; e < G * D; e += blockDim.x) {
    const int g = e / D, dd = e % D;
    float M = -INFINITY;
    for (int ww = 0; ww < WARPS; ww++) M = fmaxf(M, sm[ww][g]);
    float L = 0.f, A = 0.f;
    if (M > -INFINITY)
      for (int ww = 0; ww < WARPS; ww++) {
        if (sm[ww][g] == -INFINITY) continue;
        const float f = __expf(sm[ww][g] - M); L += sl[ww][g] * f; A += sacc[ww][g][dd] * f;
      }
    p.pacc[(ob * G + g) * D + dd] = A;
    if (dd == 0) { p.pm[ob * G + g] = M; p.pl[ob * G + g] = L; }
  }
}

// ------------------------------------------------------------------------------------------- combine
__global__ void combine_k(const float* pm1, const float* pl1, const float* pa1, int nb1,
                          const float* pm2, const float* pl2, const float* pa2, int nb2, int HG,
                          const float* R, float* out) {
  const int hg = blockIdx.x, dd = threadIdx.x;
  __shared__ float sa[D];
  float M = -INFINITY;
  for (int b = 0; b < nb1; b++) M = fmaxf(M, pm1[b * HG + hg]);
  for (int b = 0; b < nb2; b++) M = fmaxf(M, pm2[b * HG + hg]);
  float L = 0.f, A1 = 0.f, A2 = 0.f;
  for (int b = 0; b < nb1; b++) {
    const float f = (pm1[b * HG + hg] > -INFINITY) ? __expf(pm1[b * HG + hg] - M) : 0.f;
    L += pl1[b * HG + hg] * f; A1 += pa1[((size_t)b * HG + hg) * D + dd] * f;
  }
  for (int b = 0; b < nb2; b++) {
    const float f = (pm2[b * HG + hg] > -INFINITY) ? __expf(pm2[b * HG + hg] - M) : 0.f;
    L += pl2[b * HG + hg] * f; A2 += pa2[((size_t)b * HG + hg) * D + dd] * f;
  }
  float o = A2;
  if (R) {
    sa[dd] = A1;
    __syncthreads();
    for (int j = 0; j < D; j++) o += sa[j] * R[j * D + dd];
  } else {
    o += A1;
  }
  out[hg * D + dd] = o / L;
}

// ------------------------------------------------------------------------------------------- host
static const void* ptr(const c10::optional<torch::Tensor>& t) { return (t.has_value() && t->defined()) ? t->data_ptr() : nullptr; }

void score(torch::Tensor qs, torch::Tensor qb, torch::Tensor mu, torch::Tensor U, torch::Tensor lamres,
           torch::Tensor kmx, torch::Tensor kmn, int64_t npg, int64_t cov, int64_t bound, double sig, torch::Tensor scores,
           c10::optional<torch::Tensor> npg_dev) {
  const int Hkv = mu.size(0), npg_cap = mu.size(1);
  if (npg == 0) return;                  // with npg_dev, npg is the upper bound that sizes the grid
  score_k<<<dim3((npg + WARPS - 1) / WARPS, Hkv), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(
      qs.data_ptr<float>(), qb.data_ptr<float>(), (const __half*)mu.data_ptr(), (const __half*)U.data_ptr(),
      lamres.data_ptr<float>(), (const __half*)kmx.data_ptr(), (const __half*)kmn.data_ptr(), npg, npg_cap, cov, bound,
      (float)sig, scores.data_ptr<float>(), (const int*)ptr(npg_dev));
}

void select_pages(torch::Tensor scores, int64_t npg, int64_t m, torch::Tensor sel, c10::optional<torch::Tensor> npg_dev,
                  double rho, c10::optional<torch::Tensor> ntok_dev) {
  const int Hkv = scores.size(0), npg_cap = scores.size(1);
  const size_t smem = (size_t)npg * sizeof(unsigned);
  if (npg == 0 || m == 0) return;
  if (smem > 48 * 1024) cudaFuncSetAttribute(select_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  select_k<<<Hkv, 1024, smem, c10::cuda::getCurrentCUDAStream()>>>(scores.data_ptr<float>(), npg, npg_cap, m,
                                                                   sel.data_ptr<int>(), sel.size(1), (const int*)ptr(npg_dev),
                                                                   rho, (int*)ptr(ntok_dev));
}

void pass(torch::Tensor q, torch::Tensor kb, torch::Tensor vb, int64_t t0, int64_t n, c10::optional<torch::Tensor> pages,
          int64_t tokens_per_block, torch::Tensor pm, torch::Tensor pl, torch::Tensor pacc) {
  PassArgs p{};
  p.Hkv = kb.size(0); p.Tcap = kb.size(1); p.t0 = t0; p.n = n; p.tokens_per_block = tokens_per_block;
  p.pages = (const int*)ptr(pages); p.pages_stride = pages.has_value() ? pages->size(1) : 0;
  p.q = q.data_ptr<float>(); p.kb = (const __nv_bfloat16*)kb.data_ptr(); p.vb = (const __nv_bfloat16*)vb.data_ptr();
  p.pm = pm.data_ptr<float>(); p.pl = pl.data_ptr<float>(); p.pacc = pacc.data_ptr<float>();
  const int nb = (n + tokens_per_block - 1) / tokens_per_block;
  if (nb == 0) return;
  pass_k<<<dim3(nb, p.Hkv), 32 * WARPS, 0, c10::cuda::getCurrentCUDAStream()>>>(p);
}

void combine(torch::Tensor pm1, torch::Tensor pl1, torch::Tensor pa1, int64_t nb1,
             torch::Tensor pm2, torch::Tensor pl2, torch::Tensor pa2, int64_t nb2, c10::optional<torch::Tensor> R,
             torch::Tensor out) {
  const int HG = pm1.size(1) * pm1.size(2);
  combine_k<<<HG, D, 0, c10::cuda::getCurrentCUDAStream()>>>(pm1.data_ptr<float>(), pl1.data_ptr<float>(),
      pa1.data_ptr<float>(), nb1, pm2.data_ptr<float>(), pl2.data_ptr<float>(), pa2.data_ptr<float>(), nb2, HG,
      (const float*)ptr(R), out.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("score", &score); m.def("select", &select_pages); m.def("pass_", &pass); m.def("combine", &combine);
}
