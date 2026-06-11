#include "ggml-metal-device.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

// Mirrors the CUDA diffusion_dense_sample_kernel contract: per canvas row, argmax (bit-exact with
// the host worker), entropy of softmax(logits * inv_temp) via H = log Z - dot(e, z)/Z, and a
// multinomial draw picking the first vocab-order crossing of u[row] * Z. Differences from the host
// path are limited to FP reduction order (same tolerance contract as the CUDA kernel). Unlike CUDA
// there is no device kernel or scratch upload: the shared MTLBuffer rows are reduced in place on
// the CPU with vDSP/vvexpf, which also skips the per-step full-logits fetch in the caller.
// Validation (tensor type, contiguity, metal buffer, shared storage) happens in ggml-metal.cpp.
extern "C" bool ggml_backend_metal_diffusion_sample_impl(
        const float * base,
        const float * u_host,
        int         * argmax_host,
        float       * entropy_host,
        int         * sampled_host,
        int           n_tokens,
        int           n_vocab,
        float         inv_temp) {
    if (!base || !u_host || !argmax_host || !entropy_host || !sampled_host || n_tokens <= 0 || n_vocab <= 0) {
        return false;
    }

    const unsigned nth = std::max(1u, std::min(std::thread::hardware_concurrency(), (unsigned) n_tokens));

    auto worker = [&](int p0, int p1) {
        const int n    = n_vocab;
        const int blk  = 8192;
        const int nblk = (n + blk - 1) / blk;
        std::vector<float> zbuf((size_t) n), ebuf((size_t) n), bsum((size_t) nblk);
        for (int pos = p0; pos < p1; pos++) {
            const float * row = base + (size_t) pos * n_vocab;

            vDSP_vsmul(row, 1, &inv_temp, zbuf.data(), 1, (vDSP_Length) n);      // z = row * inv_temp
            float m; vDSP_Length amax;
            vDSP_maxvi(zbuf.data(), 1, &m, &amax, (vDSP_Length) n);              // m, argmax (first max)
            const float neg_m = -m;
            vDSP_vsadd(zbuf.data(), 1, &neg_m, zbuf.data(), 1, (vDSP_Length) n); // z -= m
            // floor z so -inf logits (masked tokens) give e = exp(-80) ~ 1.8e-35 instead of e = 0
            // with z = -inf, whose 0 * -inf product would turn the dot below into NaN entropy.
            const float z_floor = -80.0f;
            vDSP_vthr(zbuf.data(), 1, &z_floor, zbuf.data(), 1, (vDSP_Length) n);
            vvexpf(ebuf.data(), zbuf.data(), &n);                                // e = exp(z)

            // per-block partial sums serve both Z and the multinomial crossing search
            float Z = 0.0f;
            for (int ib = 0; ib < nblk; ib++) {
                const int b0 = ib * blk;
                const int bn = std::min(blk, n - b0);
                vDSP_sve(ebuf.data() + b0, 1, &bsum[(size_t) ib], (vDSP_Length) bn);
                Z += bsum[(size_t) ib];
            }
            float S;
            vDSP_dotpr(ebuf.data(), 1, zbuf.data(), 1, &S, (vDSP_Length) n);     // S = dot(e, z)

            argmax_host[pos]  = (int) amax;
            entropy_host[pos] = logf(Z) - S / Z;

            // multinomial: first v (vocab order) with cumsum(e) >= u*Z. Block sums skip ahead; if FP
            // reduction order makes a claimed crossing vanish inside a block, the sequential cum
            // carries into the next block instead of falling back, so the n_vocab-1 default only
            // remains when the cumulative sum truly never reaches the target.
            const float target = u_host[pos] * Z;
            int   sampled = n_vocab - 1;
            bool  picked  = false;
            float cum     = 0.0f;
            for (int ib = 0; ib < nblk && !picked; ib++) {
                const int b0 = ib * blk;
                const int bn = std::min(blk, n - b0);
                if (cum + bsum[(size_t) ib] < target) {
                    cum += bsum[(size_t) ib];
                    continue;
                }
                for (int v = b0; v < b0 + bn; v++) {
                    cum += ebuf[(size_t) v];
                    if (cum >= target) { sampled = v; picked = true; break; }
                }
            }
            sampled_host[pos] = sampled;
        }
    };

    std::vector<std::thread> pool;
    const int chunk = (n_tokens + (int) nth - 1) / (int) nth;
    for (unsigned ti = 0; ti < nth; ti++) {
        const int p0 = (int) ti * chunk;
        const int p1 = std::min(p0 + chunk, n_tokens);
        if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
    }
    for (auto & th : pool) { th.join(); }

    return true;
}
