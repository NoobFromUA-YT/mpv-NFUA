/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

// C interface to the TensorRT / CUDA graph half of vf_rife (rife_trt.cpp).

#ifndef MP_RIFE_TRT_H
#define MP_RIFE_TRT_H

#include <stdbool.h>
#include <stdint.h>

#include "rife_kernel_args.h"

#ifdef __cplusplus
extern "C" {
#endif

struct rife_trt;

enum rife_log_level {
    RIFE_LOG_ERR,
    RIFE_LOG_WARN,
    RIFE_LOG_INFO,
    RIFE_LOG_V,
    RIFE_LOG_DEBUG,
};

enum rife_shape_mode {
    // One small engine for all sizes: compiled once, loads in a fraction of a
    // second.
    RIFE_SHAPE_DYNAMIC,
    // An engine for exactly this size, compiled when first needed. No faster,
    // and TensorRT folds the full size warp grids into it: hundreds of MiB on
    // disk and, at 4K, twice the GPU memory. For experiments.
    RIFE_SHAPE_STATIC,
};

struct rife_trt_config {
    void *cuda_ctx;           // CUcontext the frames live in
    int w, h;                 // visible picture size
    // Picture rows left out at the top and bottom (even numbers). They are
    // not interpolated: the output gets the first frame's rows there.
    int trim_top, trim_bottom;
    int depth;                // significant bits per sample (8, 10, 12, 16)
    int bps;                  // bytes per sample (1 or 2)
    float kr, kb;             // luma coefficients of the YCbCr matrix
    bool full_range;
    bool cosited_x, cosited_y; // chroma siting
    int modulus;              // tensor size is padded to a multiple of this
    const char *onnx;         // 7 channel fp16 RIFE network (see TOOLS/rife)
    const char *engine_dir;   // where compiled engines are cached
    // RIFE_SHAPE_*: which kind of engine to use (see enum rife_shape_mode)
    int shape_mode;
    int opt_level;            // TensorRT builder optimization level, 0..5
    float scene_threshold;    // <= 0 disables scene cut detection
    bool use_graph;           // false: issue the same work without a CUDA graph
    // Split each launch into pieces of about this many ms of GPU time (0: one
    // piece). Between pieces the GPU can do other work, such as the VO drawing
    // the next frame; a long uninterrupted launch makes it miss refreshes.
    float slice_ms;
    // Pause between slices (ms). The GPU only turns to other work (the VO's
    // drawing) when CUDA leaves it idle for a moment; back to back, even short
    // slices run as one block.
    float slice_gap_ms;

    void (*log)(void *ctx, int level, const char *msg);
    // Called (from the creating thread) once it is known that no cached engine
    // can be used and TensorRT has to compile one, which takes a while.
    void (*on_build)(void *ctx);
    // Polled while compiling; returning true abandons the build.
    bool (*cancelled)(void *ctx);
    void *cb_ctx;
};

// Loads (or compiles and caches) the engine, loads the kernels and captures
// the CUDA graph. May be called from any thread. Returns NULL on failure.
struct rife_trt *rife_trt_create(const struct rife_trt_config *cfg);

// How many launches may be in flight (submitted, not yet waited for) at once.
#define RIFE_TRT_MAX_INFLIGHT 32

// Check whether f0 and f1 belong to different shots using the GPU scene-cut
// grid on a dedicated high-priority stream. Returns false on CUDA error.
bool rife_trt_check_cut(struct rife_trt *r, const struct rife_frame *f0,
                        const struct rife_frame *f1, bool *is_cut);

// Enqueue one graph launch that interpolates between f0 and f1 at timestep
// (0..1) into dst. Returns immediately; all three frames must stay untouched
// until the launch has finished (rife_trt_wait_ticket() or rife_trt_wait()).
// Launches run one after another, in the order submitted. Nonzero pair_id
// matching the previous submission reuses f0/f1 RGB in the input tensor and
// updates only timestep. *ticket identifies this launch.
bool rife_trt_submit(struct rife_trt *r, const struct rife_frame *f0,
                     const struct rife_frame *f1, const struct rife_frame *dst,
                     float timestep, uint64_t pair_id, uint64_t *ticket);

// Block until the launch with this ticket has finished. If gpu_ms is not NULL,
// it is set to the time the GPU spent on it.
bool rife_trt_wait_ticket(struct rife_trt *r, uint64_t ticket, double *gpu_ms);

// Block until everything submitted so far has finished on the GPU.
bool rife_trt_wait(struct rife_trt *r);

// Debugging aid, not used during playback: copy the scene detection result of
// the last finished launch to the host.
bool rife_trt_read_scd(struct rife_trt *r, struct rife_scd_result *out);

// Tensor size the picture is padded to.
void rife_trt_get_tensor_size(struct rife_trt *r, int *pw, int *ph);

void rife_trt_destroy(struct rife_trt *r);

#ifdef __cplusplus
}
#endif

#endif
