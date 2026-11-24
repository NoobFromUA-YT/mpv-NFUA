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

// Every use of TensorRT's C++ interface, behind plain C functions.
//
// TensorRT for Windows is built with MSVC, whose C++ ABI (virtual tables,
// how structs are returned) differs from the MinGW one mpv is built with.
// trt_shim.cpp is therefore compiled on its own for the MSVC ABI, free of any
// runtime library, and the rest of mpv only ever sees this header. Elsewhere
// it is an ordinary source file.

#ifndef MP_TRT_SHIM_H
#define MP_TRT_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// What the shim needs from its host. It has no library to call otherwise.
struct trt_shim_host {
    // TensorRT's exported entry points (createInferBuilder_INTERNAL etc.)
    void *(*create_builder)(void *logger, int32_t version);
    void *(*create_runtime)(void *logger, int32_t version);
    void *(*create_parser)(void *network, void *logger, int version);

    void *(*alloc)(size_t size); // zeroed
    void (*release)(void *ptr);
    // level: 0 error, 1 warning, 2 info, 3 verbose, 4 debug.
    // from_trt: the text comes from TensorRT rather than from the shim.
    void (*log)(void *ctx, int level, int from_trt, const char *msg);
    int (*cancelled)(void *ctx); // may be NULL
    void *ctx;
};

struct trt_shim_profile {
    int min_w, min_h, opt_w, opt_h, max_w, max_h;
};

struct trt_shim_build {
    const void *onnx;
    size_t onnx_size;
    // Profiles for the [1, 7, h, w] input the engine is to accept.
    struct trt_shim_profile profiles[4];
    int num_profiles;
    int opt_level;
    const void *timing_cache;
    size_t timing_cache_size;
};

// A compiled engine. Valid until trt_shim_plan_free().
struct trt_shim_plan {
    void *handle;
    const void *data;
    size_t size;
    void *tc_handle;
    const void *tc_data;
    size_t tc_size;
};

struct trt_shim_info {
    size_t mem_size;
    int in_hwc8;
    int out_hwc8;
};

struct trt_shim; // a loaded engine with one execution context

// Compile a network. Returns 0 on failure (already logged).
int trt_shim_compile(const struct trt_shim_host *host,
                     const struct trt_shim_build *build,
                     struct trt_shim_plan *out);
void trt_shim_plan_free(struct trt_shim_plan *plan);

// Load a compiled engine and set it up for [1, 7, h, w] input. A CUDA context
// must be current. *info receives activation size and tensor layout.
// NULL on failure (already logged).
struct trt_shim *trt_shim_load(const struct trt_shim_host *host, const void *plan,
                               size_t plan_size, int w, int h, void *stream,
                               struct trt_shim_info *info);

// Give the engine its activation memory and its two tensors (device pointers).
int trt_shim_bind(struct trt_shim *s, void *mem, size_t mem_size,
                  void *in_tensor, void *out_tensor);

// Issue one inference on a CUDA stream (CUstream).
int trt_shim_enqueue(struct trt_shim *s, void *stream);

void trt_shim_free(struct trt_shim *s);

// TensorRT version the shim was compiled against.
void trt_shim_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif

#endif
