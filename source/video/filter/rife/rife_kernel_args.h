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

// Argument blocks shared between the host (rife_trt.cpp) and the device
// (rife_kernels.cu). Every kernel takes exactly one of these by value, so a
// kernel node of the captured CUDA graph is retargeted by replacing a single
// parameter. Only fixed-size types, laid out without implicit padding.

#ifndef MP_RIFE_KERNEL_ARGS_H
#define MP_RIFE_KERNEL_ARGS_H

#include <stdint.h>

// Scene detection compares the two frames on a grid of this many cells.
#define RIFE_SCD_W 64
#define RIFE_SCD_H 36
#define RIFE_SCD_CELLS (RIFE_SCD_W * RIFE_SCD_H)

// One semi-planar 4:2:0 frame in device memory (NV12 / P010 / P012 / P016).
struct rife_frame {
    uint64_t y;        // CUdeviceptr of the luma plane
    uint64_t uv;       // CUdeviceptr of the interleaved chroma plane
    int32_t pitch_y;   // bytes per luma row
    int32_t pitch_uv;  // bytes per chroma row
};

// Everything that stays constant for one video format.
struct rife_geom {
    int32_t w, h;      // visible picture size
    int32_t pw, ph;    // tensor size, a multiple of the network's modulus
    // The tensor holds picture rows y0 .. y0 + th - 1 (both even when rows are
    // left out). If that is fewer than ph rows, the last one is repeated; the
    // same goes for columns beyond w.
    int32_t y0, th;
    int32_t cw, ch;    // chroma plane size
    int32_t bps;       // bytes per sample: 1 (NV12) or 2 (P0xx)
    int32_t shift;     // unused low bits of a 16 bit sample (6 for P010)
    int32_t maxval;    // largest code value, (1 << depth) - 1
    int32_t cosited_x; // chroma samples sit on even luma columns (not between)
    int32_t cosited_y; // chroma samples sit on even luma rows (not between)
    int32_t in_hwc8;   // nonzero: in_tensor uses TensorRT kHWC8 [H][W][8] layout
    // Code values to non-linear RGB in [0, 1] and back:
    //   Y' = (y - y_off) * y_mul          Cb = (u - c_off) * c_mul
    //   R = Y' + rv * Cr   G = Y' + gu * Cb + gv * Cr   B = Y' + bu * Cb
    //   Y' = kr * R + kg * G + kb * B     Cb = (B - Y') * cb_div
    float y_off, y_mul, c_off, c_mul;
    float rv, gu, gv, bu;
    float kr, kg, kb;
    float cb_div, cr_div;
    float inv_y, inv_c;
    int32_t out_hwc8;  // nonzero: out_tensor uses TensorRT kHWC8 [H][W][8] layout
};

// Written by the scene detection kernels, read by the host via scd_stream.
struct rife_scd_result {
    float diff;  // mean absolute luma difference of the grid, 0..1
    int32_t cut; // nonzero if the two frames belong to different shots
};

struct rife_pre_args {
    struct rife_frame f0, f1;
    struct rife_geom g;
    uint64_t tensor;       // fp16 NCHW [1, 7, ph, pw]: RGB of f0, RGB of f1, timestep
    float timestep;        // position of the wanted frame between f0 and f1, 0..1
    int32_t timestep_only; // nonzero: only update channel 6 (f0/f1 RGB already in tensor)
};

struct rife_scd_args {
    struct rife_frame f0, f1;
    struct rife_geom g;
    uint64_t cells;   // float[RIFE_SCD_CELLS]
    uint64_t result;  // struct rife_scd_result
    float threshold;  // diff above this is a cut; <= 0 disables detection
    float pad;
};

struct rife_post_args {
    struct rife_frame dst;
    // Repeated in the rows the tensor does not cover.
    struct rife_frame f0;
    struct rife_geom g;
    uint64_t tensor;  // fp16 NCHW [1, 3, ph, pw] RGB
};

#endif
