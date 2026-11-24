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

// Device side of vf_rife. Compiled to PTX at build time and embedded in mpv.
//
// One CUDA graph launch runs, in this order:
//   rife_pre         NVDEC surfaces (4:2:0 code values) -> fp16 RGB tensor
//   rife_scd_cells   } is the second frame a different shot?
//   rife_scd_reduce  }
//   TensorRT         RIFE
//   rife_post        fp16 RGB tensor -> 4:2:0 output surface

#include <cuda_fp16.h>

#include "rife_kernel_args.h"

static __device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Code value of one luma sample.
static __device__ __forceinline__ float load(uint64_t plane, int pitch, int x,
                                             int y, const rife_geom &g)
{
    const unsigned char *row = (const unsigned char *)plane + (size_t)y * pitch;
    if (g.bps == 1)
        return (float)row[x];
    return (float)(((const unsigned short *)row)[x] >> g.shift);
}

// Vectorized load of one interleaved (U, V) chroma sample pair.
static __device__ __forceinline__ void load_uv(uint64_t plane, int pitch, int cx,
                                               int cy, const rife_geom &g,
                                               float &u, float &v)
{
    const unsigned char *row = (const unsigned char *)plane + (size_t)cy * pitch;
    if (g.bps == 1) {
        uchar2 uv = ((const uchar2 *)row)[cx];
        u = (float)uv.x;
        v = (float)uv.y;
    } else {
        ushort2 uv = ((const ushort2 *)row)[cx];
        u = (float)(uv.x >> g.shift);
        v = (float)(uv.y >> g.shift);
    }
}

static __device__ __forceinline__ void store(uint64_t plane, int pitch, int x,
                                             int y, const rife_geom &g, float v)
{
    int code = clampi((int)(v + 0.5f), 0, g.maxval);
    unsigned char *row = (unsigned char *)plane + (size_t)y * pitch;
    if (g.bps == 1) {
        row[x] = (unsigned char)code;
    } else {
        ((unsigned short *)row)[x] = (unsigned short)(code << g.shift);
    }
}

static __device__ __forceinline__ void store_pair(uint64_t plane, int pitch, int cx,
                                                  int y, const rife_geom &g,
                                                  float v0, float v1)
{
    int c0 = clampi((int)(v0 + 0.5f), 0, g.maxval);
    int c1 = clampi((int)(v1 + 0.5f), 0, g.maxval);
    unsigned char *row = (unsigned char *)plane + (size_t)y * pitch;
    if (g.bps == 1) {
        ((uchar2 *)row)[cx] = make_uchar2((unsigned char)c0, (unsigned char)c1);
    } else {
        ((ushort2 *)row)[cx] = make_ushort2((unsigned short)(c0 << g.shift),
                                            (unsigned short)(c1 << g.shift));
    }
}

// Non-linear RGB of luma position (x, y), chroma upsampled bilinearly from
// where the samples really are.
static __device__ __forceinline__ void load_rgb(const rife_frame &f,
                                                const rife_geom &g, int x, int y,
                                                float &r, float &gr, float &b)
{
    float luma = load(f.y, f.pitch_y, x, y, g);

    float fx = g.cosited_x ? x * 0.5f : x * 0.5f - 0.25f;
    float fy = g.cosited_y ? y * 0.5f : y * 0.5f - 0.25f;
    float flx = floorf(fx), fly = floorf(fy);
    float wx = fx - flx, wy = fy - fly;
    int x0 = clampi((int)flx, 0, g.cw - 1), x1 = clampi((int)flx + 1, 0, g.cw - 1);
    int y0 = clampi((int)fly, 0, g.ch - 1), y1 = clampi((int)fly + 1, 0, g.ch - 1);

    float u00, v00, u10, v10, u01, v01, u11, v11;
    load_uv(f.uv, f.pitch_uv, x0, y0, g, u00, v00);
    load_uv(f.uv, f.pitch_uv, x1, y0, g, u10, v10);
    load_uv(f.uv, f.pitch_uv, x0, y1, g, u01, v01);
    load_uv(f.uv, f.pitch_uv, x1, y1, g, u11, v11);

    float u = (u00 + (u10 - u00) * wx) * (1.0f - wy) + (u01 + (u11 - u01) * wx) * wy;
    float v = (v00 + (v10 - v00) * wx) * (1.0f - wy) + (v01 + (v11 - v01) * wx) * wy;

    float yn = (luma - g.y_off) * g.y_mul;
    float cb = (u - g.c_off) * g.c_mul;
    float cr = (v - g.c_off) * g.c_mul;
    r  = __saturatef(yn + g.rv * cr);
    gr = __saturatef(yn + g.gu * cb + g.gv * cr);
    b  = __saturatef(yn + g.bu * cb);
}

// One thread per tensor pixel. The padding repeats the picture's edge, which
// keeps the optical flow near the border sane.
extern "C" __global__ void rife_pre(rife_pre_args a)
{
    const rife_geom &g = a.g;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.pw || y >= g.ph)
        return;

    size_t idx = (size_t)y * g.pw + x;
    if (g.in_hwc8) {
        if (a.timestep_only) {
            ((__half *)a.tensor)[idx * 8 + 6] = __float2half_rn(a.timestep);
            return;
        }
        int sx = x < g.w ? x : g.w - 1;
        int sy = g.y0 + (y < g.th ? y : g.th - 1);
        float r0, g0, b0, r1, g1, b1;
        load_rgb(a.f0, g, sx, sy, r0, g0, b0);
        load_rgb(a.f1, g, sx, sy, r1, g1, b1);
        union {
            __half2 h[4];
            uint4 u;
        } pkt;
        pkt.h[0] = __floats2half2_rn(r0, g0);
        pkt.h[1] = __floats2half2_rn(b0, r1);
        pkt.h[2] = __floats2half2_rn(g1, b1);
        pkt.h[3] = __floats2half2_rn(a.timestep, 0.0f);
        ((uint4 *)a.tensor)[idx] = pkt.u;
        return;
    }

    size_t plane = (size_t)g.pw * g.ph;
    __half *t = (__half *)a.tensor + idx;
    t[6 * plane] = __float2half_rn(a.timestep);
    if (a.timestep_only)
        return;

    int sx = x < g.w ? x : g.w - 1;
    int sy = g.y0 + (y < g.th ? y : g.th - 1);

    float r, gr, b;
    load_rgb(a.f0, g, sx, sy, r, gr, b);
    t[0]         = __float2half_rn(r);
    t[plane]     = __float2half_rn(gr);
    t[2 * plane] = __float2half_rn(b);
    load_rgb(a.f1, g, sx, sy, r, gr, b);
    t[3 * plane] = __float2half_rn(r);
    t[4 * plane] = __float2half_rn(gr);
    t[5 * plane] = __float2half_rn(b);
}

// One thread per grid cell: difference of the two frames' mean luma there.
extern "C" __global__ void rife_scd_cells(rife_scd_args a)
{
    const rife_geom &g = a.g;
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= RIFE_SCD_CELLS)
        return;
    int cx = cell % RIFE_SCD_W, cy = cell / RIFE_SCD_W;

    int x0 = (int)((long long)cx * g.w / RIFE_SCD_W);
    int x1 = (int)((long long)(cx + 1) * g.w / RIFE_SCD_W);
    int y0 = g.y0 + (int)((long long)cy * g.th / RIFE_SCD_H);
    int y1 = g.y0 + (int)((long long)(cy + 1) * g.th / RIFE_SCD_H);
    if (x1 <= x0)
        x1 = x0 + 1;
    if (y1 <= y0)
        y1 = y0 + 1;
    // At most 8x8 taps per cell, spread evenly.
    int stepx = (x1 - x0 + 7) / 8, stepy = (y1 - y0 + 7) / 8;

    float s0 = 0.0f, s1 = 0.0f;
    int n = 0;
    for (int y = y0 + stepy / 2; y < y1; y += stepy) {
        int yy = y < g.y0 + g.th ? y : g.y0 + g.th - 1;
        for (int x = x0 + stepx / 2; x < x1; x += stepx) {
            int xx = x < g.w ? x : g.w - 1;
            s0 += load(a.f0.y, a.f0.pitch_y, xx, yy, g);
            s1 += load(a.f1.y, a.f1.pitch_y, xx, yy, g);
            n++;
        }
    }
    float d = n ? fabsf(s0 - s1) * g.y_mul / (float)n : 0.0f;
    ((float *)a.cells)[cell] = d;
}

// A single block: sum the grid and leave the verdict in device memory.
extern "C" __global__ void rife_scd_reduce(rife_scd_args a)
{
    __shared__ float part[256];
    int tid = threadIdx.x;
    const float *cells = (const float *)a.cells;

    float s = 0.0f;
    for (int i = tid; i < RIFE_SCD_CELLS; i += blockDim.x)
        s += cells[i];
    part[tid] = s;
    __syncthreads();

    for (int half = blockDim.x / 2; half >= 32; half >>= 1) {
        if (tid < half)
            part[tid] += part[tid + half];
        __syncthreads();
    }

    if (tid < 32) {
        float val = part[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, offset);
        if (tid == 0) {
            rife_scd_result *res = (rife_scd_result *)a.result;
            float diff = val / (float)RIFE_SCD_CELLS;
            res->diff = diff;
            res->cut = a.threshold > 0.0f && diff > a.threshold;
        }
    }
}

static __device__ __forceinline__ void load_out(const __half *t, size_t plane,
                                                const rife_geom &g, int x, int y,
                                                float &r, float &gr, float &b)
{
    if (g.out_hwc8) {
        union {
            uint2 u;
            __half2 h[2];
        } pkt;
        pkt.u = *(const uint2 *)(t + ((size_t)y * g.pw + x) * 8);
        float2 rg = __half22float2(pkt.h[0]);
        r  = rg.x;
        gr = rg.y;
        b  = __low2float(pkt.h[1]);
        return;
    }
    const __half *p = t + (size_t)y * g.pw + x;
    r  = __half2float(p[0]);
    gr = __half2float(p[plane]);
    b  = __half2float(p[2 * plane]);
}

// One thread per chroma sample: writes it and the (up to) four luma samples it
// covers. Chroma is filtered down to where its samples sit, mirroring rife_pre.
extern "C" __global__ void rife_post(rife_post_args a)
{
    const rife_geom &g = a.g;
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= g.cw || cy >= g.ch)
        return;

    bool outside = 2 * cy < g.y0 || 2 * cy >= g.y0 + g.th;
    if (outside) {
        // Rows left out of the tensor are not shown, and repeat f0 so they hold a picture.
        for (int dy = 0; dy < 2; dy++) {
            int y = 2 * cy + dy;
            if (y >= g.h)
                break;
            if (2 * cx + 1 < g.w) {
                if (g.bps == 1) {
                    ((uchar2 *)((unsigned char *)a.dst.y + (size_t)y * a.dst.pitch_y))[cx] =
                        ((const uchar2 *)((const unsigned char *)a.f0.y + (size_t)y * a.f0.pitch_y))[cx];
                } else {
                    ((ushort2 *)((unsigned char *)a.dst.y + (size_t)y * a.dst.pitch_y))[cx] =
                        ((const ushort2 *)((const unsigned char *)a.f0.y + (size_t)y * a.f0.pitch_y))[cx];
                }
            } else {
                int x = 2 * cx;
                if (g.bps == 1) {
                    ((unsigned char *)a.dst.y)[(size_t)y * a.dst.pitch_y + x] =
                        ((const unsigned char *)a.f0.y)[(size_t)y * a.f0.pitch_y + x];
                } else {
                    ((unsigned short *)((unsigned char *)a.dst.y + (size_t)y * a.dst.pitch_y))[x] =
                        ((const unsigned short *)((const unsigned char *)a.f0.y + (size_t)y * a.f0.pitch_y))[x];
                }
            }
        }
        if (g.bps == 1) {
            ((uchar2 *)((unsigned char *)a.dst.uv + (size_t)cy * a.dst.pitch_uv))[cx] =
                ((const uchar2 *)((const unsigned char *)a.f0.uv + (size_t)cy * a.f0.pitch_uv))[cx];
        } else {
            ((ushort2 *)((unsigned char *)a.dst.uv + (size_t)cy * a.dst.pitch_uv))[cx] =
                ((const ushort2 *)((const unsigned char *)a.f0.uv + (size_t)cy * a.f0.pitch_uv))[cx];
        }
        return;
    }

    const __half *t = (const __half *)a.tensor;
    size_t plane = (size_t)g.pw * g.ph;
    float lr[2][2] = {}, lg[2][2] = {}, lb[2][2] = {};

    for (int dy = 0; dy < 2; dy++) {
        int y = 2 * cy + dy;
        if (y >= g.h)
            break;
        float yn0 = 0.0f;
        for (int dx = 0; dx < 2; dx++) {
            int x = 2 * cx + dx;
            if (x >= g.w)
                break;
            float r, gr, b;
            load_out(t, plane, g, x, y - g.y0, r, gr, b);
            lr[dy][dx] = r;
            lg[dy][dx] = gr;
            lb[dy][dx] = b;
            float yn = (g.kr * r + g.kg * gr + g.kb * b) * g.inv_y + g.y_off;
            if (dx == 0) {
                yn0 = yn;
                if (2 * cx + 1 >= g.w)
                    store(a.dst.y, a.dst.pitch_y, x, y, g, yn);
            } else {
                store_pair(a.dst.y, a.dst.pitch_y, cx, y, g, yn0, yn);
            }
        }
    }

    // Undo rife_pre's chroma upsampling, reusing the 2x2 luma samples already
    // in registers when taps fall inside this chroma cell.
    // Co-sited horizontal: 3-tap [1 2 1] / 4 prevents horizontal chroma aliasing
    // on interpolated frames while preserving smooth edges.
    static const float cosited3[3] = {0.25f, 0.5f, 0.25f};
    static const float between[6] = {-1.0f / 12, -1.0f / 12, 8.0f / 12,
                                     8.0f / 12, -1.0f / 12, -1.0f / 12};
    int nx = g.cosited_x ? 3 : 6, ny = g.cosited_y ? 1 : 6;
    int bx = g.cosited_x ? 2 * cx - 1 : 2 * cx - 2;
    int by = g.cosited_y ? 2 * cy : 2 * cy - 2;
    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    for (int j = 0; j < ny; j++) {
        int sy = clampi(by + j, g.y0, g.y0 + g.th - 1);
        int dy = sy - 2 * cy;
        float wy = ny == 1 ? 1.0f : between[j];
        for (int i = 0; i < nx; i++) {
            int sx = clampi(bx + i, 0, g.w - 1);
            int dx = sx - 2 * cx;
            float wx = g.cosited_x ? cosited3[i] : between[i];
            float r, gr, b;
            if ((dy & ~1) == 0 && (dx & ~1) == 0 && sx < g.w && sy < g.h) {
                r  = lr[dy][dx];
                gr = lg[dy][dx];
                b  = lb[dy][dx];
            } else {
                load_out(t, plane, g, sx, sy - g.y0, r, gr, b);
            }
            float w = wx * wy;
            sr += r * w;
            sg += gr * w;
            sb += b * w;
        }
    }
    float yn = g.kr * sr + g.kg * sg + g.kb * sb;
    store_pair(a.dst.uv, a.dst.pitch_uv, cx, cy, g,
               (sb - yn) * g.cb_div * g.inv_c + g.c_off,
               (sr - yn) * g.cr_div * g.inv_c + g.c_off);
}
