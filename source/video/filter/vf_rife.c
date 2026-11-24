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

/*
 * RIFE frame interpolation on TensorRT.
 *
 * Takes the decoder's CUDA surfaces as they are (--hwdec=nvdec), and puts
 * multiplier - 1 new frames between each two of them. The frames never leave
 * the GPU: colour conversion, scene cut detection, the network and the
 * conversion back all run in one CUDA graph launch per new frame (see
 * rife/rife_trt.cpp), writing straight into a CUDA surface the VO maps.
 * Original frames are passed on untouched.
 *
 * Software decoded video works too, at the price of one upload per frame.
 */

#include <math.h>
#include <stdatomic.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixfmt.h>

#include "common/common.h"
#include "common/msg.h"
#include "filters/f_autoconvert.h"
#include "filters/filter.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"
#include "options/path.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/fmt-conversion.h"
#include "video/hwdec.h"
#include "video/img_format.h"
#include "video/mp_image.h"
#include "video/mp_image_pool.h"

#include "rife/rife_trt.h"

enum trim_mode {
    TRIM_NO,
    TRIM_NEAREST,
    TRIM_ALWAYS,
};

enum flow_mode {
    FLOW_AUTO,
    FLOW_FULL,
    FLOW_HALF,
};

struct rife_opts {
    int multiplier;
    double fps;
    int queue;
    double slice;
    double slice_gap;
    char *model;
    char *model_half;
    char *engine_dir;
    int flow;
    double full_flow_rate;
    float scene_threshold;
    int shape_mode;
    int align;
    int trim;
    int opt_level;
    bool graph;
    bool wait;
    double max_gap;
    bool stats;
};

// What an engine instance was set up for. A change of any of these means a
// new instance (and usually just a new graph capture, not a new engine).
struct fmt_key {
    void *cuda_ctx;
    int w, h;
    enum AVPixelFormat sw_format;
    enum pl_color_system sys;
    enum pl_color_levels levels;
    enum pl_chroma_location chroma;
    bool half;
    int trim_top, trim_bottom;
};

enum engine_state {
    ENGINE_NONE,
    ENGINE_LOADING,  // reading a cached engine: short, playback waits
    ENGINE_BUILDING, // TensorRT is compiling: long, playback goes on without us
    ENGINE_READY,
    ENGINE_FAILED,
};

struct priv {
    struct rife_opts *opts;
    struct mp_filter *f;
    struct mp_pin *in;

    AVBufferRef *upload_dev; // device for software decoded input
    AVBufferRef *upload_pool;
    AVBufferRef *out_pool;

    // Output frames in order, the interpolated ones already submitted to the
    // GPU. Frames are computed this far ahead of the player, so that a frame
    // which takes longer than one output interval (4K) is still ready in time.
    struct rife_job *jobs;
    int num_jobs;
    uint64_t pair_id;       // monotonic id for frame pairs (0 = none)
    struct mp_image *last;  // last input frame (its output is already queued)
    bool eof;               // input ended; send EOF once the queue is empty

    mp_mutex lock;
    mp_cond cond;
    mp_thread thread;
    bool thread_running;
    enum engine_state state; // (lock)
    struct rife_trt *trt;    // (lock) valid in ENGINE_READY
    atomic_bool cancel;
    struct fmt_key key;
    struct rife_trt_config cfg;
    char *cfg_onnx, *cfg_engine_dir;
    bool warned_format, warned_building;

    int64_t stat_frames, stat_last_report;
    double stat_gpu_ms, stat_wait_ms;
};

struct rife_job {
    struct mp_image *img;        // what goes out
    struct mp_image *src0, *src1; // inputs, held until the GPU is done
    bool interp;
    bool waited;                 // already waited on (e.g. before engine rebuild)
    double pos;                  // interp: position between src0 and src1
    bool submitted;              // interp: handed to the GPU (ticket valid)
    uint64_t pair_id;
    uint64_t ticket;
};

static void engine_log(void *ctx, int level, const char *msg)
{
    struct priv *p = ctx;
    static const int levels[] = {
        [RIFE_LOG_ERR] = MSGL_ERR,
        [RIFE_LOG_WARN] = MSGL_WARN,
        [RIFE_LOG_INFO] = MSGL_INFO,
        [RIFE_LOG_V] = MSGL_V,
        [RIFE_LOG_DEBUG] = MSGL_DEBUG,
    };
    mp_msg(p->f->log, levels[MPCLAMP(level, 0, RIFE_LOG_DEBUG)], "%s\n", msg);
}

static void engine_on_build(void *ctx)
{
    struct priv *p = ctx;
    mp_mutex_lock(&p->lock);
    p->state = ENGINE_BUILDING;
    mp_cond_broadcast(&p->cond);
    mp_mutex_unlock(&p->lock);
}

static bool engine_cancelled(void *ctx)
{
    struct priv *p = ctx;
    return atomic_load(&p->cancel);
}

static MP_THREAD_VOID engine_thread(void *ptr)
{
    struct priv *p = ptr;
    mp_thread_set_name("rife/engine");

    int64_t start = mp_time_ns();
    struct rife_trt *trt = rife_trt_create(&p->cfg);
    if (trt)
        MP_VERBOSE(p->f, "Engine ready after %.2f s.\n",
                   MP_TIME_NS_TO_S(mp_time_ns() - start));

    mp_mutex_lock(&p->lock);
    p->trt = trt;
    p->state = trt ? ENGINE_READY : ENGINE_FAILED;
    mp_cond_broadcast(&p->cond);
    mp_mutex_unlock(&p->lock);

    // Start interpolating from the next frame on.
    mp_filter_wakeup(p->f);
    MP_THREAD_RETURN();
}

static void free_job(struct rife_job *job)
{
    talloc_free(job->img);
    talloc_free(job->src0);
    talloc_free(job->src1);
    *job = (struct rife_job){0};
}

static void engine_destroy(struct priv *p)
{
    if (p->thread_running) {
        atomic_store(&p->cancel, true);
        mp_thread_join(p->thread);
        p->thread_running = false;
    }
    rife_trt_destroy(p->trt);
    p->trt = NULL;
    p->state = ENGINE_NONE;
}

static void report_stats(struct priv *p, bool final)
{
    if (!p->stat_frames)
        return;
    int64_t now = mp_time_ns();
    if (!final && now - p->stat_last_report < MP_TIME_S_TO_NS(10))
        return;
    mp_msg(p->f->log, p->opts->stats ? MSGL_INFO : MSGL_V,
           "%"PRId64" frames interpolated: %.2f ms each on the GPU, player "
           "waited %.2f ms each\n", p->stat_frames,
           p->stat_gpu_ms / p->stat_frames, p->stat_wait_ms / p->stat_frames);
    p->stat_last_report = now;
    p->stat_frames = 0;
    p->stat_gpu_ms = p->stat_wait_ms = 0;
}

static bool sw_format_supported(enum AVPixelFormat fmt, int *depth, int *bps)
{
    switch (fmt) {
    case AV_PIX_FMT_NV12: *depth = 8;  *bps = 1; return true;
    case AV_PIX_FMT_P010: *depth = 10; *bps = 2; return true;
    case AV_PIX_FMT_P012: *depth = 12; *bps = 2; return true;
    case AV_PIX_FMT_P016: *depth = 16; *bps = 2; return true;
    default: return false;
    }
}

static AVHWFramesContext *frames_ctx(struct mp_image *img)
{
    return img->hwctx ? (void *)img->hwctx->data : NULL;
}

// Half resolution optical flow costs a little quality and roughly halves the
// time per frame. It is used when full flow could not keep up.
static bool want_half_flow(struct priv *p, struct mp_image *img)
{
    struct rife_opts *o = p->opts;
    if (o->flow != FLOW_AUTO)
        return o->flow == FLOW_HALF;
    double pixels = (double)img->w * img->h;
    if (pixels > 2560.0 * 1440.0)
        return true;
    double fps = img->nominal_fps > 0 ? img->nominal_fps : 30.0;
    double new_fps;
    if (o->fps > 0) {
        double ratio = o->fps / fps;
        int whole = (int)lrint(ratio);
        new_fps = (whole >= 1 && fabs(ratio - whole) <= 0.02 * ratio)
                      ? fps * (whole - 1) : o->fps;
    } else {
        new_fps = fps * (o->multiplier - 1);
    }
    double capacity = o->full_flow_rate * (1920.0 * 1088.0) / pixels;
    return new_fps > capacity;
}

// The network takes sizes divisible by its modulus. A height that is not can be
// padded up inside the tensor (the whole picture is interpolated, at the cost
// of the next size up) or have rows left out at the top and bottom (those are
// not shown; the next size down is 2 to 12% faster).
static void get_trim(struct priv *p, struct mp_image *img, bool half,
                     int *top, int *bottom)
{
    struct rife_opts *o = p->opts;
    *top = *bottom = 0;
    int mod = o->align ? o->align : (half ? 64 : 32);
    int below = img->h / mod * mod;
    int lose = img->h - below, pad = below + mod - img->h;
    if (!lose || below < 128 || o->trim == TRIM_NO)
        return;
    if (o->trim == TRIM_NEAREST && lose > pad)
        return;
    // Chroma rows stay aligned if the first row kept is an even one.
    *top = (lose / 2) & ~1;
    *bottom = lose - *top;
}

static bool make_key(struct priv *p, struct mp_image *img, struct fmt_key *key)
{
    AVHWFramesContext *fctx = frames_ctx(img);
    if (img->imgfmt != IMGFMT_CUDA || !fctx)
        return false;
    AVCUDADeviceContext *dev = fctx->device_ctx->hwctx;
    *key = (struct fmt_key){
        .cuda_ctx = dev->cuda_ctx,
        .w = img->w,
        .h = img->h,
        .sw_format = fctx->sw_format,
        // Dolby Vision on an HDR10 base layer (profiles 7 and 8): the samples
        // are plain BT.2020, which is what the network should see.
        .sys = img->params.repr.sys == PL_COLOR_SYSTEM_DOLBYVISION &&
               img->params.sys_orig != PL_COLOR_SYSTEM_UNKNOWN
                   ? img->params.sys_orig : img->params.repr.sys,
        .levels = img->params.repr.levels,
        .chroma = img->params.chroma_location,
        .half = want_half_flow(p, img),
    };
    get_trim(p, img, key->half, &key->trim_top, &key->trim_bottom);
    return true;
}

// Have the VO show only the rows that are interpolated. Every frame gets this,
// not just the new ones, so that the picture does not change size on the way.
static void tag_trim(struct priv *p, struct mp_image *img)
{
    struct fmt_key key;
    int depth, bps;
    mp_mutex_lock(&p->lock);
    bool engine_failed = p->state == ENGINE_FAILED;
    mp_mutex_unlock(&p->lock);
    if (engine_failed || (p->opts->multiplier < 2 && p->opts->fps <= 0) ||
        !make_key(p, img, &key) ||
        !sw_format_supported(key.sw_format, &depth, &bps) ||
        !(key.trim_top || key.trim_bottom))
        return;
    struct mp_rect *rc = &img->params.crop;
    if (!mp_image_crop_valid(&img->params) || (!rc->x1 && !rc->y1))
        *rc = (struct mp_rect){0, 0, img->w, img->h};
    int y0 = MPMAX(rc->y0, key.trim_top);
    int y1 = MPMIN(rc->y1, img->h - key.trim_bottom);
    if (y1 > y0) {
        rc->y0 = y0;
        rc->y1 = y1;
    }
}

static bool key_equal(const struct fmt_key *a, const struct fmt_key *b)
{
    return a->cuda_ctx == b->cuda_ctx && a->w == b->w && a->h == b->h &&
           a->sw_format == b->sw_format && a->sys == b->sys &&
           a->levels == b->levels && a->chroma == b->chroma && a->half == b->half &&
           a->trim_top == b->trim_top && a->trim_bottom == b->trim_bottom;
}

static void engine_start(struct priv *p, const struct fmt_key *key)
{
    struct rife_opts *o = p->opts;
    struct mp_filter *f = p->f;

    p->key = *key;
    p->state = ENGINE_FAILED;

    int depth, bps;
    if (!sw_format_supported(key->sw_format, &depth, &bps))
        return;

    const char *model = key->half && o->model_half && o->model_half[0]
                            ? o->model_half : o->model;
    if (!model || !model[0]) {
        MP_ERR(f, "No network given (model=...).\n");
        return;
    }
    talloc_free(p->cfg_onnx);
    talloc_free(p->cfg_engine_dir);
    p->cfg_onnx = mp_get_user_path(p, f->global, model);
    if (o->engine_dir && o->engine_dir[0]) {
        p->cfg_engine_dir = mp_get_user_path(p, f->global, o->engine_dir);
    } else {
        // Next to the network.
        char *slash = strrchr(p->cfg_onnx, '/');
#ifdef _WIN32
        char *backslash = strrchr(p->cfg_onnx, '\\');
        if (backslash && (!slash || backslash > slash))
            slash = backslash;
#endif
        p->cfg_engine_dir = slash
            ? talloc_strndup(p, p->cfg_onnx, slash - p->cfg_onnx)
            : talloc_strdup(p, ".");
    }

    // Only the matrix has to be self-consistent, since the same one is used
    // to go back: anything unusual is run through BT.709.
    float kr = 0.2126f, kb = 0.0722f;
    switch (key->sys) {
    case PL_COLOR_SYSTEM_BT_601:     kr = 0.299f;  kb = 0.114f;  break;
    case PL_COLOR_SYSTEM_SMPTE_240M: kr = 0.212f;  kb = 0.087f;  break;
    case PL_COLOR_SYSTEM_BT_2020_NC:
    case PL_COLOR_SYSTEM_BT_2020_C:  kr = 0.2627f; kb = 0.0593f; break;
    default: break;
    }
    enum pl_chroma_location loc = key->chroma;

    p->cfg = (struct rife_trt_config){
        .cuda_ctx = key->cuda_ctx,
        .w = key->w,
        .h = key->h,
        .trim_top = key->trim_top,
        .trim_bottom = key->trim_bottom,
        .depth = depth,
        .bps = bps,
        .kr = kr,
        .kb = kb,
        .full_range = key->levels == PL_COLOR_LEVELS_FULL,
        .cosited_x = loc != PL_CHROMA_CENTER && loc != PL_CHROMA_TOP_CENTER &&
                     loc != PL_CHROMA_BOTTOM_CENTER,
        .cosited_y = loc == PL_CHROMA_TOP_LEFT || loc == PL_CHROMA_TOP_CENTER,
        .modulus = o->align ? o->align : (key->half ? 64 : 32),
        .onnx = p->cfg_onnx,
        .engine_dir = p->cfg_engine_dir,
        .shape_mode = o->shape_mode,
        .opt_level = o->opt_level,
        .scene_threshold = o->scene_threshold,
        .use_graph = o->graph,
        .slice_ms = o->slice,
        .slice_gap_ms = o->slice_gap,
        .log = engine_log,
        .on_build = engine_on_build,
        .cancelled = engine_cancelled,
        .cb_ctx = p,
    };

    atomic_store(&p->cancel, false);
    p->state = ENGINE_LOADING;
    p->warned_building = false;
    if (mp_thread_create(&p->thread, engine_thread, p)) {
        p->state = ENGINE_FAILED;
        return;
    }
    p->thread_running = true;
}

// True if frames like img can be interpolated right now.
static bool engine_ready(struct priv *p, struct mp_image *img)
{
    struct mp_filter *f = p->f;
    struct fmt_key key;
    if (!make_key(p, img, &key))
        return false;

    int depth, bps;
    if (!sw_format_supported(key.sw_format, &depth, &bps)) {
        if (!p->warned_format) {
            MP_WARN(f, "Cannot interpolate %s video; passing it through.\n",
                    mp_imgfmt_to_name(img->params.hw_subfmt));
            p->warned_format = true;
        }
        return false;
    }

    // (Only this thread moves the state to or away from ENGINE_NONE.)
    mp_mutex_lock(&p->lock);
    bool started = p->state != ENGINE_NONE;
    mp_mutex_unlock(&p->lock);
    if (started && !key_equal(&p->key, &key)) {
        report_stats(p, true);
        // Drain any in-flight GPU jobs before destroying the old engine, and
        // drop unsubmitted interpolated jobs from the old format.
        bool gpu = false;
        for (int n = 0; n < p->num_jobs; n++)
            gpu |= p->jobs[n].interp && p->jobs[n].submitted && !p->jobs[n].waited;
        if (gpu && p->trt)
            rife_trt_wait(p->trt);
        for (int n = p->num_jobs - 1; n >= 0; n--) {
            struct rife_job *job = &p->jobs[n];
            if (job->interp && job->submitted) {
                job->waited = true;
                talloc_free(job->src0);
                talloc_free(job->src1);
                job->src0 = job->src1 = NULL;
            } else if (job->interp && !job->submitted) {
                free_job(job);
                MP_TARRAY_REMOVE_AT(p->jobs, p->num_jobs, n);
            }
        }
        engine_destroy(p);
        started = false;
    }
    if (!started)
        engine_start(p, &key);

    mp_mutex_lock(&p->lock);
    while (p->state == ENGINE_LOADING ||
           (p->state == ENGINE_BUILDING && p->opts->wait))
        mp_cond_wait(&p->cond, &p->lock);
    enum engine_state state = p->state;
    mp_mutex_unlock(&p->lock);

    if (state == ENGINE_BUILDING && !p->warned_building) {
        MP_INFO(f, "Playing at the original frame rate until the engine is "
                "ready.\n");
        p->warned_building = true;
    }
    if (state == ENGINE_READY || state == ENGINE_FAILED) {
        if (p->thread_running) {
            mp_thread_join(p->thread);
            p->thread_running = false;
        }
    }
    return state == ENGINE_READY;
}

static struct rife_frame frame_ref(struct mp_image *img)
{
    return (struct rife_frame){
        .y = (uintptr_t)img->planes[0],
        .uv = (uintptr_t)img->planes[1],
        .pitch_y = img->stride[0],
        .pitch_uv = img->stride[1],
    };
}

// A new CUDA surface shaped like ref.
static struct mp_image *alloc_surface(struct mp_filter *f, AVBufferRef **pool,
                                      AVBufferRef *device, int sw_imgfmt,
                                      int surface_w, int surface_h, int w, int h)
{
    if (!mp_update_av_hw_frames_pool(pool, device, IMGFMT_CUDA, sw_imgfmt,
                                     surface_w, surface_h, false))
    {
        MP_ERR(f, "Failed to create a CUDA frame pool.\n");
        return NULL;
    }
    AVFrame *av_frame = av_frame_alloc();
    MP_HANDLE_OOM(av_frame);
    if (av_hwframe_get_buffer(*pool, av_frame, 0) < 0) {
        MP_ERR(f, "Failed to allocate a CUDA frame.\n");
        av_frame_free(&av_frame);
        return NULL;
    }
    struct mp_image *img = mp_image_from_av_frame(av_frame);
    av_frame_free(&av_frame);
    if (img)
        mp_image_set_size(img, w, h);
    return img;
}

// Output rate of the frames a pair yields: the multiplier, or the fixed rate.
static void set_output_rate(struct mp_image *img, double factor, double fps)
{
    if (fps > 0) {
        if (img->nominal_fps > 0)
            img->nominal_fps = fps;
        if (img->pkt_duration > 0)
            img->pkt_duration = 1.0 / fps;
    } else if (factor > 1) {
        img->nominal_fps *= factor;
        if (img->pkt_duration > 0)
            img->pkt_duration /= factor;
    }
}

static void queue_original(struct priv *p, struct mp_image *src, double pts,
                           double factor, double fps)
{
    struct mp_image *img = mp_image_new_ref(src);
    MP_HANDLE_OOM(img);
    img->pts = pts;
    set_output_rate(img, factor, fps);
    tag_trim(p, img);
    MP_TARRAY_APPEND(p, p->jobs, p->num_jobs, (struct rife_job){.img = img});
}

// Plan the frame between a and b at pos (0..1), shown at pts. The GPU gets it
// later, from submit_jobs().
static bool queue_interpolated(struct mp_filter *f, struct mp_image *a,
                               struct mp_image *b, double pos, double pts,
                               double factor, double fps, uint64_t pair_id)
{
    struct priv *p = f->priv;
    AVHWFramesContext *fctx = frames_ctx(a);
    struct mp_image *out =
        alloc_surface(f, &p->out_pool, fctx->device_ref, a->params.hw_subfmt,
                      fctx->width, fctx->height, a->w, a->h);
    if (!out)
        return false;
    mp_image_copy_attributes(out, a);
    out->pts = pts;
    out->dts = MP_NOPTS_VALUE;
    set_output_rate(out, factor, fps);
    tag_trim(p, out);

    struct rife_job job = {
        .img = out,
        .src0 = mp_image_new_ref(a),
        .src1 = mp_image_new_ref(b),
        .interp = true,
        .pos = pos,
        .pair_id = pair_id,
    };
    MP_HANDLE_OOM(job.src0);
    MP_HANDLE_OOM(job.src1);
    MP_TARRAY_APPEND(p, p->jobs, p->num_jobs, job);
    return true;
}

// Hand the GPU the planned frames among the first `limit` in the queue. Only a
// few at a time: a long stretch of GPU work holds up the VO, which draws on the
// same GPU, and it then misses refreshes.
static bool submit_jobs(struct mp_filter *f, int limit)
{
    struct priv *p = f->priv;
    for (int n = 0; n < MPMIN(limit, p->num_jobs); n++) {
        struct rife_job *job = &p->jobs[n];
        if (!job->interp || job->submitted)
            continue;
        struct rife_frame f0 = frame_ref(job->src0), f1 = frame_ref(job->src1),
                          dst = frame_ref(job->img);
        if (!p->trt || !rife_trt_submit(p->trt, &f0, &f1, &dst, job->pos,
                                        job->pair_id, &job->ticket))
            return false;
        job->submitted = true;
    }
    return true;
}

// Wait for the head of the queue to be ready and take it.
static struct mp_image *take_job(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct rife_job job = p->jobs[0];
    MP_TARRAY_REMOVE_AT(p->jobs, p->num_jobs, 0);
    if (job.interp && !job.submitted) {
        // (Only after a failure.)
        free_job(&job);
        return NULL;
    }
    if (job.interp && !job.waited) {
        double gpu_ms = 0;
        int64_t start = mp_time_ns();
        bool ok = p->trt && rife_trt_wait_ticket(p->trt, job.ticket, &gpu_ms);
        p->stat_wait_ms += (mp_time_ns() - start) / 1e6;
        p->stat_gpu_ms += gpu_ms;
        p->stat_frames++;
        if (!ok) {
            free_job(&job);
            return NULL;
        }
        report_stats(p, false);
    }
    struct mp_image *img = job.img;
    job.img = NULL;
    free_job(&job);
    return img;
}

// Nothing may be in flight when the frames it reads and writes are released.
static void drop_jobs(struct mp_filter *f)
{
    struct priv *p = f->priv;
    bool gpu = false;
    for (int n = 0; n < p->num_jobs; n++)
        gpu |= p->jobs[n].submitted;
    if (gpu && p->trt)
        rife_trt_wait(p->trt);
    for (int n = 0; n < p->num_jobs; n++)
        free_job(&p->jobs[n]);
    p->num_jobs = 0;
}

// Software decoded frames go to the GPU first. Returns NULL on failure.
static struct mp_image *upload(struct mp_filter *f, struct mp_image *img)
{
    struct priv *p = f->priv;

    if (!p->upload_dev) {
        struct mp_hwdec_ctx *hwctx =
            mp_filter_load_hwdec_device(f, IMGFMT_CUDA, AV_HWDEVICE_TYPE_CUDA);
        if (hwctx && hwctx->av_device_ref) {
            p->upload_dev = av_buffer_ref(hwctx->av_device_ref);
        } else if (av_hwdevice_ctx_create(&p->upload_dev, AV_HWDEVICE_TYPE_CUDA,
                                          NULL, NULL, 0) < 0)
        {
            p->upload_dev = NULL;
        }
        if (!p->upload_dev) {
            MP_ERR(f, "No CUDA device.\n");
            return NULL;
        }
    }

    struct mp_image *hw = alloc_surface(f, &p->upload_pool, p->upload_dev,
                                        img->imgfmt, img->w, img->h, img->w, img->h);
    AVFrame *src = hw ? mp_image_to_av_frame(img) : NULL;
    AVFrame *dst = src ? mp_image_to_av_frame(hw) : NULL;
    bool ok = dst && av_hwframe_transfer_data(dst, src, 0) >= 0;
    av_frame_free(&src);
    av_frame_free(&dst);
    if (!ok) {
        MP_ERR(f, "Failed to upload a frame to the GPU.\n");
        talloc_free(hw);
        return NULL;
    }
    mp_image_copy_attributes(hw, img);
    return hw;
}

// Whether the pair (a, b) can be interpolated at all.
static bool pair_usable(struct mp_filter *f, struct mp_image *a, struct mp_image *b)
{
    struct priv *p = f->priv;
    if (p->opts->multiplier < 2 && p->opts->fps <= 0)
        return false;
    if (a->imgfmt != IMGFMT_CUDA || b->imgfmt != IMGFMT_CUDA)
        return false;
    if (a->pts == MP_NOPTS_VALUE || b->pts == MP_NOPTS_VALUE || b->pts <= a->pts)
        return false;
    // A hole in the timestamps is not motion.
    if (b->pts - a->pts > p->opts->max_gap)
        return false;
    AVHWFramesContext *fa = frames_ctx(a), *fb = frames_ctx(b);
    if (!fa || !fb || a->w != b->w || a->h != b->h ||
        fa->sw_format != fb->sw_format ||
        fa->device_ctx->hwctx != fb->device_ctx->hwctx)
        return false;
    if (a->params.repr.sys != b->params.repr.sys ||
        a->params.repr.levels != b->params.repr.levels ||
        a->params.chroma_location != b->params.chroma_location)
        return false;
    return engine_ready(p, b);
}

static void fail_engine(struct priv *p)
{
    mp_mutex_lock(&p->lock);
    p->state = ENGINE_FAILED;
    mp_mutex_unlock(&p->lock);
    // Planned frames that never reached the GPU are dropped; the original
    // frames around them still go out.
    for (int n = p->num_jobs - 1; n >= 0; n--) {
        if (p->jobs[n].interp && !p->jobs[n].submitted) {
            free_job(&p->jobs[n]);
            MP_TARRAY_REMOVE_AT(p->jobs, p->num_jobs, n);
        }
    }
}

// Queue the output frames for the stretch (a, b]: the ones in between, then b
// itself where it falls on an output time.
//
// With fps= set, output frames are put on a fixed grid of that rate (normally
// the screen's refresh rate), so every refresh shows a new picture even when
// the video's rate does not divide it (25, 30, 50, 60 fps on 144 Hz). If the
// rate is a whole multiple of the video's anyway (24 -> 144), this comes down
// to that multiplier, with the original frames kept as they are.
static void queue_pair(struct mp_filter *f, struct mp_image *a, struct mp_image *b)
{
    struct priv *p = f->priv;
    int before = p->num_jobs;
    bool usable = pair_usable(f, a, b);
    double dt = b->pts - a->pts;
    double fps = p->opts->fps;
    int n = p->opts->multiplier;
    bool grid = false;

    if (usable && fps > 0) {
        double ratio = fps * dt;
        int whole = (int)lrint(ratio);
        if (whole >= 1 && fabs(ratio - whole) <= 0.02 * ratio) {
            n = whole;
        } else {
            grid = true;
        }
    }

    bool ok = true;
    bool is_cut = false;
    if (usable && p->opts->scene_threshold > 0) {
        struct rife_frame f0 = frame_ref(a), f1 = frame_ref(b);
        ok = p->trt && rife_trt_check_cut(p->trt, &f0, &f1, &is_cut);
    }
    uint64_t pair_id = ++p->pair_id;

    if (!usable) {
        queue_original(p, b, b->pts, 1, 0);
    } else if (!ok) {
        // Scene cut check failed; fall through to fail_engine below.
    } else if (!grid) {
        for (int k = 1; k < n && ok; k++) {
            double pts = a->pts + dt * k / n;
            if (is_cut) {
                queue_original(p, a, pts, n, 0);
            } else {
                ok = queue_interpolated(f, a, b, (double)k / n, pts, n, 0, pair_id);
            }
        }
        queue_original(p, b, b->pts, n, 0);
    } else {
        // b itself is only shown if it falls on an output time; elsewhere the
        // output frame nearest to it is interpolated like the others.
        double eps = 0.25 / fps;
        int64_t gi = (int64_t)ceil((a->pts + eps) * fps);
        for (;; gi++) {
            double t = (double)gi / fps;
            if (t > b->pts + eps || !ok)
                break;
            if (fabs(t - b->pts) <= eps) {
                queue_original(p, b, t, 0, fps);
            } else if (is_cut) {
                queue_original(p, a, t, 0, fps);
            } else {
                ok = queue_interpolated(f, a, b, (t - a->pts) / dt, t, 0, fps, pair_id);
            }
        }
    }
    if (!ok) {
        MP_ERR(f, "Interpolation failed; passing video through.\n");
        fail_engine(p);
        // Whatever made it into the queue is kept; b follows as it is.
        if (p->num_jobs == before || p->jobs[p->num_jobs - 1].img->pts < b->pts)
            queue_original(p, b, b->pts, 1, 0);
    }
}

static void vf_rife_process(struct mp_filter *f)
{
    struct priv *p = f->priv;
    struct mp_pin *out = f->ppins[1];

    // Output frames kept queued (the one about to go out included). More lets
    // the GPU work further ahead, but long stretches of uninterrupted GPU work
    // hold up the VO's own drawing on the same GPU.
    int want = MPCLAMP(p->opts->queue, 1, RIFE_TRT_MAX_INFLIGHT - 4);
    while (!p->eof && p->num_jobs < want) {
        // (A whole pair is planned at once, but goes to the GPU frame by frame.)
        struct mp_frame frame = mp_pin_out_read(p->in);
        if (frame.type == MP_FRAME_NONE)
            break;
        if (frame.type == MP_FRAME_EOF) {
            p->eof = true;
            mp_image_unrefp(&p->last);
            break;
        }
        if (frame.type != MP_FRAME_VIDEO) {
            MP_ERR(f, "Unexpected frame type.\n");
            mp_frame_unref(&frame);
            mp_filter_internal_mark_failed(f);
            return;
        }
        struct mp_image *img = frame.data;
        mp_mutex_lock(&p->lock);
        bool engine_failed = p->state == ENGINE_FAILED;
        mp_mutex_unlock(&p->lock);
        if (!engine_failed && img->imgfmt != IMGFMT_CUDA &&
            (p->opts->multiplier >= 2 || p->opts->fps > 0))
        {
            struct mp_image *hw = upload(f, img);
            if (hw) {
                talloc_free(img);
                img = hw;
            } else {
                MP_ERR(f, "CUDA upload failed; passing video through.\n");
                fail_engine(p);
            }
        }
        if (p->last) {
            queue_pair(f, p->last, img);
        } else {
            double fps = p->opts->fps > 0 ? p->opts->fps : 0;
            queue_original(p, img, img->pts, fps > 0 ? 0 : 1, fps);
        }
        talloc_free(p->last);
        p->last = img;
    }

    if (!submit_jobs(f, want)) {
        MP_ERR(f, "Interpolation failed; passing video through.\n");
        fail_engine(p);
    }

    if (!mp_pin_in_needs_data(out))
        return;

    if (p->num_jobs) {
        bool interp = p->jobs[0].interp;
        struct mp_image *img = take_job(f);
        if (img) {
            MP_TRACE(f, "emit %s pts %.4f\n", interp ? "interpolated" : "original",
                     img->pts);
            if (!submit_jobs(f, want)) {
                MP_ERR(f, "Interpolation failed; passing video through.\n");
                fail_engine(p);
            }
            mp_pin_in_write(out, MAKE_FRAME(MP_FRAME_VIDEO, img));
        } else {
            MP_ERR(f, "Interpolation failed; passing video through.\n");
            fail_engine(p);
        }
        mp_filter_internal_mark_progress(f);
        return;
    }

    if (p->eof) {
        p->eof = false;
        mp_pin_in_write(out, MP_EOF_FRAME);
    }
}

static void vf_rife_reset(struct mp_filter *f)
{
    struct priv *p = f->priv;
    drop_jobs(f);
    mp_image_unrefp(&p->last);
    p->eof = false;
    p->pair_id++;
}

static void vf_rife_destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    vf_rife_reset(f);
    report_stats(p, true);
    engine_destroy(p);
    av_buffer_unref(&p->out_pool);
    av_buffer_unref(&p->upload_pool);
    av_buffer_unref(&p->upload_dev);
    mp_cond_destroy(&p->cond);
    mp_mutex_destroy(&p->lock);
}

static const struct mp_filter_info vf_rife_filter = {
    .name = "rife",
    .process = vf_rife_process,
    .reset = vf_rife_reset,
    .destroy = vf_rife_destroy,
    .priv_size = sizeof(struct priv),
};

static struct mp_filter *vf_rife_create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &vf_rife_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    p->f = f;
    mp_mutex_init(&p->lock);
    mp_cond_init(&p->cond);

    // CUDA surfaces in a layout the kernels read are taken as they are;
    // anything else is converted to the nearest of them (and uploaded by us).
    struct mp_autoconvert *conv = mp_autoconvert_create(f);
    MP_HANDLE_OOM(conv);
    static const enum AVPixelFormat hw_formats[] = {
        AV_PIX_FMT_NV12, AV_PIX_FMT_P010, AV_PIX_FMT_P012, AV_PIX_FMT_P016,
    };
    for (int n = 0; n < MP_ARRAY_SIZE(hw_formats); n++) {
        int imgfmt = pixfmt2imgfmt(hw_formats[n]);
        if (imgfmt)
            mp_autoconvert_add_imgfmt(conv, IMGFMT_CUDA, imgfmt);
    }
    mp_autoconvert_add_imgfmt(conv, IMGFMT_NV12, 0);
    mp_autoconvert_add_imgfmt(conv, IMGFMT_P010, 0);
    mp_pin_connect(conv->f->pins[0], f->ppins[0]);
    p->in = conv->f->pins[1];

    return f;
}

#define OPT_BASE_STRUCT struct rife_opts
static const m_option_t vf_opts_fields[] = {
    {"multiplier", OPT_INT(multiplier), M_RANGE(1, 16)},
    {"fps", OPT_DOUBLE(fps), M_RANGE(0, 1000)},
    {"queue", OPT_INT(queue), M_RANGE(1, 24)},
    {"slice", OPT_DOUBLE(slice), M_RANGE(0, 1000)},
    {"slice-gap", OPT_DOUBLE(slice_gap), M_RANGE(0, 100)},
    {"model", OPT_STRING(model), .flags = M_OPT_FILE},
    {"model-half", OPT_STRING(model_half), .flags = M_OPT_FILE},
    {"engine-dir", OPT_STRING(engine_dir), .flags = M_OPT_FILE},
    {"flow", OPT_CHOICE(flow,
        {"auto", FLOW_AUTO}, {"full", FLOW_FULL}, {"half", FLOW_HALF})},
    {"full-flow-rate", OPT_DOUBLE(full_flow_rate), M_RANGE(1, 100000)},
    {"scene-threshold", OPT_FLOAT(scene_threshold), M_RANGE(0, 1)},
    {"shape", OPT_CHOICE(shape_mode,
        {"dynamic", RIFE_SHAPE_DYNAMIC}, {"static", RIFE_SHAPE_STATIC})},
    {"align", OPT_CHOICE(align, {"auto", 0}), M_RANGE(8, 256)},
    {"trim", OPT_CHOICE(trim,
        {"no", TRIM_NO}, {"nearest", TRIM_NEAREST}, {"always", TRIM_ALWAYS})},
    {"opt-level", OPT_INT(opt_level), M_RANGE(0, 5)},
    {"graph", OPT_BOOL(graph)},
    {"wait", OPT_BOOL(wait)},
    {"max-gap", OPT_DOUBLE(max_gap), M_RANGE(0, 1000)},
    {"stats", OPT_BOOL(stats)},
    {0}
};

const struct mp_user_filter_entry vf_rife = {
    .desc = {
        .description = "RIFE frame interpolation (TensorRT, CUDA)",
        .name = "rife",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .priv_defaults = &(const OPT_BASE_STRUCT){
            .multiplier = 2,
            .queue = 2,
            .slice = 6,
            .slice_gap = 1,
            .model = "~~/rife/rife_v4.6.onnx",
            .model_half = "~~/rife/rife_v4.6_half.onnx",
            .flow = FLOW_AUTO,
            .shape_mode = RIFE_SHAPE_DYNAMIC,
            .trim = TRIM_ALWAYS,
            .full_flow_rate = 90,
            .scene_threshold = 0.2,
            .opt_level = 3,
            .graph = true,
            .max_gap = 0.5,
        },
        .options = vf_opts_fields,
    },
    .create = vf_rife_create,
};
