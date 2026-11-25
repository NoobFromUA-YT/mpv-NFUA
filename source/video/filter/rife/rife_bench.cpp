// rife-bench: measures how long one RIFE v4.6 interpolation takes on this GPU,
// for a few common picture sizes at full and half flow resolution. Used by
// portable_config/scripts/rife.lua (the performance assessor) to pick the
// multipliers automatic mode uses. Needs no window or display: it makes its own
// CUDA context and runs the same engine, kernels and CUDA graph as vf_rife.
//
//   rife-bench --model FULL.onnx --model-half HALF.onnx [--engine-dir DIR]
//              [--sizes 1280x720,1920x1080,...] [--full-max-pixels N] [--runs N]
//
// Output, one line each, for the script to read:
//   gpu <name>
//   result <flow> <w>x<h> <tensor megapixels> <median ms>
//   building                (an engine is being compiled: takes minutes)
//   error <text>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "rife_trt.h"

namespace {

typedef int CUresult_;
typedef int CUdevice_;
typedef void *CUcontext_;
typedef unsigned long long CUdeviceptr_;

struct Cuda {
    CUresult_ (*Init)(unsigned);
    CUresult_ (*DeviceGet)(CUdevice_ *, int);
    CUresult_ (*DeviceGetName)(char *, int, CUdevice_);
    CUresult_ (*PrimaryCtxRetain)(CUcontext_ *, CUdevice_);
    CUresult_ (*PrimaryCtxRelease)(CUdevice_);
    CUresult_ (*CtxPushCurrent)(CUcontext_);
    CUresult_ (*CtxPopCurrent)(CUcontext_ *);
    CUresult_ (*MemAlloc)(CUdeviceptr_ *, size_t);
    CUresult_ (*MemFree)(CUdeviceptr_);
    CUresult_ (*MemsetD8)(CUdeviceptr_, unsigned char, size_t);
    CUresult_ (*CtxSynchronize)();
};

bool load_cuda(Cuda *c)
{
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvcuda.dll");
    auto sym = [&](const char *n) { return (void *)GetProcAddress(lib, n); };
#else
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    auto sym = [&](const char *n) { return dlsym(lib, n); };
#endif
    if (!lib)
        return false;
    bool ok = true;
#define L(field, name) ok = ok && (*(void **)&c->field = sym(name));
    L(Init, "cuInit")
    L(DeviceGet, "cuDeviceGet")
    L(DeviceGetName, "cuDeviceGetName")
    L(PrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
    L(PrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2")
    L(CtxPushCurrent, "cuCtxPushCurrent_v2")
    L(CtxPopCurrent, "cuCtxPopCurrent_v2")
    L(MemAlloc, "cuMemAlloc_v2")
    L(MemFree, "cuMemFree_v2")
    L(MemsetD8, "cuMemsetD8_v2")
    L(CtxSynchronize, "cuCtxSynchronize")
#undef L
    return ok;
}

int g_verbose = 0;

void log_cb(void *, int level, const char *msg)
{
    if (level <= RIFE_LOG_ERR || (g_verbose && level <= RIFE_LOG_INFO))
        fprintf(stderr, "rife: %s\n", msg);
}

void build_cb(void *)
{
    printf("building\n");
    fflush(stdout);
}

struct Frame {
    CUdeviceptr_ mem = 0;
    rife_frame f{};
};

bool alloc_frame(Cuda &cu, int w, int h, unsigned char fill, Frame *out)
{
    int pitch = (w + 255) / 256 * 256;
    size_t y = (size_t)pitch * h, uv = (size_t)pitch * ((h + 1) / 2);
    if (cu.MemAlloc(&out->mem, y + uv) != 0)
        return false;
    cu.MemsetD8(out->mem, fill, y);
    cu.MemsetD8(out->mem + y, 128, uv);
    out->f.y = out->mem;
    out->f.uv = out->mem + y;
    out->f.pitch_y = pitch;
    out->f.pitch_uv = pitch;
    return true;
}

// Rows left out at the top and bottom, as vf_rife does with trim=always.
void trim_rows(int h, int mod, int *top, int *bottom)
{
    *top = *bottom = 0;
    int below = h / mod * mod;
    int lose = h - below;
    if (!lose || below < 128)
        return;
    *top = (lose / 2) & ~1;
    *bottom = lose - *top;
}

bool measure(Cuda &cu, CUcontext_ ctx, const char *onnx, const char *engine_dir,
             bool half, int mod, bool trim, int w, int h, int runs, int opt_level)
{
    rife_trt_config cfg{};
    cfg.cuda_ctx = ctx;
    cfg.w = w;
    cfg.h = h;
    if (trim)
        trim_rows(h, mod, &cfg.trim_top, &cfg.trim_bottom);
    cfg.depth = 8;
    cfg.bps = 1;
    cfg.kr = 0.2126f;
    cfg.kb = 0.0722f;
    cfg.cosited_x = true;
    cfg.modulus = mod;
    cfg.onnx = onnx;
    cfg.engine_dir = engine_dir;
    cfg.shape_mode = RIFE_SHAPE_DYNAMIC;
    cfg.opt_level = opt_level;
    cfg.scene_threshold = 0;   // always run the network
    cfg.use_graph = true;
    cfg.slice_ms = 0;          // one piece: the pure GPU time
    cfg.log = log_cb;
    cfg.on_build = build_cb;

    rife_trt *r = rife_trt_create(&cfg);
    if (!r) {
        printf("error engine for %dx%d did not load\n", w, h);
        return false;
    }
    int pw = 0, ph = 0;
    rife_trt_get_tensor_size(r, &pw, &ph);

    Frame f0, f1, dst;
    bool ok = alloc_frame(cu, w, h, 40, &f0) && alloc_frame(cu, w, h, 200, &f1) &&
              alloc_frame(cu, w, h, 0, &dst);
    std::vector<double> ms;
    for (int i = 0; ok && i < runs + 5; i++) {
        uint64_t ticket;
        double t = 0;
        ok = rife_trt_submit(r, &f0.f, &f1.f, &dst.f, 0.5f, (uint64_t)(i + 1), &ticket) &&
             rife_trt_wait_ticket(r, ticket, &t);
        if (i >= 5)   // the first launches warm up clocks and caches
            ms.push_back(t);
    }
    rife_trt_destroy(r);
    for (Frame *f : {&f0, &f1, &dst})
        if (f->mem)
            cu.MemFree(f->mem);
    if (!ok || ms.empty()) {
        printf("error measuring %dx%d failed\n", w, h);
        return false;
    }
    std::sort(ms.begin(), ms.end());
    double median = ms[ms.size() / 2];
    printf("result %s %dx%d %.4f %.3f\n", half ? "half" : "full", w, h,
           (double)pw * ph / 1e6, median);
    fflush(stdout);
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string model, model_half, engine_dir;
    std::string sizes = "1280x720,1920x1080,2560x1440,3840x2160";
    long full_max_pixels = 2560L * 1440;
    int align = 32;
    int align_half = 0;
    bool trim = false;
    int runs = 30;
    int opt_level = 3;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--model") model = next();
        else if (a == "--model-half") model_half = next();
        else if (a == "--engine-dir") engine_dir = next();
        else if (a == "--sizes") sizes = next();
        else if (a == "--full-max-pixels") full_max_pixels = atol(next().c_str());
        else if (a == "--align") align = std::clamp(atoi(next().c_str()), 8, 256);
        else if (a == "--align-half") align_half = std::clamp(atoi(next().c_str()), 0, 256);
        else if (a == "--trim") trim = true;
        else if (a == "--runs") runs = std::max(3, atoi(next().c_str()));
        else if (a == "--opt-level") opt_level = std::clamp(atoi(next().c_str()), 0, 5);
        else if (a == "-v") g_verbose = 1;
        else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (model.empty()) {
        fprintf(stderr, "usage: rife-bench --model FULL.onnx [--model-half HALF.onnx] "
                        "[--engine-dir DIR] [--sizes WxH,...] [--align N] "
                        "[--align-half N] [--trim] [--runs N]\n");
        return 2;
    }
    bool has_half = !model_half.empty();
    int eff_align_half = align_half > 0 ? align_half : std::min(256, align * 2);
    if (engine_dir.empty()) {
        size_t slash = model.find_last_of("/\\");
        engine_dir = slash == std::string::npos ? "." : model.substr(0, slash);
    }

    Cuda cu{};
    CUdevice_ dev = 0;
    CUcontext_ ctx = nullptr;
    if (!load_cuda(&cu) || cu.Init(0) != 0 || cu.DeviceGet(&dev, 0) != 0 ||
        cu.PrimaryCtxRetain(&ctx, dev) != 0) {
        printf("error no CUDA device (NVIDIA driver missing?)\n");
        return 1;
    }
    char name[256] = "";
    cu.DeviceGetName(name, sizeof(name), dev);
    printf("gpu %s\n", name);
    fflush(stdout);
    cu.CtxPushCurrent(ctx);

    int done = 0, failed = 0;
    size_t pos = 0;
    while (pos <= sizes.size()) {
        size_t comma = sizes.find(',', pos);
        std::string s = sizes.substr(pos, comma == std::string::npos ? std::string::npos
                                                                      : comma - pos);
        pos = comma == std::string::npos ? sizes.size() + 1 : comma + 1;
        int w = 0, h = 0;
        if (sscanf(s.c_str(), "%dx%d", &w, &h) != 2 || w < 128 || h < 128)
            continue;
        if (!has_half) {
            (measure(cu, ctx, model.c_str(), engine_dir.c_str(), false, align, trim, w, h, runs, opt_level)
                 ? done : failed)++;
        } else {
            if ((long)w * h <= full_max_pixels)
                (measure(cu, ctx, model.c_str(), engine_dir.c_str(), false, align, trim, w, h, runs, opt_level)
                     ? done : failed)++;
            (measure(cu, ctx, model_half.c_str(), engine_dir.c_str(), true, eff_align_half, trim, w, h, runs, opt_level)
                 ? done : failed)++;
        }
    }

    CUcontext_ dummy;
    cu.CtxPopCurrent(&dummy);
    cu.PrimaryCtxRelease(dev);
    return done && !failed ? 0 : 1;
}
