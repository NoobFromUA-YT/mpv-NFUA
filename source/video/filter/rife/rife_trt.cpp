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

// TensorRT / CUDA graph half of vf_rife.
//
// Everything here works on device memory that already exists: the decoder's
// surfaces, the output surface, and two tensors owned by this file. The whole
// per-frame job (colour conversion in, scene cut detection, RIFE, colour
// conversion out) is captured once into a CUDA graph. Producing a frame is
// then: point four kernel nodes at the frames in question, launch the graph.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cuda.h>

#include "rife_trt.h"
#include "trt_shim.h"

#ifndef _WIN32
// TensorRT's entry points. Declared here rather than through its headers:
// everything else of TensorRT is used in trt_shim.cpp only.
extern "C" void *createInferBuilder_INTERNAL(void *logger, int32_t version) noexcept;
extern "C" void *createInferRuntime_INTERNAL(void *logger, int32_t version) noexcept;
extern "C" void *createNvOnnxParser_INTERNAL(void *network, void *logger, int version) noexcept;
#endif

// PTX of rife_kernels.cu, generated at build time.
extern "C" const char rife_kernels_ptx[];

static_assert(sizeof(rife_frame) == 24, "kernel ABI");
static_assert(sizeof(rife_geom) == 120, "kernel ABI");
static_assert(sizeof(rife_pre_args) == 184, "kernel ABI");
static_assert(sizeof(rife_scd_args) == 192, "kernel ABI");
static_assert(sizeof(rife_post_args) == 176, "kernel ABI");
static_assert(sizeof(CUdeviceptr) == sizeof(uint64_t), "kernel ABI");

namespace {

// --- what differs between Linux and Windows ---------------------------------
//
// Paths are UTF-8 everywhere in mpv; Windows wants them as UTF-16.

#ifdef _WIN32

const char *const kCudaLib = "nvcuda.dll";

std::wstring widen(const std::string &s)
{
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0)
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &w[0], n);
    w.resize(wcslen(w.c_str()));
    return w;
}

void *lib_open(const char *name)
{
    return (void *)LoadLibraryA(name);
}

void *lib_sym(void *lib, const char *name)
{
    return (void *)GetProcAddress((HMODULE)lib, name);
}

FILE *file_open(const std::string &path, bool write)
{
    return _wfopen(widen(path).c_str(), write ? L"wb" : L"rb");
}

bool file_readable(const std::string &path)
{
    return GetFileAttributesW(widen(path).c_str()) != INVALID_FILE_ATTRIBUTES;
}

void make_dir(const std::string &path)
{
    for (size_t i = 0; i < path.size(); i++) {
        if (i > 0 && path[i] == '\\') {
            std::string sub = path.substr(0, i);
            CreateDirectoryW(widen(sub).c_str(), nullptr);
        } else if (i > 0 && path[i] == '/') {
            std::string sub = path.substr(0, i);
            CreateDirectoryW(widen(sub).c_str(), nullptr);
        }
    }
    CreateDirectoryW(widen(path).c_str(), nullptr);
}

size_t file_size_of(const std::string &path)
{
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExW(widen(path).c_str(), GetFileExInfoStandard, &fad))
        return 0;
    return ((size_t)fad.nFileSizeHigh << 32) | fad.nFileSizeLow;
}

bool file_replace(const std::string &from, const std::string &to)
{
    return MoveFileExW(widen(from).c_str(), widen(to).c_str(),
                       MOVEFILE_REPLACE_EXISTING) != 0;
}

void file_remove(const std::string &path)
{
    DeleteFileW(widen(path).c_str());
}

long process_id()
{
    return (long)GetCurrentProcessId();
}

#else

const char *const kCudaLib = "libcuda.so.1";

void *lib_open(const char *name)
{
    return dlopen(name, RTLD_NOW | RTLD_GLOBAL);
}

void *lib_sym(void *lib, const char *name)
{
    return dlsym(lib, name);
}

FILE *file_open(const std::string &path, bool write)
{
    return fopen(path.c_str(), write ? "wb" : "rb");
}

bool file_readable(const std::string &path)
{
    return access(path.c_str(), R_OK) == 0;
}

void make_dir(const std::string &path)
{
    for (size_t i = 1; i < path.size(); i++) {
        if (path[i] == '/')
            mkdir(path.substr(0, i).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

size_t file_size_of(const std::string &path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || st.st_size < 0)
        return 0;
    return (size_t)st.st_size;
}

bool file_replace(const std::string &from, const std::string &to)
{
    return rename(from.c_str(), to.c_str()) == 0;
}

void file_remove(const std::string &path)
{
    unlink(path.c_str());
}

long process_id()
{
    return (long)getpid();
}

#endif

// TensorRT's three exported entry points. On Windows its DLLs are opened on
// first use, so that mpv starts without them.
struct TrtEntry {
    void *(*builder)(void *, int32_t) = nullptr;
    void *(*runtime)(void *, int32_t) = nullptr;
    void *(*parser)(void *, void *, int) = nullptr;

    bool load(std::string &err)
    {
#ifdef _WIN32
        void *infer = lib_open("nvinfer_11.dll");
        void *onnx = lib_open("nvonnxparser_11.dll");
        if (!infer || !onnx) {
            err = "cannot open nvinfer_11.dll / nvonnxparser_11.dll (they belong "
                  "next to mpv.exe)";
            return false;
        }
        builder = reinterpret_cast<decltype(builder)>(lib_sym(infer, "createInferBuilder_INTERNAL"));
        runtime = reinterpret_cast<decltype(runtime)>(lib_sym(infer, "createInferRuntime_INTERNAL"));
        parser = reinterpret_cast<decltype(parser)>(lib_sym(onnx, "createNvOnnxParser_INTERNAL"));
        if (!builder || !runtime || !parser) {
            err = "the TensorRT DLLs lack their entry points";
            return false;
        }
#else
        (void)err;
        builder = createInferBuilder_INTERNAL;
        runtime = createInferRuntime_INTERNAL;
        parser = createNvOnnxParser_INTERNAL;
#endif
        return true;
    }
};

const TrtEntry *get_trt(std::string &err)
{
    static TrtEntry entry;
    static std::string load_err;
    static bool ok = entry.load(load_err);
    if (!ok)
        err = load_err;
    return ok ? &entry : nullptr;
}

// --- CUDA driver API, resolved at runtime ----------------------------------
//
// mpv must still start on machines without an NVIDIA driver, so libcuda is
// opened on first use rather than linked. cuda.h maps most of these names to
// versioned symbols with macros; XSTR() picks up the mapped name.

#define CU_FUNCS(X) \
    X(cuCtxPushCurrent) X(cuCtxPopCurrent) X(cuCtxGetDevice) X(cuDeviceGetName) \
    X(cuCtxGetStreamPriorityRange) \
    X(cuGetErrorName) X(cuGetErrorString) \
    X(cuModuleLoadDataEx) X(cuModuleUnload) X(cuModuleGetFunction) \
    X(cuMemAlloc) X(cuMemFree) X(cuMemHostAlloc) X(cuMemFreeHost) \
    X(cuMemcpyDtoH) X(cuMemcpyDtoHAsync) X(cuMemsetD8) \
    X(cuStreamCreate) X(cuStreamCreateWithPriority) X(cuStreamDestroy) \
    X(cuStreamSynchronize) X(cuStreamWaitEvent) \
    X(cuEventCreate) X(cuEventDestroy) X(cuEventRecord) X(cuEventElapsedTime) X(cuEventSynchronize) \
    X(cuLaunchKernel) \
    X(cuStreamBeginCapture) X(cuStreamEndCapture) \
    X(cuGraphInstantiateWithFlags) X(cuGraphLaunch) \
    X(cuGraphGetNodes) X(cuGraphNodeGetType) X(cuGraphKernelNodeGetParams) \
    X(cuGraphExecKernelNodeSetParams) X(cuGraphExecDestroy) X(cuGraphDestroy) \
    X(cuGraphGetEdges) X(cuGraphNodeSetEnabled)

#define STR(s) #s
#define XSTR(s) STR(s)

struct CudaApi {
#define X(n) decltype(&::n) n = nullptr;
    CU_FUNCS(X)
#undef X
    void *lib = nullptr;

    bool load(std::string &err)
    {
        lib = lib_open(kCudaLib);
        if (!lib) {
            err = std::string("cannot open ") + kCudaLib + " (NVIDIA driver missing?)";
            return false;
        }
#define X(n) \
        n = reinterpret_cast<decltype(n)>(lib_sym(lib, XSTR(n))); \
        if (!n) { \
            err = std::string(kCudaLib) + " lacks " XSTR(n) " (NVIDIA driver too old)"; \
            return false; \
        }
        CU_FUNCS(X)
#undef X
        return true;
    }
};

// One process-wide table. dlopen() handles are never closed: TensorRT keeps
// its own references to the driver anyway.
const CudaApi *get_cuda(std::string &err)
{
    static CudaApi api;
    static std::string load_err;
    static bool ok = api.load(load_err);
    if (!ok)
        err = load_err;
    return ok ? &api : nullptr;
}

struct Log {
    void (*fn)(void *, int, const char *) = nullptr;
    void *ctx = nullptr;

    __attribute__((format(printf, 3, 4)))
    void msg(int level, const char *fmt, ...) const
    {
        if (!fn)
            return;
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        fn(ctx, level, buf);
    }
};

// What trt_shim.cpp calls back into.
struct HostCtx {
    Log log;
    bool (*cancelled)(void *) = nullptr;
    void *cancel_ctx = nullptr;
};

void *shim_alloc(size_t size)
{
    return calloc(1, size);
}

void shim_log(void *ctx, int level, int from_trt, const char *msg)
{
    static_cast<HostCtx *>(ctx)->log.msg(level, from_trt ? "TensorRT: %s" : "%s", msg);
}

int shim_cancelled(void *ctx)
{
    HostCtx *host = static_cast<HostCtx *>(ctx);
    return host->cancelled && host->cancelled(host->cancel_ctx);
}

// Makes a CUDA context current for the lifetime of the object.
class ContextGuard {
public:
    ContextGuard(const CudaApi *cu, CUcontext ctx) : cu_(cu)
    {
        pushed_ = cu_->cuCtxPushCurrent(ctx) == CUDA_SUCCESS;
    }
    ~ContextGuard()
    {
        CUcontext dummy;
        if (pushed_)
            cu_->cuCtxPopCurrent(&dummy);
    }
    bool ok() const { return pushed_; }

private:
    const CudaApi *cu_;
    bool pushed_;
};

bool read_file(const std::string &path, std::vector<char> &out)
{
    FILE *f = file_open(path, false);
    if (!f)
        return false;
    char buf[1 << 16];
    size_t n;
    out.clear();
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.insert(out.end(), buf, buf + n);
    bool ok = !ferror(f) && !out.empty();
    fclose(f);
    return ok;
}

bool write_file_atomic(const std::string &path, const void *data, size_t size)
{
    std::string tmp = path + ".tmp" + std::to_string(process_id());
    FILE *f = file_open(tmp, true);
    if (!f)
        return false;
    bool ok = fwrite(data, 1, size, f) == size;
    ok = fclose(f) == 0 && ok;
    if (!ok || !file_replace(tmp, path)) {
        file_remove(tmp);
        return false;
    }
    return true;
}

std::string basename_of(const std::string &path)
{
    size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

int round_up(int v, int mod)
{
    return (v + mod - 1) / mod * mod;
}

// Size range of the shared (dynamic shape) engine.
const int kDynMin = 128;
const int kDynMax = 4096;

} // namespace

struct rife_trt {
    const CudaApi *cu = nullptr;
    const TrtEntry *trt = nullptr;
    Log log;
    CUcontext ctx = nullptr;
    rife_geom geom = {};
    float scene_threshold = 0;

    HostCtx host_ctx;
    trt_shim_host host = {};
    trt_shim *shim = nullptr; // the engine, see trt_shim.h

    CUmodule module = nullptr;
    CUfunction fn_pre = nullptr, fn_cells = nullptr, fn_reduce = nullptr, fn_post = nullptr;
    CUstream stream = nullptr, scd_stream = nullptr;
    // Around each launch in flight, indexed by ticket % RIFE_TRT_MAX_INFLIGHT.
    CUevent ev_start[RIFE_TRT_MAX_INFLIGHT] = {}, ev_end[RIFE_TRT_MAX_INFLIGHT] = {};
    uint64_t next_ticket = 0;
    uint64_t last_pair_id = 0;
    CUevent ev_gate = nullptr, ev_scd_gate = nullptr; // see rife_trt_submit() / check_cut()

    CUdeviceptr trt_mem = 0;    // TensorRT's activations
    CUdeviceptr in_tensor = 0;  // fp16 [1, 7, ph, pw] (or HWC8 [ph][pw][8])
    CUdeviceptr out_tensor = 0; // fp16 [1, 3, ph, pw] (or HWC8 [ph][pw][8])
    CUdeviceptr scd_cells = 0;  // float[RIFE_SCD_CELLS]
    CUdeviceptr scd_result = 0; // rife_scd_result
    rife_scd_result *scd_host = nullptr; // pinned host buffer for fast DMA readback

    CUgraph graph = nullptr;
    CUgraphExec graph_exec = nullptr;
    CUgraphNode node_pre = nullptr, node_post = nullptr;
    // The same graph instantiated once per slice, each with only its share of
    // the nodes enabled; launched one after another instead of graph_exec.
    std::vector<CUgraphExec> slices;
    // Which slice holds each of our own kernel nodes.
    int slice_pre = 0, slice_post = 0;

    // With slices, a worker thread launches each one only once the one before
    // has finished, so that the GPU is free in between for the VO's drawing.
    // (Queued back to back, the GPU runs them as one block anyway.)
    struct Job {
        rife_pre_args pre;
        rife_post_args post;
        uint64_t ticket;
    };
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    std::deque<Job> jobs;    // (mu)
    uint64_t done = 0;       // (mu) tickets below this have finished
    bool worker_busy = false; // (mu) true while worker is launching/syncing a job
    bool quit = false;       // (mu)
    bool failed = false;     // (mu)
    double slot_gpu_ms[RIFE_TRT_MAX_INFLIGHT] = {}; // (mu) pure GPU ms for sliced jobs
    CUevent ev_slice = nullptr, ev_slice_t0 = nullptr, ev_slice_t1 = nullptr;
    bool sliced_timing = false;
    float slice_gap_ms = 0;

    bool launch(rife_pre_args &pre, rife_post_args &post, int slot, bool paced,
                double *gpu_ms_out);
    void worker_loop();

    bool check(CUresult res, const char *what) const
    {
        if (res == CUDA_SUCCESS)
            return true;
        const char *name = nullptr, *desc = nullptr;
        cu->cuGetErrorName(res, &name);
        cu->cuGetErrorString(res, &desc);
        log.msg(RIFE_LOG_ERR, "%s failed: %s (%s)", what, name ? name : "?",
                desc ? desc : "?");
        return false;
    }

    // Kernel launch shapes. Rows of 32 keep the tensor writes coalesced.
    void dims_pre(unsigned grid[2], unsigned block[2]) const
    {
        block[0] = 32; block[1] = 8;
        grid[0] = (geom.pw + 31) / 32; grid[1] = (geom.ph + 7) / 8;
    }
    void dims_post(unsigned grid[2], unsigned block[2]) const
    {
        block[0] = 32; block[1] = 8;
        grid[0] = (geom.cw + 31) / 32; grid[1] = (geom.ch + 7) / 8;
    }

    void fill_args(rife_pre_args &pre, rife_post_args &post,
                   const rife_frame &f0, const rife_frame &f1,
                   const rife_frame &dst, float timestep,
                   bool timestep_only) const
    {
        pre = {};
        pre.f0 = f0;
        pre.f1 = f1;
        pre.g = geom;
        pre.tensor = in_tensor;
        pre.timestep = timestep;
        pre.timestep_only = timestep_only ? 1 : 0;

        post = {};
        post.dst = dst;
        post.f0 = f0;
        post.g = geom;
        post.tensor = out_tensor;
    }

    void fill_scd_args(rife_scd_args &scd, const rife_frame &f0,
                       const rife_frame &f1) const
    {
        scd = {};
        scd.f0 = f0;
        scd.f1 = f1;
        scd.g = geom;
        scd.cells = scd_cells;
        scd.result = scd_result;
        scd.threshold = scene_threshold;
    }

    // Issue one frame's interpolation work on the stream. During graph capture
    // this only records it.
    bool enqueue_all(rife_pre_args &pre, rife_post_args &post)
    {
        unsigned grid[2], block[2];
        void *p_pre[] = {&pre}, *p_post[] = {&post};

        dims_pre(grid, block);
        if (!check(cu->cuLaunchKernel(fn_pre, grid[0], grid[1], 1, block[0], block[1], 1,
                                      0, stream, p_pre, nullptr), "launching rife_pre"))
            return false;
        if (!trt_shim_enqueue(shim, stream)) {
            log.msg(RIFE_LOG_ERR, "TensorRT enqueue failed");
            return false;
        }
        dims_post(grid, block);
        return check(cu->cuLaunchKernel(fn_post, grid[0], grid[1], 1, block[0], block[1], 1,
                                        0, stream, p_post, nullptr), "launching rife_post");
    }

    bool set_node(CUgraphNode node, CUfunction fn, const unsigned grid[3],
                  const unsigned block[3], void *args, int slice)
    {
        CUgraphExec exec = slices.empty() ? graph_exec : slices[slice];
        void *params[] = {args};
        CUDA_KERNEL_NODE_PARAMS np = {};
        np.func = fn;
        np.gridDimX = grid[0]; np.gridDimY = grid[1]; np.gridDimZ = grid[2];
        np.blockDimX = block[0]; np.blockDimY = block[1]; np.blockDimZ = block[2];
        np.kernelParams = params;
        return check(cu->cuGraphExecKernelNodeSetParams(exec, node, &np),
                     "retargeting graph node");
    }

    bool capture_graph();
    bool build_slices(float slice_ms);
    void free_graph();
};

void rife_trt::free_graph()
{
    for (CUgraphExec e : slices)
        cu->cuGraphExecDestroy(e);
    slices.clear();
    if (graph_exec)
        cu->cuGraphExecDestroy(graph_exec);
    if (graph)
        cu->cuGraphDestroy(graph);
    graph_exec = nullptr;
    graph = nullptr;
    node_pre = node_post = nullptr;
}

// Record one frame's worth of work on the stream into a graph, and find the
// nodes of our own kernels in it, so that later launches can be pointed at
// other frames. The TensorRT part stays as captured: its tensors never move.
bool rife_trt::capture_graph()
{
    rife_pre_args pre;
    rife_post_args post;
    rife_frame none = {};
    fill_args(pre, post, none, none, none, 0.5f, false);

    // Thread local: the VO thread uses the same context (to copy frames into
    // textures) and must not be made to fail while we record.
    if (!check(cu->cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
               "cuStreamBeginCapture"))
        return false;
    bool ok = enqueue_all(pre, post);
    CUgraph g = nullptr;
    CUresult res = cu->cuStreamEndCapture(stream, &g);
    if (!ok || !check(res, "cuStreamEndCapture") || !g) {
        if (g)
            cu->cuGraphDestroy(g);
        return false;
    }
    graph = g;

    if (!check(cu->cuGraphInstantiateWithFlags(&graph_exec, graph, 0),
               "cuGraphInstantiate"))
        return false;

    size_t count = 0;
    if (!check(cu->cuGraphGetNodes(graph, nullptr, &count), "cuGraphGetNodes"))
        return false;
    std::vector<CUgraphNode> nodes(count);
    if (!check(cu->cuGraphGetNodes(graph, nodes.data(), &count), "cuGraphGetNodes"))
        return false;

    size_t kernels = 0;
    for (CUgraphNode node : nodes) {
        CUgraphNodeType type;
        if (cu->cuGraphNodeGetType(node, &type) != CUDA_SUCCESS ||
            type != CU_GRAPH_NODE_TYPE_KERNEL)
            continue;
        kernels++;
        CUDA_KERNEL_NODE_PARAMS np = {};
        if (cu->cuGraphKernelNodeGetParams(node, &np) != CUDA_SUCCESS)
            continue;
        if (np.func == fn_pre) {
            node_pre = node;
        } else if (np.func == fn_post) {
            node_post = node;
        }
    }
    if (!node_pre || !node_post) {
        log.msg(RIFE_LOG_ERR, "captured graph lacks the conversion kernels");
        return false;
    }
    log.msg(RIFE_LOG_V, "captured CUDA graph: %zu nodes (%zu kernels)", count, kernels);
    return true;
}

// Split the captured graph into slices of about slice_ms of GPU time each.
bool rife_trt::build_slices(float slice_ms)
{
    // How long one launch takes (best of a few). Our own kernels still point at
    // no frames at all (see capture_graph()), so they sit this out; they take
    // a fraction of a millisecond anyway.
    CUgraphNode own[] = {node_pre, node_post};
    bool own_disabled = false;
    auto restore_own = [&]() {
        if (!own_disabled || !graph_exec)
            return true;
        own_disabled = false;
        bool ok = true;
        for (CUgraphNode x : own)
            ok = check(cu->cuGraphNodeSetEnabled(graph_exec, x, 1), "cuGraphNodeSetEnabled") && ok;
        return ok;
    };

    for (CUgraphNode x : own) {
        if (!check(cu->cuGraphNodeSetEnabled(graph_exec, x, 0), "cuGraphNodeSetEnabled")) {
            restore_own();
            return false;
        }
        own_disabled = true;
    }
    CUevent e0 = nullptr, e1 = nullptr;
    float total = 1e9f;
    if (cu->cuEventCreate(&e0, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cu->cuEventCreate(&e1, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
    {
        if (e0) cu->cuEventDestroy(e0);
        if (e1) cu->cuEventDestroy(e1);
        restore_own();
        return false;
    }
    for (int i = 0; i < 3; i++) {
        float ms = 0;
        cu->cuEventRecord(e0, stream);
        cu->cuGraphLaunch(graph_exec, stream);
        cu->cuEventRecord(e1, stream);
        if (cu->cuEventSynchronize(e1) != CUDA_SUCCESS ||
            cu->cuEventElapsedTime(&ms, e0, e1) != CUDA_SUCCESS)
            break;
        total = ms < total ? ms : total;
    }
    cu->cuEventDestroy(e0);
    cu->cuEventDestroy(e1);
    if (!restore_own())
        return false;
    if (total > 1e8f)
        return false;
    int n = (int)(total / slice_ms + 0.999f);
    n = n < 1 ? 1 : (n > 24 ? 24 : n);
    if (n <= 1) {
        log.msg(RIFE_LOG_V, "one launch takes %.1f ms; not split", total);
        return true;
    }

    // Nodes in dependency order.
    size_t count = 0, nedges = 0;
    if (!check(cu->cuGraphGetNodes(graph, nullptr, &count), "cuGraphGetNodes"))
        return false;
    std::vector<CUgraphNode> nodes(count);
    if (!check(cu->cuGraphGetNodes(graph, nodes.data(), &count), "cuGraphGetNodes") ||
        !check(cu->cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &nedges), "cuGraphGetEdges"))
        return false;
    std::vector<CUgraphNode> from(nedges), to(nedges);
    std::vector<CUgraphEdgeData> edata(nedges);
    if (nedges && !check(cu->cuGraphGetEdges(graph, from.data(), to.data(), edata.data(), &nedges),
                         "cuGraphGetEdges"))
        return false;
    std::vector<int> indeg(count, 0);
    auto index_of = [&](CUgraphNode x) {
        for (size_t i = 0; i < count; i++)
            if (nodes[i] == x)
                return (int)i;
        return -1;
    };
    std::vector<std::vector<int>> succ(count);
    for (size_t e = 0; e < nedges; e++) {
        int a = index_of(from[e]), b = index_of(to[e]);
        if (a < 0 || b < 0)
            return false;
        succ[a].push_back(b);
        indeg[b]++;
    }
    std::vector<int> order, ready;
    for (size_t i = 0; i < count; i++)
        if (!indeg[i])
            ready.push_back((int)i);
    while (!ready.empty()) {
        int i = ready.front();
        ready.erase(ready.begin());
        order.push_back(i);
        for (int b : succ[i])
            if (--indeg[b] == 0)
                ready.push_back(b);
    }
    if (order.size() != count)
        return false;

    // Only plain work can be switched off per slice; anything else (events,
    // host callbacks) would run in every slice.
    std::vector<int> work;
    for (int i : order) {
        CUgraphNodeType type;
        if (!check(cu->cuGraphNodeGetType(nodes[i], &type), "cuGraphNodeGetType"))
            return false;
        if (type == CU_GRAPH_NODE_TYPE_KERNEL || type == CU_GRAPH_NODE_TYPE_MEMSET ||
            type == CU_GRAPH_NODE_TYPE_MEMCPY) {
            work.push_back(i);
        } else if (type != CU_GRAPH_NODE_TYPE_EMPTY) {
            log.msg(RIFE_LOG_V, "graph has node type %d; not split", (int)type);
            return true;
        }
    }

    // Time of the work nodes [lo, hi) on their own (best of two runs) using one
    // reusable scratch CUgraphExec. Our own kernels stay off during timing.
    CUevent t0 = nullptr, t1 = nullptr;
    if (cu->cuEventCreate(&t0, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
        cu->cuEventCreate(&t1, CU_EVENT_DEFAULT) != CUDA_SUCCESS)
    {
        if (t0) cu->cuEventDestroy(t0);
        if (t1) cu->cuEventDestroy(t1);
        return false;
    }
    CUgraphExec scratch = nullptr;
    if (!check(cu->cuGraphInstantiateWithFlags(&scratch, graph, 0), "cuGraphInstantiate")) {
        cu->cuEventDestroy(t0);
        cu->cuEventDestroy(t1);
        return false;
    }
    std::vector<unsigned int> scratch_state(work.size(), 1);
    auto make_exec = [&](size_t lo, size_t hi, bool own_on, CUgraphExec *out) {
        CUgraphExec e = nullptr;
        if (!check(cu->cuGraphInstantiateWithFlags(&e, graph, 0), "cuGraphInstantiate"))
            return false;
        for (size_t k = 0; k < work.size(); k++) {
            CUgraphNode x = nodes[work[k]];
            bool is_own = x == node_pre || x == node_post;
            bool on = k >= lo && k < hi && (own_on || !is_own);
            if (!on && !check(cu->cuGraphNodeSetEnabled(e, x, 0), "cuGraphNodeSetEnabled")) {
                cu->cuGraphExecDestroy(e);
                return false;
            }
        }
        *out = e;
        return true;
    };
    auto time_range = [&](size_t lo, size_t hi) {
        for (size_t k = 0; k < work.size(); k++) {
            CUgraphNode x = nodes[work[k]];
            bool is_own = x == node_pre || x == node_post;
            unsigned int on = (k >= lo && k < hi && !is_own) ? 1u : 0u;
            if (scratch_state[k] != on) {
                if (!check(cu->cuGraphNodeSetEnabled(scratch, x, on), "cuGraphNodeSetEnabled"))
                    return -1.0f;
                scratch_state[k] = on;
            }
        }
        float best = 1e9f;
        for (int run = 0; run < 2; run++) {
            float ms = 0;
            cu->cuEventRecord(t0, stream);
            cu->cuGraphLaunch(scratch, stream);
            cu->cuEventRecord(t1, stream);
            if (cu->cuEventSynchronize(t1) != CUDA_SUCCESS ||
                cu->cuEventElapsedTime(&ms, t0, t1) != CUDA_SUCCESS)
                return -1.0f;
            best = ms < best ? ms : best;
        }
        return best;
    };

    // Halve any piece that takes too long, then join neighbours that fit together.
    struct Piece { size_t lo, hi; float ms; };
    std::vector<Piece> pieces;
    std::vector<Piece> todo = {{0, work.size(), total}};
    bool ok_t = true;
    while (!todo.empty() && ok_t) {
        Piece pc = todo.back();
        todo.pop_back();
        if (pc.ms <= slice_ms || pc.hi - pc.lo <= 1) {
            pieces.push_back(pc);
            continue;
        }
        size_t mid = (pc.lo + pc.hi) / 2;
        float a = time_range(pc.lo, mid), b = time_range(mid, pc.hi);
        ok_t = a >= 0 && b >= 0;
        todo.push_back({mid, pc.hi, b});
        todo.push_back({pc.lo, mid, a});
    }
    std::vector<Piece> merged;
    for (const Piece &pc : pieces) {
        if (!ok_t)
            break;
        if (!merged.empty() && merged.back().ms + pc.ms <= slice_ms) {
            float both = time_range(merged.back().lo, pc.hi);
            if (both >= 0 && both <= slice_ms) {
                merged.back().hi = pc.hi;
                merged.back().ms = both;
                continue;
            }
        }
        merged.push_back(pc);
    }
    cu->cuGraphExecDestroy(scratch);
    cu->cuEventDestroy(t0);
    cu->cuEventDestroy(t1);
    if (!ok_t)
        return false;
    if (merged.size() <= 1) {
        log.msg(RIFE_LOG_V, "one launch takes %.1f ms; not split", total);
        return true;
    }

    std::vector<int> slice_of(count, -1);
    std::string desc;
    for (size_t s = 0; s < merged.size(); s++) {
        for (size_t k = merged[s].lo; k < merged[s].hi; k++)
            slice_of[work[k]] = (int)s;
        char buf[32];
        snprintf(buf, sizeof(buf), "%s%.1f", s ? " " : "", merged[s].ms);
        desc += buf;
    }
    for (size_t s = 0; s < merged.size(); s++) {
        CUgraphExec e = nullptr;
        if (!make_exec(merged[s].lo, merged[s].hi, true, &e))
            return false;
        slices.push_back(e);
    }
    n = (int)merged.size();
    slice_pre = slice_of[index_of(node_pre)];
    slice_post = slice_of[index_of(node_post)];
    // Slices replace graph_exec; free the unsplit executable graph.
    cu->cuGraphExecDestroy(graph_exec);
    graph_exec = nullptr;
    log.msg(RIFE_LOG_V, "one launch takes %.1f ms; split into %d slices (ms: %s)",
            total, n, desc.c_str());
    return true;
}

static std::vector<char> build_engine(rife_trt *r, const rife_trt_config *cfg,
                                      bool static_shape, int pw, int ph,
                                      const std::string &tc_path)
{
    std::vector<char> plan, onnx, tc_in;
    if (!read_file(cfg->onnx, onnx)) {
        r->log.msg(RIFE_LOG_ERR, "cannot read network '%s'", cfg->onnx);
        return plan;
    }
    read_file(tc_path, tc_in);

    trt_shim_build build = {};
    build.onnx = onnx.data();
    build.onnx_size = onnx.size();
    build.opt_level = cfg->opt_level;
    build.timing_cache = tc_in.empty() ? nullptr : tc_in.data();
    build.timing_cache_size = tc_in.size();
    if (static_shape) {
        build.num_profiles = 1;
        build.profiles[0] = {pw, ph, pw, ph, pw, ph};
    } else {
        build.num_profiles = 2;
        build.profiles[0] = {kDynMin, kDynMin, 1920, 1024, 2560, 1536};
        build.profiles[1] = {kDynMin, kDynMin, 3840, 2176, kDynMax, kDynMax};
    }

    trt_shim_plan compiled = {};
    if (!trt_shim_compile(&r->host, &build, &compiled)) {
        r->log.msg(RIFE_LOG_ERR, "no engine for '%s'", cfg->onnx);
        return plan;
    }
    if (compiled.tc_data && compiled.tc_size > 0) {
        make_dir(cfg->engine_dir);
        write_file_atomic(tc_path, compiled.tc_data, compiled.tc_size);
    }
    plan.assign((const char *)compiled.data, (const char *)compiled.data + compiled.size);
    trt_shim_plan_free(&compiled);
    return plan;
}

static bool create(rife_trt *r, const rife_trt_config *cfg)
{
    const CudaApi *cu = r->cu;
    const Log &log = r->log;

    ContextGuard guard(cu, r->ctx);
    if (!guard.ok()) {
        log.msg(RIFE_LOG_ERR, "cannot make the CUDA context current");
        return false;
    }

    // --- geometry and colour ---
    rife_geom &g = r->geom;
    g.w = cfg->w;
    g.h = cfg->h;
    int mod = cfg->modulus > 0 ? cfg->modulus : 32;
    g.y0 = cfg->trim_top;
    g.th = cfg->h - cfg->trim_top - cfg->trim_bottom;
    if (g.th < 2 || (g.y0 & 1)) {
        log.msg(RIFE_LOG_ERR, "invalid trim");
        return false;
    }
    g.pw = round_up(cfg->w < kDynMin ? kDynMin : cfg->w, mod);
    g.ph = round_up(g.th < kDynMin ? kDynMin : g.th, mod);
    g.cw = (cfg->w + 1) / 2;
    g.ch = (cfg->h + 1) / 2;
    g.bps = cfg->bps;
    g.shift = cfg->bps == 2 ? 16 - cfg->depth : 0;
    g.maxval = (1 << cfg->depth) - 1;
    g.cosited_x = cfg->cosited_x;
    g.cosited_y = cfg->cosited_y;
    float scale = (float)(1 << (cfg->depth - 8));
    if (cfg->full_range) {
        g.y_off = 0;
        g.y_mul = 1.0f / (float)g.maxval;
        g.c_mul = 1.0f / (float)g.maxval;
    } else {
        g.y_off = 16.0f * scale;
        g.y_mul = 1.0f / (219.0f * scale);
        g.c_mul = 1.0f / (224.0f * scale);
    }
    g.c_off = 128.0f * scale;
    float kr = cfg->kr, kb = cfg->kb, kg = 1.0f - kr - kb;
    g.kr = kr; g.kg = kg; g.kb = kb;
    g.rv = 2.0f * (1.0f - kr);
    g.bu = 2.0f * (1.0f - kb);
    g.gu = -2.0f * kb * (1.0f - kb) / kg;
    g.gv = -2.0f * kr * (1.0f - kr) / kg;
    g.cb_div = 1.0f / (2.0f * (1.0f - kb));
    g.cr_div = 1.0f / (2.0f * (1.0f - kr));
    g.inv_y = 1.0f / g.y_mul;
    g.inv_c = 1.0f / g.c_mul;
    r->scene_threshold = cfg->scene_threshold;


    // --- kernels ---
    if (!r->check(cu->cuModuleLoadDataEx(&r->module, rife_kernels_ptx, 0, nullptr, nullptr),
                  "loading the conversion kernels") ||
        !r->check(cu->cuModuleGetFunction(&r->fn_pre, r->module, "rife_pre"), "rife_pre") ||
        !r->check(cu->cuModuleGetFunction(&r->fn_cells, r->module, "rife_scd_cells"), "rife_scd_cells") ||
        !r->check(cu->cuModuleGetFunction(&r->fn_reduce, r->module, "rife_scd_reduce"), "rife_scd_reduce") ||
        !r->check(cu->cuModuleGetFunction(&r->fn_post, r->module, "rife_post"), "rife_post"))
        return false;

    // --- streams and events ---
    // Non-blocking and lowest priority so VO copies on stream 0 preempt RIFE
    // kernels naturally. scd_stream gets highest priority for instant cut checks.
    int least_prio = 0, greatest_prio = 0;
    if (cu->cuCtxGetStreamPriorityRange(&least_prio, &greatest_prio) == CUDA_SUCCESS &&
        least_prio != greatest_prio)
    {
        if (!r->check(cu->cuStreamCreateWithPriority(&r->stream, CU_STREAM_NON_BLOCKING, least_prio),
                      "cuStreamCreateWithPriority") ||
            !r->check(cu->cuStreamCreateWithPriority(&r->scd_stream, CU_STREAM_NON_BLOCKING, greatest_prio),
                      "cuStreamCreateWithPriority"))
            return false;
    } else {
        if (!r->check(cu->cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate") ||
            !r->check(cu->cuStreamCreate(&r->scd_stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate"))
            return false;
    }
    if (!r->check(cu->cuEventCreate(&r->ev_gate, CU_EVENT_DISABLE_TIMING), "cuEventCreate") ||
        !r->check(cu->cuEventCreate(&r->ev_scd_gate, CU_EVENT_DISABLE_TIMING), "cuEventCreate"))
        return false;
    for (int i = 0; i < RIFE_TRT_MAX_INFLIGHT; i++) {
        if (!r->check(cu->cuEventCreate(&r->ev_start[i], CU_EVENT_DEFAULT), "cuEventCreate") ||
            !r->check(cu->cuEventCreate(&r->ev_end[i], CU_EVENT_DEFAULT), "cuEventCreate"))
            return false;
    }

    // --- engine ---
    char gpu[128] = "gpu";
    CUdevice dev;
    if (cu->cuCtxGetDevice(&dev) == CUDA_SUCCESS)
        cu->cuDeviceGetName(gpu, sizeof(gpu), dev);
    for (char *c = gpu; *c; c++) {
        if (*c == ' ' || *c == '/' || *c == '\\' || *c == ':')
            *c = '-';
    }
    const TrtEntry *trt = r->trt;
    r->host_ctx.log = log;
    r->host_ctx.cancelled = cfg->cancelled;
    r->host_ctx.cancel_ctx = cfg->cb_ctx;
    r->host.create_builder = trt->builder;
    r->host.create_runtime = trt->runtime;
    r->host.create_parser = trt->parser;
    r->host.alloc = shim_alloc;
    r->host.release = free;
    r->host.log = shim_log;
    r->host.cancelled = shim_cancelled;
    r->host.ctx = &r->host_ctx;

    int trt_major, trt_minor, trt_patch;
    trt_shim_version(&trt_major, &trt_minor, &trt_patch);
    size_t onnx_size = file_size_of(cfg->onnx);
    char static_name[96], dynamic_name[96];
    snprintf(static_name, sizeof(static_name), "%dx%d-hwc8-s%zu", g.pw, g.ph, onnx_size);
    snprintf(dynamic_name, sizeof(dynamic_name), "dyn2p-hwc8-%d-%d-s%zu",
             kDynMin, kDynMax, onnx_size);
    auto path_for = [&](const char *shape_name) {
        char suffix[256];
        snprintf(suffix, sizeof(suffix), ".%s.opt%d.trt%d.%d.%d.%s.engine", shape_name,
                 cfg->opt_level, trt_major, trt_minor, trt_patch, gpu);
        return std::string(cfg->engine_dir) + "/" + basename_of(cfg->onnx) + suffix;
    };
    char tc_buf[256];
    snprintf(tc_buf, sizeof(tc_buf), "/rife.hwc8.trt%d.%d.%d.%s.timingcache",
             trt_major, trt_minor, trt_patch, gpu);
    std::string tc_path = std::string(cfg->engine_dir) + tc_buf;

    bool fits_dynamic = g.pw <= kDynMax && g.ph <= kDynMax;
    bool static_shape = cfg->shape_mode == RIFE_SHAPE_STATIC || !fits_dynamic;
    const char *shape = static_shape ? static_name : dynamic_name;
    std::string engine_path = path_for(shape);

    // Activation memory is sized for this video rather than for the largest
    // one the engine could ever take; the shim reports how much that is.
    trt_shim_info trt_info = {};
    std::vector<char> plan;
    if (read_file(engine_path, plan)) {
        r->shim = trt_shim_load(&r->host, plan.data(), plan.size(), g.pw, g.ph,
                                r->stream, &trt_info);
        if (!r->shim)
            log.msg(RIFE_LOG_WARN, "cached engine '%s' is unusable, compiling again",
                    engine_path.c_str());
    }
    if (!r->shim) {
        if (!file_readable(cfg->onnx)) {
            log.msg(RIFE_LOG_ERR, "network '%s' not found", cfg->onnx);
            return false;
        }
        if (cfg->on_build)
            cfg->on_build(cfg->cb_ctx);
        log.msg(RIFE_LOG_INFO, "compiling %s for %s with TensorRT (once, takes about "
                "a minute)...", basename_of(cfg->onnx).c_str(), shape);
        plan = build_engine(r, cfg, static_shape, g.pw, g.ph, tc_path);
        if (plan.empty())
            return false;
        make_dir(cfg->engine_dir);
        if (!write_file_atomic(engine_path, plan.data(), plan.size()))
            log.msg(RIFE_LOG_WARN, "cannot write '%s': the engine will be compiled "
                    "again next time", engine_path.c_str());
        r->shim = trt_shim_load(&r->host, plan.data(), plan.size(), g.pw, g.ph,
                                r->stream, &trt_info);
        if (!r->shim) {
            log.msg(RIFE_LOG_ERR, "cannot load the compiled engine for %dx%d", g.pw, g.ph);
            return false;
        }
    }
    plan = std::vector<char>();

    size_t trt_size = trt_info.mem_size;
    g.in_hwc8 = trt_info.in_hwc8;
    g.out_hwc8 = trt_info.out_hwc8;
    size_t plane = (size_t)g.pw * g.ph * sizeof(uint16_t);
    size_t in_bytes = (size_t)(g.in_hwc8 ? 8 : 7) * plane;
    size_t out_bytes = (size_t)(g.out_hwc8 ? 8 : 3) * plane;
    if (!r->check(cu->cuMemAlloc(&r->trt_mem, trt_size ? trt_size : 1), "allocating activations") ||
        !r->check(cu->cuMemAlloc(&r->in_tensor, in_bytes), "allocating the input tensor") ||
        !r->check(cu->cuMemAlloc(&r->out_tensor, out_bytes), "allocating the output tensor") ||
        !r->check(cu->cuMemAlloc(&r->scd_cells, RIFE_SCD_CELLS * sizeof(float)), "cuMemAlloc") ||
        !r->check(cu->cuMemAlloc(&r->scd_result, sizeof(rife_scd_result)), "cuMemAlloc") ||
        !r->check(cu->cuMemHostAlloc((void **)&r->scd_host, sizeof(rife_scd_result),
                                     CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc"))
        return false;

    cu->cuMemsetD8(r->in_tensor, 0, in_bytes);
    cu->cuMemsetD8(r->scd_result, 0, sizeof(rife_scd_result));
    if (!trt_shim_bind(r->shim, (void *)r->trt_mem, trt_size, (void *)r->in_tensor,
                       (void *)r->out_tensor))
    {
        log.msg(RIFE_LOG_ERR, "cannot bind the tensors");
        return false;
    }

    // TensorRT settles shape dependent state on the first enqueue. That must
    // be over before a capture, which may only record plain kernel launches.
    if (!trt_shim_enqueue(r->shim, r->stream) ||
        !r->check(cu->cuStreamSynchronize(r->stream), "first inference"))
    {
        log.msg(RIFE_LOG_ERR, "first inference failed");
        return false;
    }

    if (cfg->use_graph && !r->capture_graph()) {
        log.msg(RIFE_LOG_WARN, "CUDA graph capture failed; issuing kernels one by one");
        r->free_graph();
    }
    if (r->graph_exec && cfg->slice_ms > 0 && !r->build_slices(cfg->slice_ms)) {
        log.msg(RIFE_LOG_WARN, "could not split the CUDA graph; launching it whole");
        for (CUgraphExec e : r->slices)
            cu->cuGraphExecDestroy(e);
        r->slices.clear();
    }
    r->slice_gap_ms = cfg->slice_gap_ms;
    if (!r->slices.empty()) {
        if (!r->check(cu->cuEventCreate(&r->ev_slice, CU_EVENT_DISABLE_TIMING), "cuEventCreate") ||
            !r->check(cu->cuEventCreate(&r->ev_slice_t0, CU_EVENT_DEFAULT), "cuEventCreate") ||
            !r->check(cu->cuEventCreate(&r->ev_slice_t1, CU_EVENT_DEFAULT), "cuEventCreate"))
            return false;
        r->sliced_timing = true;
        r->worker = std::thread([r] { r->worker_loop(); });
    }

    log.msg(RIFE_LOG_V, "engine %s: picture %dx%d (rows %d-%d), tensor %dx%d, %zu MiB "
            "activations, %s/%s, %s",
            basename_of(engine_path).c_str(), g.w, g.h, g.y0, g.y0 + g.th - 1, g.pw,
            g.ph, trt_size >> 20,
            g.in_hwc8 ? "HWC8" : "NCHW", g.out_hwc8 ? "HWC8" : "NCHW",
            (r->graph_exec || !r->slices.empty()) ? "one CUDA graph launch per frame"
                                                  : "no CUDA graph");
    return true;
}

extern "C" struct rife_trt *rife_trt_create(const struct rife_trt_config *cfg)
{
    Log log;
    log.fn = cfg->log;
    log.ctx = cfg->cb_ctx;

    std::string err;
    const CudaApi *cu = get_cuda(err);
    if (!cu) {
        log.msg(RIFE_LOG_ERR, "%s", err.c_str());
        return nullptr;
    }

    const TrtEntry *trt = get_trt(err);
    if (!trt) {
        log.msg(RIFE_LOG_ERR, "%s", err.c_str());
        return nullptr;
    }

    rife_trt *r = nullptr;
    bool ok = false;
    try {
        r = new rife_trt();
        r->cu = cu;
        r->trt = trt;
        r->log = log;
        r->ctx = (CUcontext)cfg->cuda_ctx;
        ok = create(r, cfg);
    } catch (const std::exception &e) {
        log.msg(RIFE_LOG_ERR, "exception: %s", e.what());
    } catch (...) {
        log.msg(RIFE_LOG_ERR, "unknown exception");
    }
    if (!ok) {
        rife_trt_destroy(r);
        return nullptr;
    }
    return r;
}

extern "C" bool rife_trt_check_cut(struct rife_trt *r, const struct rife_frame *f0,
                                   const struct rife_frame *f1, bool *is_cut)
{
    if (r->scene_threshold <= 0.0f) {
        *is_cut = false;
        return true;
    }
    ContextGuard guard(r->cu, r->ctx);
    if (!guard.ok())
        return false;

    const CudaApi *cu = r->cu;
    if (!r->check(cu->cuEventRecord(r->ev_scd_gate, nullptr), "cuEventRecord") ||
        !r->check(cu->cuStreamWaitEvent(r->scd_stream, r->ev_scd_gate, 0), "cuStreamWaitEvent"))
        return false;

    rife_scd_args scd;
    r->fill_scd_args(scd, *f0, *f1);
    void *p_scd[] = {&scd};
    if (!r->check(cu->cuLaunchKernel(r->fn_cells, (RIFE_SCD_CELLS + 255) / 256, 1, 1,
                                     256, 1, 1, 0, r->scd_stream, p_scd, nullptr),
                  "launching rife_scd_cells") ||
        !r->check(cu->cuLaunchKernel(r->fn_reduce, 1, 1, 1,
                                     256, 1, 1, 0, r->scd_stream, p_scd, nullptr),
                  "launching rife_scd_reduce"))
        return false;

    if (!r->check(cu->cuMemcpyDtoHAsync(r->scd_host, r->scd_result, sizeof(*r->scd_host),
                                        r->scd_stream),
                  "cuMemcpyDtoHAsync") ||
        !r->check(cu->cuStreamSynchronize(r->scd_stream), "cuStreamSynchronize"))
        return false;

    *is_cut = r->scd_host->cut != 0;
    return true;
}

extern "C" bool rife_trt_submit(struct rife_trt *r, const struct rife_frame *f0,
                                const struct rife_frame *f1,
                                const struct rife_frame *dst, float timestep,
                                uint64_t pair_id, uint64_t *ticket)
{
    ContextGuard guard(r->cu, r->ctx);
    if (!guard.ok())
        return false;

    bool timestep_only = pair_id != 0 && pair_id == r->last_pair_id;
    rife_pre_args pre;
    rife_post_args post;
    r->fill_args(pre, post, *f0, *f1, *dst, timestep, timestep_only);

    if (r->worker.joinable()) {
        std::lock_guard<std::mutex> lock(r->mu);
        if (r->failed)
            return false;
        uint64_t t = r->next_ticket++;
        *ticket = t;
        r->last_pair_id = pair_id;
        r->jobs.push_back({pre, post, t});
        r->cv_job.notify_one();
        return true;
    }

    uint64_t t = r->next_ticket;
    int slot = (int)(t % RIFE_TRT_MAX_INFLIGHT);
    if (!r->launch(pre, post, slot, false, nullptr))
        return false;
    r->next_ticket = t + 1;
    r->last_pair_id = pair_id;
    *ticket = t;
    return true;
}

// Issue one frame's work on the stream. paced: wait for each slice to finish
// before issuing the next (worker thread only).
bool rife_trt::launch(rife_pre_args &pre, rife_post_args &post, int slot, bool paced,
                      double *gpu_ms_out)
{
    // dst may be a recycled surface which the VO has only just asked to be
    // copied from, on the default stream. Start after whatever is queued there.
    if (!check(cu->cuEventRecord(ev_gate, nullptr), "cuEventRecord") ||
        !check(cu->cuStreamWaitEvent(stream, ev_gate, 0), "cuStreamWaitEvent"))
        return false;

    // (Changing the graph's nodes below does not affect launches already in
    // flight: each launch keeps the parameters it was launched with.)
    if (!paced)
        cu->cuEventRecord(ev_start[slot], stream);
    if (!graph_exec && slices.empty()) {
        bool ok = enqueue_all(pre, post);
        cu->cuEventRecord(ev_end[slot], stream);
        return ok;
    }

    unsigned grid[3] = {1, 1, 1}, block[3] = {1, 1, 1};
    dims_pre(grid, block);
    if (!set_node(node_pre, fn_pre, grid, block, &pre, slice_pre))
        return false;
    dims_post(grid, block);
    if (!set_node(node_post, fn_post, grid, block, &post, slice_post))
        return false;

    bool ok = true;
    double gpu_sum = 0;
    if (slices.empty()) {
        ok = check(cu->cuGraphLaunch(graph_exec, stream), "cuGraphLaunch");
    } else {
        for (size_t k = 0; k < slices.size() && ok; k++) {
            if (sliced_timing)
                cu->cuEventRecord(ev_slice_t0, stream);
            ok = check(cu->cuGraphLaunch(slices[k], stream), "cuGraphLaunch");
            if (ok && sliced_timing)
                cu->cuEventRecord(ev_slice_t1, stream);
            if (ok && paced && k + 1 < slices.size()) {
                ok = check(cu->cuEventRecord(ev_slice, stream), "cuEventRecord") &&
                     check(cu->cuEventSynchronize(ev_slice), "cuEventSynchronize");
                if (ok && sliced_timing) {
                    float ms = 0;
                    if (cu->cuEventElapsedTime(&ms, ev_slice_t0, ev_slice_t1) == CUDA_SUCCESS)
                        gpu_sum += ms;
                }
                if (ok && slice_gap_ms > 0)
                    std::this_thread::sleep_for(
                        std::chrono::microseconds((long)(slice_gap_ms * 1000)));
            } else if (ok && sliced_timing) {
                if (cu->cuEventSynchronize(ev_slice_t1) == CUDA_SUCCESS) {
                    float ms = 0;
                    if (cu->cuEventElapsedTime(&ms, ev_slice_t0, ev_slice_t1) == CUDA_SUCCESS)
                        gpu_sum += ms;
                }
            }
        }
    }
    cu->cuEventRecord(ev_end[slot], stream);
    if (gpu_ms_out)
        *gpu_ms_out = gpu_sum;
    return ok;
}

void rife_trt::worker_loop()
{
    ContextGuard guard(cu, ctx);
    std::unique_lock<std::mutex> lock(mu);
    while (true) {
        cv_job.wait(lock, [&] { return quit || !jobs.empty(); });
        if (quit)
            break;
        if (failed) {
            if (!jobs.empty()) {
                done = jobs.back().ticket + 1;
                jobs.clear();
            }
            cv_done.notify_all();
            continue;
        }
        Job job = jobs.front();
        jobs.pop_front();
        worker_busy = true;
        lock.unlock();

        int slot = (int)(job.ticket % RIFE_TRT_MAX_INFLIGHT);
        double gpu_ms = 0;
        bool ok = guard.ok() && launch(job.pre, job.post, slot, true, &gpu_ms) &&
                  check(cu->cuEventSynchronize(ev_end[slot]), "cuEventSynchronize");

        lock.lock();
        worker_busy = false;
        slot_gpu_ms[slot] = gpu_ms;
        failed |= !ok;
        if (failed && !jobs.empty()) {
            done = jobs.back().ticket + 1;
            jobs.clear();
        } else {
            done = job.ticket + 1;
        }
        cv_done.notify_all();
    }
}

extern "C" bool rife_trt_wait_ticket(struct rife_trt *r, uint64_t ticket, double *gpu_ms)
{
    if (r->worker.joinable()) {
        std::unique_lock<std::mutex> lock(r->mu);
        r->cv_done.wait(lock, [&] { return r->done > ticket || r->failed; });
        if (r->failed)
            return false;
        if (gpu_ms) {
            int slot = (int)(ticket % RIFE_TRT_MAX_INFLIGHT);
            *gpu_ms = r->slot_gpu_ms[slot];
        }
        return true;
    }
    ContextGuard guard(r->cu, r->ctx);
    if (!guard.ok())
        return false;
    if (ticket >= r->next_ticket || r->next_ticket - ticket > RIFE_TRT_MAX_INFLIGHT) {
        // Not one that is (or could still be) tracked: wait for everything.
        return r->check(r->cu->cuStreamSynchronize(r->stream), "cuStreamSynchronize");
    }
    int slot = (int)(ticket % RIFE_TRT_MAX_INFLIGHT);
    if (!r->check(r->cu->cuEventSynchronize(r->ev_end[slot]), "cuEventSynchronize"))
        return false;
    if (gpu_ms) {
        float ms = 0;
        r->cu->cuEventElapsedTime(&ms, r->ev_start[slot], r->ev_end[slot]);
        *gpu_ms = ms;
    }
    return true;
}

extern "C" bool rife_trt_wait(struct rife_trt *r)
{
    if (r->worker.joinable()) {
        std::unique_lock<std::mutex> lock(r->mu);
        r->cv_done.wait(lock, [&] { return r->jobs.empty() && !r->worker_busy; });
    }
    ContextGuard guard(r->cu, r->ctx);
    return guard.ok() &&
           r->check(r->cu->cuStreamSynchronize(r->stream), "cuStreamSynchronize");
}

extern "C" bool rife_trt_read_scd(struct rife_trt *r, struct rife_scd_result *out)
{
    ContextGuard guard(r->cu, r->ctx);
    return guard.ok() &&
           r->check(r->cu->cuMemcpyDtoH(out, r->scd_result, sizeof(*out)), "cuMemcpyDtoH");
}

extern "C" void rife_trt_get_tensor_size(struct rife_trt *r, int *pw, int *ph)
{
    *pw = r->geom.pw;
    *ph = r->geom.ph;
}

extern "C" void rife_trt_destroy(struct rife_trt *r)
{
    if (!r)
        return;
    const CudaApi *cu = r->cu;
    if (r->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(r->mu);
            r->quit = true;
            r->cv_job.notify_all();
        }
        r->worker.join();
    }
    {
        ContextGuard guard(cu, r->ctx);
        if (r->ev_slice)
            cu->cuEventDestroy(r->ev_slice);
        if (r->ev_slice_t0)
            cu->cuEventDestroy(r->ev_slice_t0);
        if (r->ev_slice_t1)
            cu->cuEventDestroy(r->ev_slice_t1);
        if (r->stream)
            cu->cuStreamSynchronize(r->stream);
        if (r->scd_stream)
            cu->cuStreamSynchronize(r->scd_stream);
        r->free_graph();
        trt_shim_free(r->shim);
        r->shim = nullptr;
        for (CUdeviceptr p : {r->trt_mem, r->in_tensor, r->out_tensor, r->scd_cells,
                              r->scd_result})
        {
            if (p)
                cu->cuMemFree(p);
        }
        if (r->scd_host)
            cu->cuMemFreeHost(r->scd_host);
        if (r->ev_gate)
            cu->cuEventDestroy(r->ev_gate);
        if (r->ev_scd_gate)
            cu->cuEventDestroy(r->ev_scd_gate);
        for (int i = 0; i < RIFE_TRT_MAX_INFLIGHT; i++) {
            if (r->ev_start[i])
                cu->cuEventDestroy(r->ev_start[i]);
            if (r->ev_end[i])
                cu->cuEventDestroy(r->ev_end[i]);
        }
        if (r->stream)
            cu->cuStreamDestroy(r->stream);
        if (r->scd_stream)
            cu->cuStreamDestroy(r->scd_stream);
        if (r->module)
            cu->cuModuleUnload(r->module);
    }
    delete r;
}
