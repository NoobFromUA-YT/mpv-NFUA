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

// See trt_shim.h. This file must stay free standing: no standard library, no
// exceptions, no statics with constructors. On Windows it is compiled for the
// MSVC C++ ABI and linked into a MinGW program, and anything beyond calls
// through TensorRT's interfaces and the host's callbacks would need a runtime
// that is not there.

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "trt_shim.h"

#ifdef TRT_SHIM_FREESTANDING
// The compiler wants these for the destructors of the two classes below.
// Nothing here is ever allocated with new, so they are never reached.
void operator delete(void *) noexcept {}
void operator delete(void *, size_t) noexcept {}
#endif

namespace {

struct Place {};

} // namespace

inline void *operator new(size_t, Place, void *ptr) noexcept
{
    return ptr;
}

namespace {

using namespace nvinfer1;

class Logger final : public ILogger {
public:
    explicit Logger(const trt_shim_host *host) : host_(*host) {}
    void log(Severity severity, const char *msg) noexcept override
    {
        int level;
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            level = 0;
            break;
        case Severity::kWARNING:
            level = 3;
            break;
        case Severity::kINFO:
            level = 4;
            break;
        default:
            return;
        }
        host_.log(host_.ctx, level, 1, msg);
    }
    void say(int level, const char *msg) const
    {
        host_.log(host_.ctx, level, 0, msg);
    }
    const trt_shim_host &host() const { return host_; }

private:
    trt_shim_host host_;
};

// Lets a build be abandoned (mpv is quitting, or the video changed).
class Monitor final : public IProgressMonitor {
public:
    explicit Monitor(const trt_shim_host *host) : host_(host) {}
    void phaseStart(char const *, char const *, int32_t) noexcept override {}
    void phaseFinish(char const *) noexcept override {}
    bool stepComplete(char const *, int32_t) noexcept override
    {
        return !(host_->cancelled && host_->cancelled(host_->ctx));
    }

private:
    const trt_shim_host *host_;
};

template <typename T>
void destroy(T *&obj)
{
    if (obj)
        delete obj;
    obj = nullptr;
}

} // namespace

struct trt_shim {
    Logger *logger;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *exec;
    const char *in_name, *out_name;
    void (*release)(void *ptr);
};

extern "C" int trt_shim_compile(const struct trt_shim_host *host,
                                const struct trt_shim_build *b,
                                struct trt_shim_plan *out)
{
    out->handle = nullptr;
    out->data = nullptr;
    out->size = 0;
    out->tc_handle = nullptr;
    out->tc_data = nullptr;
    out->tc_size = 0;

    Logger logger(host);
    Monitor monitor(host);
    IBuilder *builder = nullptr;
    INetworkDefinition *network = nullptr;
    nvonnxparser::IParser *parser = nullptr;
    IBuilderConfig *config = nullptr;
    ITimingCache *tc = nullptr;
    IHostMemory *mem = nullptr;
    IHostMemory *tc_mem = nullptr;
    int ok = 0;

    builder = static_cast<IBuilder *>(host->create_builder(
        static_cast<ILogger *>(&logger), NV_TENSORRT_VERSION));
    if (!builder) {
        logger.say(0, "cannot start the TensorRT builder");
        goto done;
    }
    network = builder->createNetworkV2(0);
    if (!network)
        goto done;
    parser = static_cast<nvonnxparser::IParser *>(host->create_parser(
        network, static_cast<ILogger *>(&logger), NV_ONNX_PARSER_VERSION));
    if (!parser || !parser->parse(b->onnx, b->onnx_size)) {
        logger.say(0, "cannot parse the network");
        goto done;
    }
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1) {
        logger.say(0, "not a single input RIFE network");
        goto done;
    }
    {
        ITensor *input = network->getInput(0);
        ITensor *output = network->getOutput(0);
        Dims dims = input->getDimensions();
        if (dims.nbDims != 4 || dims.d[1] != 7 || input->getType() != DataType::kHALF ||
            output->getType() != DataType::kHALF)
        {
            logger.say(0, "the network must take a fp16 [1, 7, H, W] tensor and "
                          "return fp16 (see make_rife_model.py)");
            goto done;
        }

        // Allow TensorRT's auto-tuner to choose kHWC8 or kLINEAR per profile so
        // it avoids reformat overhead without forcing slower Conv/GridSample tactics.
        uint32_t fmt_mask = (1U << static_cast<int32_t>(TensorFormat::kHWC8)) |
                            (1U << static_cast<int32_t>(TensorFormat::kLINEAR));
        input->setAllowedFormats(fmt_mask);
        output->setAllowedFormats(fmt_mask);

        config = builder->createBuilderConfig();
        if (!config)
            goto done;
        if (b->num_profiles < 1 || b->num_profiles > 4) {
            logger.say(0, "invalid number of optimization profiles");
            goto done;
        }
        const char *name = input->getName();
        for (int p = 0; p < b->num_profiles; p++) {
            const trt_shim_profile &pr = b->profiles[p];
            IOptimizationProfile *profile = builder->createOptimizationProfile();
            if (!profile)
                goto done;
            profile->setDimensions(name, OptProfileSelector::kMIN, Dims4{1, 7, pr.min_h, pr.min_w});
            profile->setDimensions(name, OptProfileSelector::kOPT, Dims4{1, 7, pr.opt_h, pr.opt_w});
            profile->setDimensions(name, OptProfileSelector::kMAX, Dims4{1, 7, pr.max_h, pr.max_w});
            if (config->addOptimizationProfile(profile) < 0) {
                logger.say(0, "failed to add optimization profile");
                goto done;
            }
        }
    }
    config->setFlag(BuilderFlag::kTF32);
    config->setBuilderOptimizationLevel(b->opt_level);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, static_cast<size_t>(4ULL << 30));
    // Everything on the caller's stream: a graph capture sees all of it.
    config->setMaxAuxStreams(0);
    config->setProgressMonitor(&monitor);
    tc = config->createTimingCache(b->timing_cache, b->timing_cache_size);
    if (tc)
        config->setTimingCache(*tc, false);

    mem = builder->buildSerializedNetwork(*network, *config);
    if (host->cancelled && host->cancelled(host->ctx)) {
        logger.say(3, "engine build abandoned");
        goto done;
    }
    if (!mem || !mem->size()) {
        logger.say(0, "TensorRT could not compile the network");
        goto done;
    }
    if (tc) {
        tc_mem = tc->serialize();
        if (tc_mem && tc_mem->size()) {
            out->tc_handle = tc_mem;
            out->tc_data = tc_mem->data();
            out->tc_size = tc_mem->size();
            tc_mem = nullptr;
        }
    }
    out->handle = mem;
    out->data = mem->data();
    out->size = mem->size();
    mem = nullptr;
    ok = 1;

done:
    destroy(tc_mem);
    destroy(mem);
    destroy(config);
    destroy(parser);
    destroy(network);
    destroy(builder);
    destroy(tc);
    return ok;
}

extern "C" void trt_shim_plan_free(struct trt_shim_plan *plan)
{
    IHostMemory *mem = static_cast<IHostMemory *>(plan->handle);
    IHostMemory *tc_mem = static_cast<IHostMemory *>(plan->tc_handle);
    destroy(mem);
    destroy(tc_mem);
    plan->handle = nullptr;
    plan->data = nullptr;
    plan->size = 0;
    plan->tc_handle = nullptr;
    plan->tc_data = nullptr;
    plan->tc_size = 0;
}

extern "C" struct trt_shim *trt_shim_load(const struct trt_shim_host *host,
                                          const void *plan, size_t plan_size,
                                          int w, int h, void *stream,
                                          struct trt_shim_info *info)
{
    *info = {};
    trt_shim *s = static_cast<trt_shim *>(host->alloc(sizeof(trt_shim)));
    void *logger_mem = host->alloc(sizeof(Logger));
    if (!s || !logger_mem) {
        host->release(s);
        host->release(logger_mem);
        return nullptr;
    }
    s->release = host->release;
    s->logger = new (Place(), logger_mem) Logger(host);

    s->runtime = static_cast<IRuntime *>(host->create_runtime(
        static_cast<ILogger *>(s->logger), NV_TENSORRT_VERSION));
    if (!s->runtime) {
        s->logger->say(0, "cannot start the TensorRT runtime");
        goto fail;
    }
    s->engine = s->runtime->deserializeCudaEngine(plan, plan_size);
    if (!s->engine) {
        s->logger->say(1, "the engine cannot be loaded");
        goto fail;
    }

    for (int i = 0; i < s->engine->getNbIOTensors(); i++) {
        const char *name = s->engine->getIOTensorName(i);
        if (s->engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
            s->in_name = name;
        } else {
            s->out_name = name;
        }
    }
    if (!s->in_name || !s->out_name || s->engine->getNbIOTensors() != 2 ||
        s->engine->getTensorDataType(s->in_name) != DataType::kHALF ||
        s->engine->getTensorDataType(s->out_name) != DataType::kHALF)
    {
        s->logger->say(0, "the engine has unexpected tensors");
        goto fail;
    }

    // Activation memory is sized for this video rather than for the largest
    // one the engine could ever take.
    s->exec = s->engine->createExecutionContext(
        ExecutionContextAllocationStrategy::kUSER_MANAGED);
    if (!s->exec) {
        s->logger->say(0, "cannot create execution context");
        goto fail;
    }
    {
        int best_prof = -1;
        int64_t best_diff = 0;
        int64_t target_area = static_cast<int64_t>(h) * w;
        int nprof = s->engine->getNbOptimizationProfiles();
        for (int p = 0; p < nprof; p++) {
            Dims dmin = s->engine->getProfileShape(s->in_name, p, OptProfileSelector::kMIN);
            Dims dopt = s->engine->getProfileShape(s->in_name, p, OptProfileSelector::kOPT);
            Dims dmax = s->engine->getProfileShape(s->in_name, p, OptProfileSelector::kMAX);
            if (dmin.nbDims == 4 && dmax.nbDims == 4 && dopt.nbDims == 4 &&
                h >= dmin.d[2] && h <= dmax.d[2] && w >= dmin.d[3] && w <= dmax.d[3])
            {
                int64_t opt_area = static_cast<int64_t>(dopt.d[2]) * dopt.d[3];
                int64_t diff = opt_area >= target_area
                    ? opt_area - target_area : target_area - opt_area;
                if (best_prof < 0 || diff < best_diff) {
                    best_prof = p;
                    best_diff = diff;
                }
            }
        }
        if (best_prof < 0)
            best_prof = 0;
        if (best_prof > 0 &&
            !s->exec->setOptimizationProfileAsync(best_prof, static_cast<cudaStream_t>(stream)))
        {
            s->logger->say(0, "cannot select optimization profile");
            goto fail;
        }
        if (!s->exec->setInputShape(s->in_name, Dims4{1, 7, h, w})) {
            s->logger->say(0, "the engine does not accept this size");
            goto fail;
        }
        {
            Dims dims = s->exec->getTensorShape(s->out_name);
            if (dims.nbDims != 4 || dims.d[1] != 3 || dims.d[2] != h || dims.d[3] != w) {
                s->logger->say(0, "the engine has an unexpected output shape");
                goto fail;
            }
        }
        info->mem_size = s->exec->updateDeviceMemorySizeForShapes();
        info->in_hwc8 = s->engine->getTensorFormat(s->in_name, best_prof) == TensorFormat::kHWC8 ? 1 : 0;
        info->out_hwc8 = s->engine->getTensorFormat(s->out_name, best_prof) == TensorFormat::kHWC8 ? 1 : 0;
    }
    return s;

fail:
    trt_shim_free(s);
    return nullptr;
}

extern "C" int trt_shim_bind(struct trt_shim *s, void *mem, size_t mem_size,
                             void *in_tensor, void *out_tensor)
{
    s->exec->setDeviceMemoryV2(mem, static_cast<int64_t>(mem_size));
    return s->exec->setTensorAddress(s->in_name, in_tensor) &&
           s->exec->setTensorAddress(s->out_name, out_tensor);
}

extern "C" int trt_shim_enqueue(struct trt_shim *s, void *stream)
{
    return s->exec->enqueueV3(static_cast<cudaStream_t>(stream));
}

extern "C" void trt_shim_free(struct trt_shim *s)
{
    if (!s)
        return;
    void (*release)(void *) = s->release;
    destroy(s->exec);
    destroy(s->engine);
    destroy(s->runtime);
    // (The logger is trivially destructible apart from its vtable.)
    release(s->logger);
    release(s);
}

extern "C" void trt_shim_version(int *major, int *minor, int *patch)
{
    *major = NV_TENSORRT_MAJOR;
    *minor = NV_TENSORRT_MINOR;
    *patch = NV_TENSORRT_PATCH;
}
