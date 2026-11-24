#!/usr/bin/env bash
# Builds mpv with the native RIFE filter (vf_rife) into build/.
#   ./build.sh            configure if needed, then compile
#   ./build.sh --clean    start from an empty build directory
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA="$ROOT/toolchain/cuda-wheels/nvidia/cu13"   # nvcc + cuda.h (CUDA 13.4)
TRT_INC="$ROOT/toolchain/trt-src/include"        # TensorRT 11.4 headers
TRT_LIB="$ROOT/lib"                              # libnvinfer, libnvonnxparser

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$ROOT/build"
fi

# The wheels ship the compiler without its executable bits.
chmod +x "$CUDA/bin/nvcc" "$CUDA/bin/cudafe++" "$CUDA/bin/ptxas" "$CUDA/nvvm/bin/cicc" 2>/dev/null || true

if [[ ! -f "$ROOT/build/build.ninja" ]]; then
    # mpv finds TensorRT next to itself (../lib from build/), wherever this
    # folder is moved to.
    # DT_RPATH rather than DT_RUNPATH: TensorRT dlopen()s its builder resources by
    # bare name, and only the old tag is honoured for lookups made by a library.
    RPATH='-Wl,--disable-new-dtags,-rpath,$ORIGIN/../lib'
    meson setup "$ROOT/build" "$ROOT/mpv" \
        --buildtype=release \
        -Dc_link_args="$RPATH" -Dcpp_link_args="$RPATH" \
        -Dcuda-hwaccel=enabled -Dcuda-interop=enabled \
        -Dvulkan=enabled -Dx11=enabled -Dlua=luajit \
        -Drife-trt=enabled \
        -Dnvcc="$CUDA/bin/nvcc" \
        -Dcuda-include="$CUDA/include" \
        -Dtensorrt-include="$TRT_INC" \
        -Dtensorrt-libdir="$TRT_LIB" \
        -Dlibplacebo:vulkan=enabled -Dlibplacebo:opengl=enabled \
        -Dlibplacebo:tests=false -Dlibplacebo:demos=false
fi

ninja -C "$ROOT/build"
