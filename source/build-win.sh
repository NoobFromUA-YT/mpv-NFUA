#!/usr/bin/env bash
# Cross-compiles the Windows build of mpv with the native RIFE filter into build-win/
# and, with --install DIR, puts a ready-to-run folder together there.
#   ./build-win.sh
#   ./build-win.sh --install /path/to/output/MPV-NFUA
# Needs toolchain/win/ (llvm-mingw, the MSYS2 CLANG64 sysroot from msys2_fetch.py and
# the Windows TensorRT wheel unpacked in trt/); see README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$ROOT/toolchain/win"
CUDA="$ROOT/toolchain/cuda-wheels/nvidia/cu13"

if [[ ! -f "$ROOT/build-win/build.ninja" ]]; then
    meson setup "$ROOT/build-win" "$ROOT/mpv" --cross-file "$W/cross.ini" \
        --buildtype=release --wrap-mode=nodownload \
        -Dcuda-hwaccel=enabled -Dcuda-interop=enabled -Dvulkan=enabled -Dlua=luajit \
        -Drife-trt=enabled \
        -Dnvcc="$CUDA/bin/nvcc" -Dcuda-include="$CUDA/include" \
        -Dtensorrt-include="$ROOT/toolchain/trt-src/include" \
        -Drife-shim-cxx="$W/llvm-mingw/bin/clang++"
fi
ninja -C "$ROOT/build-win"

if [[ "${1:-}" == "--install" ]]; then
    DEST="${2:?--install needs a folder}"
    mkdir -p "$DEST/portable_config/rife" "$DEST/portable_config/scripts" \
             "$DEST/portable_config/script-opts" "$DEST/portable_config/cache"
    cp "$ROOT/build-win/mpv.exe" "$ROOT/build-win/mpv.com" "$DEST/"
    EXES=("$ROOT/build-win/mpv.exe")
    if [[ -f "$ROOT/build-win/video/filter/rife/rife-bench.exe" ]]; then
        cp "$ROOT/build-win/video/filter/rife/rife-bench.exe" "$DEST/"
        EXES+=("$ROOT/build-win/video/filter/rife/rife-bench.exe")
    fi
    # Every DLL mpv.exe and rife-bench.exe need from the sysroot, found by following imports.
    python3 - "$W" "$DEST" "${EXES[@]}" <<'PY'
import os, re, shutil, subprocess, sys
W, dest = sys.argv[1:3]
todo = list(sys.argv[3:])
objdump, bins = W + '/llvm-mingw/bin/x86_64-w64-mingw32-objdump', W + '/sysroot/clang64/bin'
avail = {f.lower(): f for f in os.listdir(bins) if f.lower().endswith('.dll')}
need = {}
while todo:
    out = subprocess.run([objdump, '-p', todo.pop()], capture_output=True, text=True).stdout
    for d in (m.lower() for m in re.findall(r'DLL Name: (\S+)', out)):
        if d in avail and d not in need:
            need[d] = os.path.join(bins, avail[d]); todo.append(need[d])
for p in need.values():
    shutil.copy2(p, dest)
print(len(need), 'DLLs copied')
PY
    cp "$W"/trt/tensorrt_libs/nvinfer_11.dll "$W"/trt/tensorrt_libs/nvonnxparser_11.dll \
       "$W"/trt/tensorrt_libs/nvinfer_builder_resource_*_11.dll "$DEST/"
    cp "$ROOT"/portable_config/rife/rife_v4.6.onnx "$ROOT"/portable_config/rife/rife_v4.6_half.onnx \
       "$DEST/portable_config/rife/"
    cp "$ROOT/portable_config/scripts/rife.lua" "$ROOT/portable_config/scripts/look.lua" \
       "$ROOT/portable_config/scripts/speedstep.lua" "$DEST/portable_config/scripts/"
    # mpv.conf, input.conf, rife.conf, osc.conf in DEST are kept if they exist already.
    cp --update=none "$ROOT/portable_config/input.conf" "$DEST/portable_config/"
    cp --update=none "$ROOT/portable_config/script-opts/rife.conf" "$DEST/portable_config/script-opts/"
    cp --update=none "$ROOT/portable_config/script-opts/osc.conf" "$DEST/portable_config/script-opts/"
    cp --update=none "$ROOT/windows/mpv.conf" "$DEST/portable_config/"
    cp "$ROOT/windows/README-WINDOWS.txt" "$ROOT/windows/test-rife.bat" "$DEST/"
    echo "installed to $DEST"
fi
