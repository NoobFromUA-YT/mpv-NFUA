#!/usr/bin/env bash
# mpv-NFUA Linux 1-Click Smart Installer
# Downloads the slim Linux portable runtime (~24 MB) + ONLY the 3 TensorRT 11.4
# shared libraries needed for your NVIDIA GPU from NVIDIA PyPI via HTTP Range
# requests (~469 MB instead of the 3.64 GB multi-arch wheel), verifies CRC32,
# calibrates rife-preset.conf, and optionally registers mpv-nfua + desktop entry.
set -euo pipefail

DEFAULT_APP_URL="https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest/download/MPV-NFUA-Linux-x86_64.tar.gz"
TRT_WHEEL_URL="https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-11.4.0.106-py3-none-manylinux_2_28_x86_64.whl"
REPO_SOURCE_TAR_URL="https://github.com/NoobFromUA-YT/mpv-NFUA/archive/refs/heads/main.tar.gz"
MPV_UPSTREAM_TAG="a61f6c3abd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_DIR=""
APP_URL="$DEFAULT_APP_URL"
OPT_REGISTER=""
NO_PROMPT=0
FORCE_DOWNLOAD=0
KEEP_ALL_SM=0
SKIP_BENCH=0
BUILD_FROM_SOURCE=0

usage() {
    cat <<'EOF'
Usage: install-mpv-nfua.sh [options]

Options:
  --dir <path>       Install directory (default: script dir if bin/mpv exists,
                     otherwise ~/.local/share/mpv-nfua)
  --app-url <url>    Portable runtime .tar.gz URL or local file path
  --register         Create ~/.local/bin/mpv-nfua symlink and .desktop launcher
  --no-register      Skip desktop/CLI registration
  --no-prompt        Non-interactive mode (accepts defaults)
  --force-download   Re-download runtime and TensorRT libraries even if present
  --keep-all-sm      Keep extra libnvinfer_builder_resource_*.so files if present
  --skip-bench       Skip initial rife-bench GPU/monitor calibration
  --build            Build mpv & rife-bench from source against the fetched TensorRT libs
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --dir" >&2; exit 2; }
            INSTALL_DIR="$2"
            shift 2
            ;;
        --app-url)
            [[ $# -ge 2 ]] || { echo "Missing value for --app-url" >&2; exit 2; }
            APP_URL="$2"
            shift 2
            ;;
        --register)
            OPT_REGISTER="yes"
            shift
            ;;
        --no-register)
            OPT_REGISTER="no"
            shift
            ;;
        --no-prompt)
            NO_PROMPT=1
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=1
            shift
            ;;
        --keep-all-sm)
            KEEP_ALL_SM=1
            shift
            ;;
        --skip-bench)
            SKIP_BENCH=1
            shift
            ;;
        --build)
            BUILD_FROM_SOURCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -t 1 ]]; then
    C_CYAN=$'\033[1;36m'
    C_GREEN=$'\033[1;32m'
    C_YELLOW=$'\033[1;33m'
    C_RED=$'\033[1;31m'
    C_RESET=$'\033[0m'
else
    C_CYAN="" C_GREEN="" C_YELLOW="" C_RED="" C_RESET=""
fi

step() {
    printf "\n%s==> %s%s\n" "$C_CYAN" "$*" "$C_RESET"
}

warn() {
    printf "%sWarning: %s%s\n" "$C_YELLOW" "$*" "$C_RESET" >&2
}

die() {
    printf "%sError: %s%s\n" "$C_RED" "$*" "$C_RESET" >&2
    exit 1
}

download_file() {
    local src="$1" dst="$2" label="$3"
    if [[ "$src" == file://* ]]; then
        src="${src#file://}"
    fi
    if [[ -f "$src" ]]; then
        cp -f "$src" "$dst"
        return 0
    fi
    if command -v curl >/dev/null 2>&1; then
        printf "  Downloading %s...\n" "$label"
        curl -fL --progress-bar "$src" -o "$dst"
    elif command -v wget >/dev/null 2>&1; then
        printf "  Downloading %s...\n" "$label"
        wget -q --show-progress "$src" -O "$dst"
    else
        die "Neither curl nor wget is installed."
    fi
}

# Check required tools
for tool in nvidia-smi python3 tar; do
    command -v "$tool" >/dev/null 2>&1 || die "'$tool' is required but not installed."
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    die "'curl' or 'wget' is required to download runtime files."
fi

# Determine install directory
if [[ -z "$INSTALL_DIR" ]]; then
    if [[ -x "$SCRIPT_DIR/bin/mpv" || -f "$SCRIPT_DIR/bin/mpv" ]]; then
        INSTALL_DIR="$SCRIPT_DIR"
    else
        DEFAULT_DIR="${HOME}/.local/share/mpv-nfua"
        if [[ "$NO_PROMPT" -eq 1 || ! -t 0 ]]; then
            INSTALL_DIR="$DEFAULT_DIR"
        else
            printf "%smpv-NFUA Linux Installer%s\n" "$C_GREEN" "$C_RESET"
            read -r -p "Install folder [$DEFAULT_DIR]: " chosen || true
            chosen="${chosen#"${chosen%%[![:space:]]*}"}"
            chosen="${chosen%"${chosen##*[![:space:]]}"}"
            INSTALL_DIR="${chosen:-$DEFAULT_DIR}"
        fi
    fi
fi

# Expand leading ~ if user typed it
if [[ "$INSTALL_DIR" == "~/"* ]]; then
    INSTALL_DIR="${HOME}/${INSTALL_DIR#~/}"
elif [[ "$INSTALL_DIR" == "~" ]]; then
    INSTALL_DIR="$HOME"
fi
mkdir -p "$INSTALL_DIR"
INSTALL_DIR="$(cd "$INSTALL_DIR" && pwd)"

printf "%smpv-NFUA Linux Setup%s\n" "$C_GREEN" "$C_RESET"
printf "Install folder: %s\n" "$INSTALL_DIR"

step "Checking NVIDIA GPU & driver"
GPU_CSV="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
[[ -n "$GPU_CSV" ]] || die "Failed to query NVIDIA GPU via nvidia-smi."
IFS=',' read -r GPU_NAME COMPUTE_CAP DRIVER_VER <<<"$GPU_CSV"
GPU_NAME="$(echo "$GPU_NAME" | xargs)"
COMPUTE_CAP="$(echo "$COMPUTE_CAP" | xargs)"
DRIVER_VER="$(echo "$DRIVER_VER" | xargs)"
SM_NUM="${COMPUTE_CAP//./}"
SM_TAG="sm${SM_NUM}"

printf "  GPU:         %s\n" "$GPU_NAME"
printf "  Compute Cap: %s (%s)\n" "$COMPUTE_CAP" "$SM_TAG"
printf "  Driver:      %s\n" "$DRIVER_VER"

if [[ "$SM_NUM" =~ ^[0-9]+$ ]] && (( SM_NUM < 75 )); then
    warn "Compute capability $COMPUTE_CAP is older than Turing (7.5); TensorRT 11.4 may fall back to PTX or fail."
fi

if [[ ! -x "$INSTALL_DIR/bin/mpv" || "$FORCE_DOWNLOAD" -eq 1 ]]; then
    step "Downloading mpv-NFUA Linux portable runtime (~24 MB)"
    TMP_TGZ="$(mktemp /tmp/mpv-nfua-linux-XXXXXX.tar.gz)"
    trap 'rm -f "${TMP_TGZ:-}"' EXIT
    download_file "$APP_URL" "$TMP_TGZ" "MPV-NFUA-Linux-x86_64.tar.gz"
    printf "  Extracting to %s...\n" "$INSTALL_DIR"
    tar -xzf "$TMP_TGZ" -C "$INSTALL_DIR"
    rm -f "$TMP_TGZ"
    [[ -x "$INSTALL_DIR/bin/mpv" ]] || die "Extracted archive does not contain bin/mpv."
fi

step "Fetching TensorRT 11.4 shared libraries for ${SM_TAG} from NVIDIA PyPI"
LIB_DIR="$INSTALL_DIR/lib"
mkdir -p "$LIB_DIR"

python3 - "$TRT_WHEEL_URL" "$LIB_DIR" "$SM_TAG" "$FORCE_DOWNLOAD" "$KEEP_ALL_SM" <<'PY'
import os
import re
import struct
import sys
import urllib.request
import zlib

wheel_url, lib_dir, sm_tag, force_s, keep_all_s = sys.argv[1:6]
force = (force_s == "1")
keep_all = (keep_all_s == "1")

def http_req(url, method="GET", headers=None):
    hdrs = {"User-Agent": "mpv-NFUA-Linux-Installer/1.0"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, method=method, headers=hdrs)
    return urllib.request.urlopen(req, timeout=60)

with http_req(wheel_url, method="HEAD") as resp:
    final_url = resp.geturl()
    total_size = int(resp.headers["Content-Length"])

tail_len = min(total_size, 65536)
with http_req(final_url, headers={"Range": f"bytes={total_size - tail_len}-{total_size - 1}"}) as resp:
    tail = resp.read()

eocd = tail.rfind(b"PK\x05\x06")
if eocd < 0:
    raise RuntimeError("ZIP End of Central Directory not found in TensorRT wheel.")

cd_size, cd_off = struct.unpack_from("<II", tail, eocd + 12)
z64loc = tail.rfind(b"PK\x06\x07", 0, eocd)
if z64loc >= 0 and (cd_size == 0xFFFFFFFF or cd_off == 0xFFFFFFFF):
    z64_off = struct.unpack_from("<Q", tail, z64loc + 8)[0]
    with http_req(final_url, headers={"Range": f"bytes={z64_off}-{z64_off + 55}"}) as resp:
        z64 = resp.read()
    cd_size, cd_off = struct.unpack_from("<QQ", z64, 40)

with http_req(final_url, headers={"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"}) as resp:
    cd = resp.read()

entries = {}
pos = 0
while pos < len(cd):
    sig, = struct.unpack_from("<I", cd, pos)
    if sig != 0x02014B50:
        raise RuntimeError(f"Bad Central Directory signature 0x{sig:08x}")
    method, _, _, crc32, comp_sz, uncomp_sz, nlen, elen, clen = struct.unpack_from(
        "<HHHIIIHHH", cd, pos + 10
    )
    lhoff, = struct.unpack_from("<I", cd, pos + 42)
    name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8")
    extra = cd[pos + 46 + nlen : pos + 46 + nlen + elen]
    ep = 0
    while ep + 4 <= len(extra):
        hdr, sz = struct.unpack_from("<HH", extra, ep)
        ep += 4
        if hdr == 0x0001:
            vp = ep
            if uncomp_sz == 0xFFFFFFFF:
                uncomp_sz, = struct.unpack_from("<Q", extra, vp); vp += 8
            if comp_sz == 0xFFFFFFFF:
                comp_sz, = struct.unpack_from("<Q", extra, vp); vp += 8
            if lhoff == 0xFFFFFFFF:
                lhoff, = struct.unpack_from("<Q", extra, vp); vp += 8
        ep += sz
    entries[name] = {
        "name": name,
        "base": os.path.basename(name),
        "method": method,
        "crc32": crc32,
        "comp_sz": comp_sz,
        "uncomp_sz": uncomp_sz,
        "lhoff": lhoff,
    }
    pos += 46 + nlen + elen + clen

builder_exact = f"tensorrt_libs/libnvinfer_builder_resource_{sm_tag}.so.11.4.0"
builder_ptx = "tensorrt_libs/libnvinfer_builder_resource_ptx.so.11.4.0"
builder_key = builder_exact if builder_exact in entries else builder_ptx
if builder_key not in entries:
    raise RuntimeError(f"No builder resource found in wheel for {sm_tag}.")

wanted_keys = [
    "tensorrt_libs/libnvinfer.so.11",
    "tensorrt_libs/libnvonnxparser.so.11",
    builder_key,
]
wanted = [entries[k] for k in wanted_keys]
comp_sum_mb = sum(e["comp_sz"] for e in wanted) / (1024 * 1024)
wheel_mb = total_size / (1024 * 1024)
print(f"  Streaming 3/{len(entries)} entries ({comp_sum_mb:.1f} MB vs {wheel_mb:.0f} MB full wheel)...")

for e in wanted:
    dest = os.path.join(lib_dir, e["base"])
    if not force and os.path.isfile(dest) and os.path.getsize(dest) == e["uncomp_sz"]:
        print(f"  [skip] {e['base']} ({e['uncomp_sz'] / (1024*1024):.1f} MB already present)")
        continue

    with http_req(final_url, headers={"Range": f"bytes={e['lhoff']}-{e['lhoff'] + 29}"}) as resp:
        lfh = resp.read()
    lsig, = struct.unpack_from("<I", lfh, 0)
    if lsig != 0x04034B50:
        raise RuntimeError(f"Invalid ZIP local header for {e['base']}")
    lnlen, lelen = struct.unpack_from("<HH", lfh, 26)
    data_start = e["lhoff"] + 30 + lnlen + lelen
    data_end = data_start + e["comp_sz"] - 1

    tmp_dest = dest + ".tmp"
    decomp = zlib.decompressobj(-15) if e["method"] == 8 else None
    if e["method"] not in (0, 8):
        raise RuntimeError(f"Unsupported ZIP compression method {e['method']} for {e['base']}")

    crc = 0
    written = 0
    downloaded = 0
    comp_mb = e["comp_sz"] / (1024 * 1024)
    tty = sys.stdout.isatty()
    last_len = -1

    with http_req(final_url, headers={"Range": f"bytes={data_start}-{data_end}"}) as resp:
        with open(tmp_dest, "wb") as out:
            while True:
                chunk = resp.read(262144)
                if not chunk:
                    break
                downloaded += len(chunk)
                plain = decomp.decompress(chunk) if decomp else chunk
                if plain:
                    out.write(plain)
                    crc = zlib.crc32(plain, crc)
                    written += len(plain)
                if tty:
                    pct = 100.0 * downloaded / e["comp_sz"]
                    msg = f"  {e['base']:45s} {downloaded/(1024*1024):6.1f}/{comp_mb:5.1f} MB ({pct:5.1f}%)"
                    if len(msg) != last_len:
                        sys.stdout.write("\r" + msg)
                        sys.stdout.flush()
                        last_len = len(msg)
            if decomp:
                tail_plain = decomp.flush()
                if tail_plain:
                    out.write(tail_plain)
                    crc = zlib.crc32(tail_plain, crc)
                    written += len(tail_plain)

    if written != e["uncomp_sz"]:
        os.remove(tmp_dest)
        raise RuntimeError(f"Size mismatch for {e['base']}: got {written}, expected {e['uncomp_sz']}")
    if (crc & 0xFFFFFFFF) != e["crc32"]:
        os.remove(tmp_dest)
        raise RuntimeError(f"CRC32 mismatch for {e['base']}: got 0x{crc & 0xFFFFFFFF:08x}, expected 0x{e['crc32']:08x}")

    os.chmod(tmp_dest, 0o755)
    os.replace(tmp_dest, dest)
    prefix = "\r" if tty else ""
    print(f"{prefix}  {e['base']:45s} {written/(1024*1024):6.1f} MB [CRC32 OK]        ")

for link_name, target in [("libnvinfer.so", "libnvinfer.so.11"), ("libnvonnxparser.so", "libnvonnxparser.so.11")]:
    link_path = os.path.join(lib_dir, link_name)
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(target, link_path)

if not keep_all:
    wanted_builder_base = os.path.basename(builder_key)
    for fname in os.listdir(lib_dir):
        if fname.startswith("libnvinfer_builder_resource_") and fname != wanted_builder_base:
            os.remove(os.path.join(lib_dir, fname))
PY

build_from_source() {
    step "Building mpv-NFUA & rife-bench from source against $LIB_DIR"
    for cmd in git meson ninja nvcc pkg-config curl; do
        command -v "$cmd" >/dev/null 2>&1 || die "Source build requires '$cmd'."
    done

    local src_root=""
    local trt_inc="${TENSORRT_INC:-}"
    local cuda_inc="${CUDA_INC:-}"
    local work_dir
    work_dir="$(mktemp -d /tmp/mpv-nfua-build-XXXXXX)"
    trap 'rm -rf "'"$work_dir"'"' RETURN

    if [[ -f "$SCRIPT_DIR/source/mpv-changes.patch" && -d "$SCRIPT_DIR/source/video/filter/rife" ]]; then
        src_root="$SCRIPT_DIR/source"
    else
        printf "  Downloading mpv-NFUA source tree...\n"
        curl -fL "$REPO_SOURCE_TAR_URL" -o "$work_dir/repo.tar.gz"
        tar -xzf "$work_dir/repo.tar.gz" -C "$work_dir"
        src_root="$(find "$work_dir" -maxdepth 2 -type d -name source | head -n 1)"
        [[ -n "$src_root" ]] || die "Could not locate source/ directory in downloaded archive."
    fi

    if [[ -z "$cuda_inc" ]]; then
        local nvcc_dir
        nvcc_dir="$(dirname "$(command -v nvcc)")"
        for cand in "$nvcc_dir/../include" /usr/local/cuda/include /usr/include; do
            if [[ -f "$cand/cuda.h" ]]; then
                cuda_inc="$cand"
                break
            fi
        done
    fi
    [[ -n "$cuda_inc" && -f "$cuda_inc/cuda.h" ]] || die "cuda.h not found. Set CUDA_INC=/path/to/cuda/include."

    if [[ -z "$trt_inc" ]]; then
        for cand in \
            "$SCRIPT_DIR/toolchain/trt-src/include" \
            "$INSTALL_DIR/include" \
            /usr/include/x86_64-linux-gnu \
            /usr/local/TensorRT/include \
            /usr/include; do
            if [[ -f "$cand/NvInfer.h" ]]; then
                trt_inc="$cand"
                break
            fi
        done
    fi
    [[ -n "$trt_inc" && -f "$trt_inc/NvInfer.h" ]] || die "NvInfer.h (TensorRT 11.4 headers) not found. Install TensorRT dev headers or set TENSORRT_INC=/path/to/include."

    printf "  Cloning mpv (commit %s)...\n" "$MPV_UPSTREAM_TAG"
    git clone --filter=blob:none --no-checkout https://github.com/mpv-player/mpv.git "$work_dir/mpv"
    git -C "$work_dir/mpv" checkout "$MPV_UPSTREAM_TAG"
    git -C "$work_dir/mpv" apply "$src_root/mpv-changes.patch"
    cp -a "$src_root/video/filter/"* "$work_dir/mpv/video/filter/"

    local rpath_arg='-Wl,--disable-new-dtags,-rpath,$ORIGIN/../lib'
    meson setup "$work_dir/build" "$work_dir/mpv" \
        --buildtype=release \
        -Dc_link_args="$rpath_arg" -Dcpp_link_args="$rpath_arg" \
        -Dcuda-hwaccel=enabled -Dcuda-interop=enabled \
        -Dvulkan=enabled -Dx11=enabled -Dlua=luajit \
        -Drife-trt=enabled \
        -Dnvcc="$(command -v nvcc)" \
        -Dcuda-include="$cuda_inc" \
        -Dtensorrt-include="$trt_inc" \
        -Dtensorrt-libdir="$LIB_DIR" \
        -Dlibplacebo:vulkan=enabled -Dlibplacebo:opengl=enabled \
        -Dlibplacebo:tests=false -Dlibplacebo:demos=false

    ninja -C "$work_dir/build" mpv video/filter/rife/rife-bench
    mkdir -p "$INSTALL_DIR/bin"
    cp -f "$work_dir/build/mpv" "$INSTALL_DIR/bin/mpv"
    cp -f "$work_dir/build/video/filter/rife/rife-bench" "$INSTALL_DIR/bin/rife-bench"
    if command -v patchelf >/dev/null 2>&1; then
        patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$INSTALL_DIR/bin/mpv"
        patchelf --force-rpath --set-rpath '$ORIGIN/../lib' "$INSTALL_DIR/bin/rife-bench"
    fi
    chmod 0755 "$INSTALL_DIR/bin/mpv" "$INSTALL_DIR/bin/rife-bench"
    printf "  Built and installed bin/mpv and bin/rife-bench.\n"
}

if [[ "$BUILD_FROM_SOURCE" -eq 1 ]]; then
    build_from_source
fi

step "Verifying runtime dependencies"
if ! "$INSTALL_DIR/bin/mpv" --version >/dev/null 2>&1; then
    warn "Prebuilt bin/mpv could not start on this distribution (it targets Ubuntu 26.04+ with glibc 2.43, FFmpeg 8, libplacebo 360)."
    ldd "$INSTALL_DIR/bin/mpv" 2>&1 | grep -E "not found|GLIBC" >&2 || true
    "$INSTALL_DIR/bin/mpv" --version 2>&1 | head -n 5 >&2 || true
    die "Run again with --build (requires git, meson, ninja, nvcc, FFmpeg/libplacebo/Vulkan/LuaJIT dev packages, and TensorRT 11.4 headers)."
fi
printf "  bin/mpv: %s\n" "$("$INSTALL_DIR/bin/mpv" --version | head -n 1)"

if [[ "$SKIP_BENCH" -eq 0 ]]; then
    step "Calibrating TensorRT engines & GPU speed (this runs once)"
    RIFE_DIR="$INSTALL_DIR/portable_config/rife"
    MODEL_FULL="$RIFE_DIR/rife_v4.6.onnx"
    MODEL_HALF="$RIFE_DIR/rife_v4.6_half.onnx"
    [[ -f "$MODEL_FULL" ]] || die "Missing model: $MODEL_FULL"
    [[ -f "$MODEL_HALF" ]] || die "Missing model: $MODEL_HALF"

    printf "  Compiling TensorRT engines and timing 720p..4K via rife-bench...\n"
    BENCH_OUT="$("$INSTALL_DIR/bin/rife-bench" --model "$MODEL_FULL" --model-half "$MODEL_HALF" --runs 10)"
    printf "%s\n" "$BENCH_OUT" | sed 's/^/  /'

    BENCH_GPU="$GPU_NAME"
    FULL_PTS=()
    HALF_PTS=()
    while IFS= read -r line; do
        if [[ "$line" =~ ^gpu[[:space:]]+(.+)$ ]]; then
            BENCH_GPU="${BASH_REMATCH[1]}"
        elif [[ "$line" =~ ^result[[:space:]]+(full|half)[[:space:]]+[^[:space:]]+[[:space:]]+([0-9.]+)[[:space:]]+([0-9.]+)$ ]]; then
            pt="${BASH_REMATCH[2]}:${BASH_REMATCH[3]}"
            if [[ "${BASH_REMATCH[1]}" == "full" ]]; then
                FULL_PTS+=("$pt")
            else
                HALF_PTS+=("$pt")
            fi
        fi
    done <<<"$BENCH_OUT"

    [[ "${#HALF_PTS[@]}" -gt 0 ]] || die "rife-bench produced no benchmark results."

    SCREEN_FPS=""
    if command -v xrandr >/dev/null 2>&1; then
        xr_rate="$(xrandr --current 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?\*' | head -n 1 | tr -d '*' || true)"
        if [[ -n "$xr_rate" ]]; then
            xr_int="$(python3 -c "import sys; print(int(round(float(sys.argv[1]))))" "$xr_rate" 2>/dev/null || true)"
            if [[ -n "$xr_int" ]] && (( xr_int >= 20 )); then SCREEN_FPS="$xr_int"; fi
        fi
    fi
    if [[ -z "$SCREEN_FPS" ]] && command -v wlr-randr >/dev/null 2>&1; then
        wr_rate="$(wlr-randr 2>/dev/null | awk '/current/ {for(i=1;i<=NF;i++) if($i~/Hz/) print $(i-1)}' | head -n 1 || true)"
        if [[ -n "$wr_rate" ]]; then
            wr_int="$(python3 -c "import sys; print(int(round(float(sys.argv[1]))))" "$wr_rate" 2>/dev/null || true)"
            if [[ -n "$wr_int" ]] && (( wr_int >= 20 )); then SCREEN_FPS="$wr_int"; fi
        fi
    fi
    if [[ -z "$SCREEN_FPS" ]] && command -v kscreen-doctor >/dev/null 2>&1; then
        kr_rate="$(kscreen-doctor -o 2>/dev/null | grep -oE '@[0-9]+(\.[0-9]+)?\*' | head -n 1 | tr -d '@*' || true)"
        if [[ -n "$kr_rate" ]]; then
            kr_int="$(python3 -c "import sys; print(int(round(float(sys.argv[1]))))" "$kr_rate" 2>/dev/null || true)"
            if [[ -n "$kr_int" ]] && (( kr_int >= 20 )); then SCREEN_FPS="$kr_int"; fi
        fi
    fi
    SCREEN_FPS="${SCREEN_FPS:-160}"

    PRESET_DIR="$INSTALL_DIR/portable_config/script-opts"
    mkdir -p "$PRESET_DIR"
    PRESET_PATH="$PRESET_DIR/rife-preset.conf"
    STAMP="$(date '+%Y-%m-%d %H:%M')"
    FULL_JOINED="$(IFS=,; echo "${FULL_PTS[*]}")"
    HALF_JOINED="$(IFS=,; echo "${HALF_PTS[*]}")"

    cat >"$PRESET_PATH" <<EOF
# Written by mpv-NFUA Linux setup on $STAMP.
# Alt+9 in the player measures again; delete this file to measure at the next start.
# A setting in rife.conf wins over the same one here.

measured=$STAMP
gpu=$BENCH_GPU
screen_fps=$SCREEN_FPS

# GPU time for one interpolated frame, as tensor megapixels:ms, at full and
# half flow resolution (sizes 1280x720,1920x1080,2560x1440,3840x2160).
full=$FULL_JOINED
half=$HALF_JOINED
EOF
    printf "  Wrote %s (screen_fps=%s Hz)\n" "$PRESET_PATH" "$SCREEN_FPS"
fi

DO_REGISTER=0
if [[ "$OPT_REGISTER" == "yes" ]]; then
    DO_REGISTER=1
elif [[ "$OPT_REGISTER" != "no" && "$NO_PROMPT" -eq 0 && -t 0 ]]; then
    read -r -p $'\nCreate ~/.local/bin/mpv-nfua command and desktop shortcut? [y/N]: ' ans || true
    if [[ "$ans" =~ ^[Yy] ]]; then
        DO_REGISTER=1
    fi
fi

if [[ "$DO_REGISTER" -eq 1 ]]; then
    step "Creating ~/.local/bin/mpv-nfua and desktop entries"
    mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
    ln -sf "$INSTALL_DIR/mpv-nfua" "$HOME/.local/bin/mpv-nfua"
    printf "  Command:  %s/.local/bin/mpv-nfua\n" "$HOME"

    DESKTOP_FILE="$HOME/.local/share/applications/mpv-nfua.desktop"
    cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=mpv-NFUA
GenericName=Media Player
Comment=Zero-copy CUDA + TensorRT RIFE v4.6 frame interpolation player
Exec="$INSTALL_DIR/mpv-nfua" %U
Icon=mpv
Terminal=false
Categories=AudioVideo;Video;Player;TV;
MimeType=application/ogg;application/x-ogg;application/mxf;application/sdp;application/smil;application/x-smil;application/streamingmedia;application/x-streamingmedia;application/vnd.rn-realmedia;application/vnd.rn-realmedia-vbr;audio/aac;audio/x-aac;audio/vnd.dolby.heaac.1;audio/vnd.dolby.heaac.2;audio/aiff;audio/x-aiff;audio/m4a;audio/x-m4a;application/x-extension-m4a;audio/mp1;audio/x-mp1;audio/mp2;audio/x-mp2;audio/mp3;audio/x-mp3;audio/mpeg;audio/mpeg2;audio/mpeg3;audio/mpegurl;audio/x-mpegurl;audio/mpg;audio/x-mpg;audio/rn-mpeg;audio/ogg;audio/scpls;audio/x-scpls;audio/vnd.rn-realaudio;audio/wav;audio/x-pn-windows-pcm;audio/x-realaudio;audio/x-pn-realaudio;audio/x-ms-wma;audio/x-pls;audio/x-wav;video/mpeg;video/x-mpeg2;video/x-mpeg3;video/mp4v-es;video/x-m4v;video/mp4;application/x-extension-mp4;video/divx;video/vnd.divx;video/msvideo;video/x-msvideo;video/ogg;video/quicktime;video/vnd.rn-realvideo;video/x-ms-asf;video/x-ms-wmv;video/x-msvideo;video/x-flv;video/webm;video/x-matroska;application/x-matroska;video/3gpp;video/3gpp2;video/vnd.mpegurl;video/dv;video/x-ogm;video/x-theora+ogg;
StartupWMClass=mpv
X-KDE-Protocols=ftp,http,https,mms,rtmp,rtsp,sftp,smb,srt,rist
EOF
    printf "  App menu: %s\n" "$DESKTOP_FILE"

    if [[ -d "$HOME/Desktop" ]]; then
        cp -f "$DESKTOP_FILE" "$HOME/Desktop/mpv-NFUA.desktop"
        chmod +x "$HOME/Desktop/mpv-NFUA.desktop"
        printf "  Desktop:  %s/Desktop/mpv-NFUA.desktop\n" "$HOME"
    fi

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$HOME/.local/share/applications" >/dev/null 2>&1 || true
    fi
fi

step "Done!"
printf "  Launch GUI:  %s/mpv-nfua\n" "$INSTALL_DIR"
printf "  Play file:   %s/mpv-nfua \"/path/to/video.mkv\"\n" "$INSTALL_DIR"
if [[ "$DO_REGISTER" -eq 1 ]]; then
    printf "  CLI alias:   mpv-nfua \"/path/to/video.mkv\" (ensure ~/.local/bin is in PATH)\n"
fi
