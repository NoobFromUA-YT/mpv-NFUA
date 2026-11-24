# mpv-NFUA

**Native CUDA + TensorRT RIFE v4.6 frame interpolation built directly into `mpv 0.42` — for Windows & Linux.**

No VapourSynth, no Python frame server, no CPU round-trips: NVDEC hardware-decoded frames stay in CUDA memory through colour conversion, scene-cut detection, TensorRT inference (captured into a single CUDA graph per frame), and Vulkan/OpenGL zero-copy presentation.

- **Project Page:** https://noobfromua-yt.github.io/mpv-NFUA/
- **Latest Release:** https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest
- **1-Click Windows Installer (~17 KB):** https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest/download/Install-MPV-NFUA.bat

---

## Requirements

- **GPU:** NVIDIA RTX 20 series or newer (Turing / Ampere / Ada / Blackwell)
- **Driver:** Current NVIDIA driver with CUDA 13 support (e.g. R580+ / 596.21 tested)

---

## Installation

### Windows (1-Click Web Installer — Recommended)

1. Download **[Install-MPV-NFUA.bat](https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest/download/Install-MPV-NFUA.bat)** (~17 KB).
2. Double-click it and pick an install folder (defaults to `%LOCALAPPDATA%\MPV-NFUA`).
3. The installer automatically:
   - Downloads the slim portable `mpv-NFUA` runtime (**85 MB** `MPV-NFUA-Windows-Portable.zip`).
   - Queries your GPU architecture via `nvidia-smi` and fetches **only the 3 TensorRT 11.4 DLLs your GPU needs** directly from `pypi.nvidia.com` via HTTP Range requests (**~349 MB** download instead of the 1.86 GB multi-arch wheel, saving ~1.6 GB on disk).
   - Compiles both Windows TensorRT engines (`full` and `half` flow), measures your GPU and monitor refresh rate, and writes `portable_config/script-opts/rife-preset.conf`.
   - Optionally creates a Desktop shortcut and sets `mpv-NFUA` as your default `mpv`.

Total size after setup: **~782 MB** on disk (vs. 2.40 GB with all TensorRT architectures bundled).

Prefer manual download? Grab **[MPV-NFUA-Windows-Portable.zip](https://github.com/NoobFromUA-YT/mpv-NFUA/releases/latest/download/MPV-NFUA-Windows-Portable.zip)** (85 MB) from Releases, extract, and run `Setup-MPV-Antigravity.bat`.

### Linux (Build from `source/`)

Both Windows and Linux share the exact same `vf_rife` filter (`source/video/filter/vf_rife.c`, `source/video/filter/rife/`) and Lua scripts (`source/portable_config/`):

1. Clone upstream mpv (`0.42`, commit `a61f6c3abd`), apply `source/mpv-changes.patch`, and copy `source/video/filter/vf_rife.c` + `source/video/filter/rife/` into `video/filter/`.
2. Install dependencies (`ffmpeg`, `libplacebo >= 7.387.0`, `luajit`, `shaderc`, `vulkan-loader`, CUDA Toolkit `nvcc`, TensorRT 11.4 headers/libs) and run [`source/build.sh`](source/build.sh).
3. Copy `source/portable_config/` to your mpv config folder and place `rife_v4.6.onnx` + `rife_v4.6_half.onnx` in `portable_config/rife/`.

---

## Verified Benchmarks (RTX 4070 Ti, 3840x2160 @ 160 Hz)

Measured via `rife-bench` (20 runs, median GPU time per interpolated frame):

| Resolution | Flow Mode | Tensor Size | Windows (Driver 596.21) | Linux Reference | Auto Multiplier (160 Hz) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **720p** (`1280x720`) | Full | `1280x704` | `2.70 ms` | `2.61 ms` | `24->6x`, `30->5x`, `60->2x` |
| **720p** (`1280x720`) | Half | `1280x704` | `1.82 ms` | `1.84 ms` | (used on high fps) |
| **1080p** (`1920x1080`) | Full | `1920x1056` | `7.42 ms` | `7.46 ms` | — |
| **1080p** (`1920x1080`) | Half | `1920x1024` | `5.04 ms` | `4.99 ms` | `24->6x`, `30->5x`, `60->2x` |
| **1440p** (`2560x1440`) | Full | `2560x1440` | `14.74 ms` | `14.80 ms` | — |
| **1440p** (`2560x1440`) | Half | `2560x1408` | `10.42 ms` | `10.44 ms` | `24->4x`, `30->3x`, `60->2x` |
| **4K** (`3840x2160`) | Half | `3840x2112` | `23.89 ms` | `24.04 ms` | `24->2x`, `30->2x` (`50/60`: off) |

Live Windows playback (`vo=gpu-next`, `gpu-api=vulkan`, `hwdec=nvdec`): **0 dropped frames** across `1080p24` (`6x` = 144 fps, `6.74 ms` GPU), `4K24` (`2x` = 48 fps, `26.3–27.3 ms` GPU), and `4K30` (`2x` = 60 fps, `25.76 ms` GPU).

---

## Hotkeys

| Key | Action |
| :--- | :--- |
| `Alt+0` | Automatic multiplier & flow mode (`rife.lua`) |
| `Alt+1` | Turn RIFE off (original fps) |
| `Alt+2` .. `Alt+6` | Fixed `2x` .. `6x` multiplier |
| `Alt+7` | Cycle HDR tone-mapping look (`look.lua`) |
| `Alt+8` | Cycle height alignment trim mode (`always` / `nearest` / `no`) |
| `Alt+9` | Re-run GPU & refresh rate benchmark (`rife-bench`) |
| `+` / `=` / `-` | Step playback speed in `1x` increments (`speedstep.lua`, auto-rescales RIFE) |
| `MBTN_BACK` / `MBTN_FORWARD` | Toggle fullscreen / seek `+10s` |
| `o` / `щ` | Quit player |
| `i` | mpv stats overlay (fps, dropped frames) |

---

## Architecture Highlights

- **Single CUDA graph per frame:** `rife_pre` (NV12/P010/P012/P016 → fp16 RGB) + TensorRT 11.4 inference + `rife_post` (fp16 RGB → NV12/P010/P012/P016) captured into one `CUgraphExec` (`kHWC8`/`kLINEAR` auto-selected).
- **Dedicated high-priority scene-cut detection:** `rife_scd_cells` + `rife_scd_reduce` run once per frame pair on a high-priority stream; cuts skip inference completely.
- **Pair-level tensor reuse (`timestep_only`):** Multi-frame pairs (`3x`–`6x`) skip re-converting `f0`/`f1` after the first interpolated frame and update only the timestep channel.
- **Graph slicing:** Heavy frames (`4K`) are automatically split into short GPU slices with 1 ms gaps (`slice=6:slice-gap=1`) on a background worker thread so display presentation stays smooth.
- **MSVC / MinGW ABI bridge (`trt_shim.cpp`):** Compiled freestanding with `clang++ --target=x86_64-pc-windows-msvc -nostdinc++` so MinGW-built `mpv.exe` calls directly into MSVC-built `nvinfer_11.dll` / `nvonnxparser_11.dll`.
