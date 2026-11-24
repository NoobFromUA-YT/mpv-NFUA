mpv with native RIFE v4.6 frame interpolation - Windows 10 build
=================================================================

The Windows twin of /home/noobfromua/superMPV on the Kubuntu side: mpv 0.42 with the
built-in "rife" video filter (NVDEC surfaces in, one CUDA graph launch per new frame on
TensorRT 11.4, CUDA surfaces out). Cross-compiled on Linux on 2026-11-10.

IT HAS NOT RUN ON WINDOWS YET. See "What was and was not checked".


Running it
----------
  mpv.exe            double-click, then drop a video on the window; or "Open with"
  test-rife.bat      drag a video onto it: plays it and writes rife-test.log with what
                     the filter did. Use this for the first try.
  mpv.com            the same player for a terminal (prints to the console)

Needs: an NVIDIA GPU (RTX 20 series or newer) with a current driver. Nothing to install.

The first video compiles the TensorRT engine for this GPU: about half a minute to a
minute, once per flow mode (two in total). The video plays at its original frame rate
meanwhile and RIFE joins in when the engine is ready. Engines are kept in
portable_config\rife\ (*.engine). Quitting during the compile abandons it.

Keys (same as on Linux)
  Alt+0            automatic
  Alt+1            RIFE off
  Alt+2 .. Alt+6   2x .. 6x
  Alt+7            HDR look (HDR video only)
  Alt+8            trim mode: always / nearest / no
  Alt+9            measure this PC again
  i                mpv's statistics (fps, dropped frames)

Settings: portable_config\mpv.conf, portable_config\script-opts\rife.conf.
All filter options: mpv.com --vf=rife=help


What was and was not checked
----------------------------
Checked on Linux, by running this very mpv.exe under Wine (Proton):
  - it starts, every DLL it needs is in this folder, the "rife" filter and its options
    are there, the config and rife.lua load
  - the CUDA driver functions it needs are found by name in nvcuda.dll
  - FFmpeg creates a CUDA device and a software-decoded frame is uploaded
  - nvinfer_11.dll and nvonnxparser_11.dll load, their entry points are found, and
    TensorRT's messages come back through the filter's logger (this crosses the
    MSVC/MinGW boundary described below, so that part of it works)
The same source, built for Linux, is the player in daily use there and is tested.

NOT checked, because TensorRT itself cannot start under Wine (its CUDA check fails
there with error 35):
  - compiling and loading the engine, and running it
  - the CUDA graph capture
  - NVDEC decoding and the Vulkan display path (no window was ever opened)
  - speed, dropped frames
Engines from Linux are not reused here; this build compiles its own.


If it does not work
-------------------
Run test-rife.bat and read rife-test.log (lines with [rife]).

  "cannot open nvcuda.dll"            NVIDIA driver missing or very old
  "cannot open nvinfer_11.dll ..."    the TensorRT DLLs are not next to mpv.exe
  "... lacks cuGraph... (driver too old)"   update the NVIDIA driver
  "TensorRT: ... CUDA initialization failure"   driver older than CUDA 13 needs;
                                      update the NVIDIA driver
  video plays but stays at original fps: the filter failed and mpv went on without it;
                                      the log says why
  black window / no picture:          try  mpv.com --gpu-api=opengl <file>
                                      (Vulkan is set in mpv.conf because Direct3D 11,
                                      mpv's Windows default, cannot take CUDA frames)
  a crash right after "compiling ...": most likely in the MSVC/MinGW shim below

To rule the filter in or out:   mpv.com --vf= <file>        (no RIFE at all)
Without the CUDA graph:         mpv.com --vf=rife=graph=no <file>
Software decoding:              mpv.com --hwdec=no <file>


How it is built (for whoever continues)
---------------------------------------
source\ holds the filter's source and the changes to mpv:
  mpv-changes.patch            what was changed in mpv's own files (4 files)
  video\filter\vf_rife.c       the filter: frame pairing, timestamps, surfaces
  video\filter\rife\rife_trt.cpp     CUDA side: kernels, graph capture/launch, engine
                                     cache; has the Windows/Linux differences
  video\filter\rife\trt_shim.cpp/.h  every use of TensorRT's C++ classes
  video\filter\rife\rife_kernels.cu  the CUDA kernels (compiled to PTX, embedded)
  build-win.sh, cross.ini, msys2_fetch.py   how this folder was produced

The one unusual thing: TensorRT's DLLs are built with MSVC, mpv with MinGW (clang), and
the two disagree about C++ (virtual tables, struct returns). So trt_shim.cpp, the only
file that touches TensorRT's classes, is compiled separately with
  clang++ --target=x86_64-pc-windows-msvc -ffreestanding -nostdinc++ ...
(MSVC ABI, no runtime library, tiny stand-in headers in rife\shim_include) and linked
into mpv.exe as an object. Everything else talks to it through the plain C functions in
trt_shim.h. The TensorRT DLLs are opened with LoadLibrary on first use.

It was cross-compiled on Linux with llvm-mingw 20261104 against MSYS2 CLANG64 packages
(FFmpeg 9.0.2, libplacebo 7.387). To rebuild there:
  cd /home/noobfromua/superMPV && ./build-win.sh --install /run/media/noobfromua/Dawn7/MPV
Building on Windows itself should work the same way in an MSYS2 CLANG64 shell (install
the mingw-w64-clang-x86_64 packages of ffmpeg, libplacebo, libass, luajit, meson, ninja,
clang) with the meson options from build-win.sh; that has not been tried. The kernels
need nvcc (CUDA toolkit) only at build time.

Licences: mpv is GPL/LGPL (source: github.com/mpv-player/mpv plus source\). The
nvinfer*.dll files are NVIDIA TensorRT and under NVIDIA's licence.
