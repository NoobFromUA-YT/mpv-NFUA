# vf_rife: RIFE frame interpolation on TensorRT, entirely in CUDA device memory.
#
# The device code is compiled to a fatbin here and embedded as a string, so the only
# thing needed at runtime besides TensorRT is the NVIDIA driver itself.

nvcc = find_program(get_option('nvcc'), required: true)
rife_cuda_inc = get_option('cuda-include')
rife_trt_inc = get_option('tensorrt-include')
rife_trt_libdir = get_option('tensorrt-libdir')

rife_inc_args = []
foreach dir: [rife_cuda_inc, rife_trt_inc]
    if dir != ''
        rife_inc_args += ['-isystem', dir]
    endif
endforeach

rife_ptx = custom_target('rife_kernels.ptx',
    input: 'rife_kernels.cu',
    output: 'rife_kernels.ptx',
    depend_files: 'rife_kernel_args.h',
    # A fatbin of ready-made GPU code for RTX 20 to RTX 50 (sm_75 is the oldest
    # architecture TensorRT still runs on). No PTX: PTX from this nvcc would need
    # a driver as new as nvcc's CUDA version to translate it, while ready-made
    # code runs on any driver of the same CUDA major version (595 included).
    command: [nvcc, '-fatbin', '-O3',
              '-gencode=arch=compute_75,code=sm_75',
              '-gencode=arch=compute_80,code=sm_80',
              '-gencode=arch=compute_86,code=sm_86',
              '-gencode=arch=compute_89,code=sm_89',
              '-gencode=arch=compute_120,code=sm_120',
              '-o', '@OUTPUT@', '@INPUT@'],
)
rife_ptx_inc = custom_target('rife_kernels.ptx.inc',
    input: rife_ptx,
    output: 'rife_kernels.ptx.inc',
    command: [file2string, '@INPUT@', '@OUTPUT@', '@SOURCE_ROOT@'],
)

sources += rife_ptx_inc
sources += files('rife_kernels_ptx.c', 'rife_trt.cpp')
sources += files('../vf_rife.c')

if win32
    # TensorRT's DLLs speak the MSVC C++ ABI, mpv is a MinGW program. The one
    # file that touches TensorRT's classes is compiled for that ABI, with no
    # runtime library behind it (see trt_shim.h), and linked in as an object.
    # The DLLs themselves are opened at runtime.
    shim_cxx = find_program(get_option('rife-shim-cxx'), required: true)
    shim_inc = join_paths(meson.current_source_dir(), 'shim_include')
    trt_shim_obj = custom_target('trt_shim.obj',
        input: 'trt_shim.cpp',
        output: 'trt_shim.obj',
        depend_files: ['trt_shim.h', 'shim_include/cstddef', 'shim_include/cstdint',
                       'shim_include/cuda_runtime_api.h'],
        command: [shim_cxx, '--target=x86_64-pc-windows-msvc', '-c', '-O2', '-std=c++17',
                  '-ffreestanding', '-nostdinc++', '-fno-exceptions', '-fno-rtti',
                  '-fno-stack-protector', '-fno-threadsafe-statics',
                  '-DTRT_SHIM_FREESTANDING', '-isystem', shim_inc,
                  '-isystem', rife_trt_inc, '-o', '@OUTPUT@', '@INPUT@'],
    )
    sources += trt_shim_obj
    dependencies += declare_dependency(compile_args: rife_inc_args)
else
    cxx = meson.get_compiler('cpp')
    nvinfer = cxx.find_library('nvinfer',
                               dirs: rife_trt_libdir != '' ? [rife_trt_libdir] : [],
                               required: true)
    nvonnxparser = cxx.find_library('nvonnxparser',
                                    dirs: rife_trt_libdir != '' ? [rife_trt_libdir] : [],
                                    required: true)
    sources += files('trt_shim.cpp')
    dependencies += [nvinfer, nvonnxparser,
                     declare_dependency(compile_args: rife_inc_args)]
endif

# rife-bench: times the network on this GPU for the performance assessor in
# portable_config/scripts/rife.lua. Same engine, kernels and CUDA graph as the
# filter, but no window or video needed.
if win32
    executable('rife-bench',
        rife_ptx_inc, trt_shim_obj,
        files('rife_bench.cpp', 'rife_kernels_ptx.c', 'rife_trt.cpp'),
        include_directories: include_directories('../../..'),
        dependencies: [declare_dependency(compile_args: rife_inc_args)],
        build_by_default: true,
    )
else
    executable('rife-bench',
        rife_ptx_inc, files('rife_bench.cpp', 'rife_kernels_ptx.c', 'rife_trt.cpp',
                            'trt_shim.cpp'),
        include_directories: include_directories('../../..'),
        dependencies: [nvinfer, nvonnxparser, libdl,
                       declare_dependency(compile_args: rife_inc_args)],
        # TensorRT is in superMPV/lib, four levels up from build/video/filter/rife.
        link_args: ['-Wl,--disable-new-dtags,-rpath,$ORIGIN/../../../../lib'],
        build_by_default: true,
    )
endif
