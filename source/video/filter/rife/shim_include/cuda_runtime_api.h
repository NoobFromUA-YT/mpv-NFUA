// Stand-in for the CUDA runtime header when trt_shim.cpp is compiled free
// standing (Windows): TensorRT's headers only need these two handle types.
#pragma once
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
