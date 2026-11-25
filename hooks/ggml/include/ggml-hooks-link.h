#pragma once

// Registration and static builds, see hooks/cmake/hooks-link.cmake.
//
// GGML_HOOK_DESCRIPTOR puts the address of a constant descriptor in an ELF section of the library (one section per
// kind). Each library with hook files gets a generated registrar, compiled without the ISA flags of the target:
// - ggml_hooks_module_<id>: the module token. The registrar registers the descriptors of its library with it and
//   removes them when the library is unloaded (a GGML_BACKEND_DL module that is not chosen, a plugin).
// - ggml_hooks_keep_<id>: references every hook file of the target. The link of the target's users gets
//   -u ggml_hooks_keep_<id>, so a static library keeps its hook files.
// Each hook file is compiled with GGML_HOOK_UNIT=<file id>, its anchor is defined below.

#define GGML_HOOK_CONCAT_(a, b) a##b
#define GGML_HOOK_CONCAT(a, b) GGML_HOOK_CONCAT_(a, b)

// host code only: the device pass of a CUDA/HIP file would emit the descriptor in device code.
// ggml-cuda/vendors/hip.h and musa.h define __CUDA_ARCH__ in host code too.
#if (defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)) || (!defined(__HIP__) && !defined(__MUSACC__) && defined(__CUDA_ARCH__)) || defined(__MUSA_ARCH__)
#define GGML_HOOK_DESCRIPTOR(sec_name, var) static_assert(true, "")
#else
#define GGML_HOOK_DESCRIPTOR(sec_name, var) \
    __attribute__((used, section(#sec_name))) static const void * const GGML_HOOK_CONCAT(var, _ptr) = &var
#endif

#ifdef GGML_HOOK_UNIT
extern "C" {
int GGML_HOOK_CONCAT(ggml_hook_unit_, GGML_HOOK_UNIT) = 0;
}
#endif

// the module of the library, for run-time registrations
#ifdef GGML_HOOKS_KEEP
extern "C" const char GGML_HOOK_CONCAT(ggml_hooks_module_, GGML_HOOKS_KEEP);
#define GGML_HOOK_MODULE ((const void *) &GGML_HOOK_CONCAT(ggml_hooks_module_, GGML_HOOKS_KEEP))
#else
#define GGML_HOOK_MODULE nullptr
#endif
