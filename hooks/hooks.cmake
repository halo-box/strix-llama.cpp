# Entry point of the hooks build, included by HOOK(hooks) in the top CMakeLists.txt after project(). The setup runs at
# the end of the top-level directory, when every upstream target exists. cmake -DLLAMA_HOOKS=OFF builds pure upstream.

include_guard(GLOBAL)

option(LLAMA_HOOKS "build the hooks (OFF: pure upstream)" ON)
if (NOT LLAMA_HOOKS)
    message(STATUS "hooks: off (LLAMA_HOOKS=OFF), pure upstream")
    return()
endif()

if (MSVC OR APPLE)
    message(STATUS "hooks: link-time wrapping needs a GNU compatible linker (ld, gold, lld, mold): hooks disabled")
    return()
endif()

set(HOOKS_DIR ${CMAKE_CURRENT_LIST_DIR})

# options of the fork (ggml/CMakeLists.txt of the fork)
option(GGML_EXL3                            "ggml: build the EXL3 (exllamav3) trellis decoder" OFF)

# the fork builds gfx1151 when no HIP arch is given (ggml/src/ggml-hip/CMakeLists.txt of the fork). Empty: the upstream
# default (every arch of the ROCm install).
set(GGML_HOOKS_HIP_DEFAULT_ARCH "gfx1151" CACHE STRING "hooks: HIP arch when GPU_TARGETS and CMAKE_HIP_ARCHITECTURES are not set")
if (GGML_HIP AND GGML_HOOKS_HIP_DEFAULT_ARCH AND NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT GPU_TARGETS AND NOT AMDGPU_TARGETS)
    set(CMAKE_HIP_ARCHITECTURES ${GGML_HOOKS_HIP_DEFAULT_ARCH})
endif()
include(${HOOKS_DIR}/cmake/hooks-link.cmake)

function(hooks_setup)
    if (NOT TARGET ggml-base)
        message(WARNING "hooks: no ggml-base target, hooks disabled")
        return()
    endif()
    include(${HOOKS_DIR}/ggml/ggml-hooks.cmake)
    if (TARGET llama)
        include(${HOOKS_DIR}/llama/llama-hooks.cmake)
    endif()
    # for hooks/tools/check-wraps.sh
    get_property(wraps GLOBAL PROPERTY HOOKS_WRAPPED_SYMBOLS)
    set(content "")
    foreach (w ${wraps})
        string(REPLACE ":" " " w "${w}")
        string(APPEND content "${w}\n")
    endforeach()
    file(WRITE ${CMAKE_BINARY_DIR}/hooks-wraps.txt "${content}")
    message(STATUS "hooks: enabled (${HOOKS_DIR})")
endfunction()

cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL hooks_setup)
