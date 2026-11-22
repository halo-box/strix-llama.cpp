# Entry point of the hooks build, included by HOOK(hooks) in the top CMakeLists.txt after project(). The setup runs at
# the end of the top-level directory, when every upstream target exists. cmake -DLLAMA_HOOKS=OFF builds pure upstream.

include_guard(GLOBAL)

option(LLAMA_HOOKS "build the hooks (OFF: pure upstream)" ON)
if (NOT LLAMA_HOOKS)
    message(STATUS "hooks: off (LLAMA_HOOKS=OFF), pure upstream")
    return()
endif()

if (WIN32 OR APPLE)
    # lld-link, link.exe and ld64 have no --wrap: the HOOK points stay inert, the build is upstream
    message(STATUS "hooks: link-time wrapping needs an ELF linker (ld, gold, lld, mold): hooks disabled")
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

# every target of the build, in every directory
function(hooks_all_targets dir out)
    get_property(targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    get_property(subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach (sub ${subdirs})
        hooks_all_targets(${sub} sub_targets)
        list(APPEND targets ${sub_targets})
    endforeach()
    set(${out} ${targets} PARENT_SCOPE)
endfunction()

# The warnings that the hooks build adds by design, for LLAMA_FATAL_WARNINGS and -Werror builds:
# - -Wswitch in the ggml targets: the hook types raise GGML_TYPE_COUNT and add an enum value, so a switch of upstream
#   or of the fork code that lists every upstream type is no longer complete (the hook types reach its default path);
# - -Wmissing-declarations and -Wdeprecated-declarations in the sources of hooks/: the __wrap_* functions and the API
#   dispatchers are looked up by name only, and the dispatchers call the deprecated batch API of the fork on purpose.
function(hooks_warning_options)
    hooks_all_targets(${CMAKE_SOURCE_DIR} targets)
    foreach (t ${targets})
        get_target_property(type ${t} TYPE)
        if (type STREQUAL "INTERFACE_LIBRARY" OR type STREQUAL "UTILITY")
            continue()
        endif()
        if (t MATCHES "^ggml")
            target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:C,CXX,HIP,CUDA>:-Wno-switch>)
        endif()
        get_target_property(srcs ${t} SOURCES)
        get_target_property(tdir ${t} SOURCE_DIR)
        set(own "")
        foreach (s ${srcs})
            if (s MATCHES "^\\$<")
                continue()
            endif()
            if (NOT IS_ABSOLUTE ${s})
                set(s ${tdir}/${s})
            endif()
            cmake_path(NORMAL_PATH s)
            if (s MATCHES "^${HOOKS_DIR}/" AND s MATCHES "\\.(c|cc|cpp|cu)$")
                list(APPEND own ${s})
            endif()
        endforeach()
        if (own)
            set_property(SOURCE ${own} TARGET_DIRECTORY ${t} APPEND PROPERTY COMPILE_OPTIONS
                         -Wno-missing-declarations -Wno-deprecated-declarations)
        endif()
    endforeach()
endfunction()

function(hooks_setup)
    if (NOT TARGET ggml-base)
        message(WARNING "hooks: no ggml-base target, hooks disabled")
        return()
    endif()
    include(${HOOKS_DIR}/ggml/ggml-hooks.cmake)
    if (TARGET llama)
        include(${HOOKS_DIR}/llama/llama-hooks.cmake)
    endif()
    hooks_warning_options()
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
