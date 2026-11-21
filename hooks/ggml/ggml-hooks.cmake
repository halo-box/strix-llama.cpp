# ggml layer of the hooks: registry, backend wrapper, drivers and kernels. Included by hooks/hooks.cmake.
# Kernel files are globbed: add a file under kernels/<backend>/ and re-run cmake.

set(GGML_HOOKS_DIR ${HOOKS_DIR}/ggml)
get_target_property(GGML_HOOKS_GGML_SRC ggml-base SOURCE_DIR) # ggml/src

# registry and generic backend wrapper: in ggml-base, shared by all backends, tests and plugins
target_sources(ggml-base PRIVATE ${GGML_HOOKS_DIR}/src/ggml-hooks-registry.cpp ${GGML_HOOKS_DIR}/src/ggml-hooks-wrap.cpp
                                 ${GGML_HOOKS_DIR}/types/types.c ${GGML_HOOKS_DIR}/types/ggml-quants-ext.c
                                 ${GGML_HOOKS_DIR}/types/rocmfp4/rocmfp4.c ${GGML_HOOKS_DIR}/types/rocmfpx/rocmfpx.c
                                 ${GGML_HOOKS_DIR}/exl3/ggml-exl3-ops.c ${GGML_HOOKS_DIR}/src/ggml-fork-ops.c)
# EXL3: the decoder library of the fork, as ggml/src/CMakeLists.txt of the fork
if (GGML_EXL3)
    target_sources(ggml-base PRIVATE ${GGML_HOOKS_DIR}/exl3/exl3.c)
endif()
target_include_directories(ggml-base PUBLIC $<BUILD_INTERFACE:${GGML_HOOKS_DIR}/include>)
target_include_directories(ggml-base PRIVATE ${GGML_HOOKS_DIR}/types)
# hooks enabled, for every user of ggml: ggml-hook-points.h (included by ggml.h) and the headers included by the
# hook lines of common and server are empty or no-ops without it, so the same tree builds as pure upstream
target_compile_definitions(ggml-base PUBLIC GGML_HOOKS)
target_link_libraries(ggml-base PRIVATE ${CMAKE_DL_LIBS})

# the fork_sched switch: the scheduler, graph allocator and sanitizer of the fork beside the upstream ones
# (hooks/ggml/src/ggml-fork-sched.cpp). The upstream ggml-backend.cpp is compiled through ggml-backend-upstream.cpp.
include(${GGML_HOOKS_DIR}/src/fork-sched/symbols.cmake)
set(fork_sched_dir ${GGML_HOOKS_DIR}/src/fork-sched)
target_sources(ggml-base PRIVATE ${fork_sched_dir}/fork-alloc.c ${fork_sched_dir}/fork-backend.cpp
                                 ${fork_sched_dir}/ggml-backend-sanitize.cpp ${GGML_HOOKS_DIR}/src/ggml-fork-sched.cpp
                                 ${GGML_HOOKS_DIR}/src/ggml-backend-upstream.cpp)
set_source_files_properties(${GGML_HOOKS_GGML_SRC}/ggml-backend.cpp TARGET_DIRECTORY ggml-base PROPERTIES HEADER_FILE_ONLY ON)
foreach (sym ${GGML_FORK_SCHED_SYMBOLS})
    set_property(SOURCE ${fork_sched_dir}/fork-alloc.c ${fork_sched_dir}/fork-backend.cpp TARGET_DIRECTORY ggml-base
                 APPEND PROPERTY COMPILE_DEFINITIONS "${sym}=fork_${sym}")
endforeach()
hooks_rename(ggml-base SOURCE ${GGML_HOOKS_GGML_SRC}/ggml-alloc.c SYMBOLS ${GGML_FORK_SCHED_ALLOC_UPSTREAM})
hooks_rename(ggml-base SOURCE ${GGML_HOOKS_DIR}/src/ggml-backend-upstream.cpp SYMBOLS ${GGML_FORK_SCHED_BACKEND_UPSTREAM})

# the fork ftypes, for the users of ggml-base
hooks_rename(ggml-base SOURCE ${GGML_HOOKS_GGML_SRC}/ggml.c SYMBOLS ggml_ftype_to_ggml_type ggml_dsv4_hc_pre ggml_dsv4_hc_pre_gated)

# activation: the public functions that list devices and create backends install the wrapper first
target_sources(ggml PRIVATE ${GGML_HOOKS_DIR}/src/ggml-hooks-activate.cpp)
target_include_directories(ggml PRIVATE ${GGML_HOOKS_GGML_SRC})
hooks_rename(ggml SOURCE ${GGML_HOOKS_GGML_SRC}/ggml-backend-reg.cpp SYMBOLS
    ggml_backend_reg_count ggml_backend_reg_get ggml_backend_reg_by_name
    ggml_backend_dev_count ggml_backend_dev_get ggml_backend_dev_by_name ggml_backend_dev_by_type
    ggml_backend_init_by_name ggml_backend_init_by_type ggml_backend_init_best
    ggml_backend_load ggml_backend_load_all ggml_backend_load_all_from_path)

# CPU, every variant: the variants are created from ggml/src or from ggml/src/ggml-cpu
if (GGML_CPU)
    file(GLOB GGML_HOOK_CPU_KERNELS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/cpu/*.cpp)
    file(GLOB GGML_HOOK_CPU_TYPES CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/types/cpu/*.c ${GGML_HOOKS_DIR}/kernels/cpu/*.c)
    get_property(cpu_targets     DIRECTORY ${GGML_HOOKS_GGML_SRC}          PROPERTY BUILDSYSTEM_TARGETS)
    get_property(cpu_targets_sub DIRECTORY ${GGML_HOOKS_GGML_SRC}/ggml-cpu PROPERTY BUILDSYSTEM_TARGETS)
    foreach (target ${cpu_targets} ${cpu_targets_sub})
        get_target_property(type ${target} TYPE)
        # skip the helper targets of GGML_CPU_ALL_VARIANTS (feature detection objects)
        if (target MATCHES "^ggml-cpu" AND NOT target MATCHES "feats" AND type MATCHES "LIBRARY")
            target_include_directories(${target} PRIVATE ${GGML_HOOKS_GGML_SRC}/ggml-cpu ${GGML_HOOKS_DIR}/types)
            # CPU traits of the hook types (C files: the trait table of ggml-cpu.c references them, static links keep them)
            target_sources(${target} PRIVATE ${GGML_HOOK_CPU_TYPES})
            set(units ${GGML_HOOKS_DIR}/src/ggml-hooks-cpu.cpp ${GGML_HOOK_CPU_KERNELS})
            if (GGML_EXL3)
                # as ggml/src/ggml-cpu/CMakeLists.txt of the fork
                target_compile_definitions(${target} PRIVATE GGML_USE_EXL3)
                target_sources(${target} PRIVATE ${GGML_HOOKS_DIR}/kernels/cpu-exl3/exl3-ops.cpp)
                list(APPEND units ${GGML_HOOKS_DIR}/kernels/cpu-exl3/fork-exl3-cpu.cpp)
            endif()
            hooks_link(${target} UNITS ${units})
            hooks_rename(${target} SOURCE ${GGML_HOOKS_GGML_SRC}/ggml-cpu/ggml-cpu.cpp SYMBOLS ggml_backend_cpu_init)
        endif()
    endforeach()
endif()

# CUDA, HIP, MUSA
option(GGML_HOOKS_FORK_CUDA "hooks: build the fork ggml-cuda code into ggml-hip" ON)
file(GLOB GGML_HOOK_CUDA_KERNELS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/cuda/*.cu)
list(FILTER GGML_HOOK_CUDA_KERNELS EXCLUDE REGEX "/fork-cuda\\.cu$") # the bridge to the fork code, HIP only (below)
set(GGML_HOOK_CUDA_SOURCES ${GGML_HOOKS_DIR}/src/ggml-hooks-cuda.cu ${GGML_HOOK_CUDA_KERNELS})
foreach (target ggml-cuda ggml-hip ggml-musa)
    if (NOT TARGET ${target})
        continue()
    endif()
    target_include_directories(${target} PRIVATE ${GGML_HOOKS_GGML_SRC}/ggml-cuda)
    set(units ${GGML_HOOK_CUDA_SOURCES})
    if (target STREQUAL "ggml-hip" AND GGML_HOOKS_FORK_CUDA)
        list(APPEND units ${GGML_HOOKS_DIR}/kernels/cuda/fork-cuda.cu)
    endif()
    hooks_link(${target} UNITS ${units})
    # the upstream ggml-cuda.cu is compiled through ggml-cuda-upstream.cu, which reaches its static functions
    set(cuda_upstream ${GGML_HOOKS_DIR}/src/ggml-cuda-upstream.cu)
    target_sources(${target} PRIVATE ${cuda_upstream})
    set_source_files_properties(${GGML_HOOKS_GGML_SRC}/ggml-cuda/ggml-cuda.cu TARGET_DIRECTORY ${target} PROPERTIES HEADER_FILE_ONLY ON)
    hooks_rename(${target} SOURCE ${cuda_upstream} SYMBOLS ggml_backend_cuda_init)
    if (target STREQUAL "ggml-hip")
        set_source_files_properties(${units} ${cuda_upstream} TARGET_DIRECTORY ggml-hip PROPERTIES LANGUAGE HIP)
    elseif (target STREQUAL "ggml-musa")
        # same as ggml-musa/CMakeLists.txt: .cu files built as CXX with the MUSA compiler flags
        set_source_files_properties(${GGML_HOOK_CUDA_SOURCES} TARGET_DIRECTORY ggml-musa PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-x musa -mtgpu")
    endif()
endforeach()

# Vulkan: shaders are compiled with glslc and embedded as headers
if (TARGET ggml-vulkan)
    file(GLOB GGML_HOOK_VK_KERNELS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/vulkan/*.cpp)
    file(GLOB GGML_HOOK_VK_SHADERS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/vulkan/*.comp)
    if (NOT Vulkan_GLSLC_EXECUTABLE)
        find_program(Vulkan_GLSLC_EXECUTABLE glslc REQUIRED)
    endif()
    set(gen_dir ${CMAKE_BINARY_DIR}/hooks-vk-shaders)
    set(gen_headers "")
    foreach (shader ${GGML_HOOK_VK_SHADERS})
        get_filename_component(name ${shader} NAME_WE)
        set(spv ${gen_dir}/${name}.spv)
        set(hdr ${gen_dir}/${name}.spv.h)
        add_custom_command(
            OUTPUT  ${hdr}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${gen_dir}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -fshader-stage=compute --target-env=vulkan1.2 -O -I${GGML_HOOKS_GGML_SRC}/ggml-vulkan/vulkan-shaders ${shader} -o ${spv}
            COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${hdr} -DNAME=ggml_hook_spv_${name} -P ${GGML_HOOKS_DIR}/cmake/embed-spv.cmake
            DEPENDS ${shader} ${GGML_HOOKS_DIR}/cmake/embed-spv.cmake
            COMMENT "hooks: compiling shader ${name}"
        )
        list(APPEND gen_headers ${hdr})
    endforeach()
    # generated files are only visible in the directory of the rule: build them with a target of this directory
    add_custom_target(hooks-vk-shaders DEPENDS ${gen_headers})
    add_dependencies(ggml-vulkan hooks-vk-shaders)
    target_include_directories(ggml-vulkan PRIVATE ${GGML_HOOKS_GGML_SRC}/ggml-vulkan ${gen_dir})
    hooks_link(ggml-vulkan UNITS ${GGML_HOOKS_DIR}/src/ggml-hooks-vk.cpp ${GGML_HOOK_VK_KERNELS})
    hooks_rename(ggml-vulkan SOURCE ${GGML_HOOKS_GGML_SRC}/ggml-vulkan/ggml-vulkan.cpp SYMBOLS ggml_backend_vk_init)
endif()

# tests: cmake --build build --target test-ggml-hooks
add_executable(test-ggml-hooks EXCLUDE_FROM_ALL ${HOOKS_DIR}/tests/test-ggml-hooks.cpp)
target_link_libraries(test-ggml-hooks PRIVATE ggml)
target_compile_features(test-ggml-hooks PRIVATE cxx_std_17)
# the test declares a backend extension: it needs the registrar of the executable
hooks_link(test-ggml-hooks UNITS)

# the EXL3 decode self-test of the fork (tests/CMakeLists.txt of the fork), next to the decoder
if (GGML_EXL3 AND TARGET vendor::hash)
    add_executable(test-exl3 ${GGML_HOOKS_DIR}/exl3/test_exl3.c)
    target_link_libraries(test-exl3 PRIVATE ggml vendor::hash)
    target_include_directories(test-exl3 PRIVATE ${GGML_HOOKS_DIR}/exl3 ${CMAKE_SOURCE_DIR}/vendor)
    add_test(NAME test-exl3 COMMAND test-exl3)
endif()

# HIP: the fork ggml-cuda sources (hooks/ggml/fork-cuda, namespace ggml_fork) in the upstream backend, the same list as
# ggml/src/ggml-hip/CMakeLists.txt of the fork. Hook kernels and the fork_cuda switch run them on the upstream streams
# (hooks/ggml/kernels/cuda/fork-cuda.cu).
if (TARGET ggml-hip AND GGML_HOOKS_FORK_CUDA)
    set(fork_cuda_dir ${GGML_HOOKS_DIR}/fork-cuda)
    file(GLOB fork_cuda_srcs ${fork_cuda_dir}/*.cu)
    foreach (pattern fattn-tile fattn-mma mmq mmf)
        file(GLOB srcs ${fork_cuda_dir}/template-instances/${pattern}*.cu)
        list(APPEND fork_cuda_srcs ${srcs})
    endforeach()
    ggml_cuda_fattn_vec_instances(${fork_cuda_dir} srcs)
    list(APPEND fork_cuda_srcs ${srcs})
    if (NOT GGML_EXL3)
        list(FILTER fork_cuda_srcs EXCLUDE REGEX "/fork-cuda/exl3\\.cu$")
    endif()
    set(fork_cuda_defs GGML_USE_TRELLIS GGML_HOOKS_FORK_CUDA)
    if (GGML_EXL3)
        list(APPEND fork_cuda_defs GGML_USE_EXL3)
    endif()
    target_sources(ggml-hip PRIVATE ${fork_cuda_srcs})
    # the bridge (a hook unit) includes the fork headers
    set(bridge ${GGML_HOOKS_DIR}/kernels/cuda/fork-cuda.cu)
    set_property(SOURCE ${bridge} TARGET_DIRECTORY ggml-hip APPEND PROPERTY COMPILE_DEFINITIONS ${fork_cuda_defs})
    set_property(SOURCE ${bridge} TARGET_DIRECTORY ggml-hip APPEND PROPERTY INCLUDE_DIRECTORIES ${GGML_HOOKS_DIR}/types)
    set_source_files_properties(${fork_cuda_srcs} TARGET_DIRECTORY ggml-hip PROPERTIES LANGUAGE HIP
        COMPILE_DEFINITIONS "${fork_cuda_defs}" INCLUDE_DIRECTORIES "${GGML_HOOKS_DIR}/types")
    list(LENGTH fork_cuda_srcs n)
    message(STATUS "hooks: fork ggml-cuda in ggml-hip, ${n} sources")
endif()
