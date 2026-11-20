# ggml layer of the hooks: registry, backend wrapper, drivers and kernels. Included by hooks/hooks.cmake.
# Kernel files are globbed: add a file under kernels/<backend>/ and re-run cmake.

set(GGML_HOOKS_DIR ${HOOKS_DIR}/ggml)
get_target_property(GGML_HOOKS_GGML_SRC ggml-base SOURCE_DIR) # ggml/src

# registry and generic backend wrapper: in ggml-base, shared by all backends, tests and plugins
target_sources(ggml-base PRIVATE ${GGML_HOOKS_DIR}/src/ggml-hooks-registry.cpp ${GGML_HOOKS_DIR}/src/ggml-hooks-wrap.cpp
                                 ${GGML_HOOKS_DIR}/src/ggml-hooks-types.c)
target_include_directories(ggml-base PUBLIC $<BUILD_INTERFACE:${GGML_HOOKS_DIR}/include>)
# hooks enabled, for every user of ggml: ggml-hook-points.h (included by ggml.h) and the headers included by the
# hook lines of common and server are empty or no-ops without it, so the same tree builds as pure upstream
target_compile_definitions(ggml-base PUBLIC GGML_HOOKS)
target_link_libraries(ggml-base PRIVATE ${CMAKE_DL_LIBS})

# activation: the public functions that list devices and create backends install the wrapper first
target_sources(ggml PRIVATE ${GGML_HOOKS_DIR}/src/ggml-hooks-activate.cpp)
target_include_directories(ggml PRIVATE ${GGML_HOOKS_GGML_SRC})
hooks_wrap(ggml API SYMBOLS
    ggml_backend_reg_count ggml_backend_reg_get ggml_backend_reg_by_name
    ggml_backend_dev_count ggml_backend_dev_get ggml_backend_dev_by_name ggml_backend_dev_by_type
    ggml_backend_init_by_name ggml_backend_init_by_type ggml_backend_init_best
    ggml_backend_load ggml_backend_load_all ggml_backend_load_all_from_path)

# CPU, every variant: the variants are created from ggml/src or from ggml/src/ggml-cpu
if (GGML_CPU)
    file(GLOB GGML_HOOK_CPU_KERNELS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/cpu/*.cpp)
    get_property(cpu_targets     DIRECTORY ${GGML_HOOKS_GGML_SRC}          PROPERTY BUILDSYSTEM_TARGETS)
    get_property(cpu_targets_sub DIRECTORY ${GGML_HOOKS_GGML_SRC}/ggml-cpu PROPERTY BUILDSYSTEM_TARGETS)
    foreach (target ${cpu_targets} ${cpu_targets_sub})
        get_target_property(type ${target} TYPE)
        # skip the helper targets of GGML_CPU_ALL_VARIANTS (feature detection objects)
        if (target MATCHES "^ggml-cpu" AND NOT target MATCHES "feats" AND type MATCHES "LIBRARY")
            target_include_directories(${target} PRIVATE ${GGML_HOOKS_GGML_SRC}/ggml-cpu)
            hooks_link(${target} UNITS ${GGML_HOOKS_DIR}/src/ggml-hooks-cpu.cpp ${GGML_HOOK_CPU_KERNELS})
        endif()
    endforeach()
endif()

# CUDA, HIP, MUSA
file(GLOB GGML_HOOK_CUDA_KERNELS CONFIGURE_DEPENDS ${GGML_HOOKS_DIR}/kernels/cuda/*.cu)
set(GGML_HOOK_CUDA_SOURCES ${GGML_HOOKS_DIR}/src/ggml-hooks-cuda.cu ${GGML_HOOK_CUDA_KERNELS})
foreach (target ggml-cuda ggml-hip ggml-musa)
    if (NOT TARGET ${target})
        continue()
    endif()
    target_include_directories(${target} PRIVATE ${GGML_HOOKS_GGML_SRC}/ggml-cuda)
    hooks_link(${target} UNITS ${GGML_HOOK_CUDA_SOURCES})
    if (target STREQUAL "ggml-hip")
        set_source_files_properties(${GGML_HOOK_CUDA_SOURCES} TARGET_DIRECTORY ggml-hip PROPERTIES LANGUAGE HIP)
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
endif()

# tests: cmake --build build --target test-ggml-hooks
add_executable(test-ggml-hooks EXCLUDE_FROM_ALL ${HOOKS_DIR}/tests/test-ggml-hooks.cpp)
target_link_libraries(test-ggml-hooks PRIVATE ggml)
target_compile_features(test-ggml-hooks PRIVATE cxx_std_17)
# the test declares a backend extension: it needs the registrar of the executable
hooks_link(test-ggml-hooks UNITS)
