find_package(hrx CONFIG REQUIRED)
find_package(loomc CONFIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(GGML_HRX_KERNELS_CPP "${CMAKE_CURRENT_BINARY_DIR}/ggml-hrx-kernels.cpp")

file(GLOB_RECURSE GGML_HRX_KERNEL_FILES CONFIGURE_DEPENDS "${GGML_HRX_KERNEL_DIR}/*.loom")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/kernels")
foreach(k IN ITEMS 2304 3072)
    foreach(groups IN ITEMS 1 2)
        set(kernel "${generated_dir}/npu/gemm_i8_k${k}_g${groups}.loom")
        add_custom_command(
            OUTPUT "${kernel}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${generated_dir}/npu"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/npu/generate-gemm.py" ${k} ${groups} "${kernel}"
            DEPENDS npu/generate-gemm.py npu/gemm-i8-core.loom.inc
            VERBATIM)
        list(APPEND GGML_HRX_KERNEL_FILES "${kernel}")
    endforeach()
endforeach()

add_custom_command(
    OUTPUT  "${GGML_HRX_KERNELS_CPP}"
    COMMAND "${CMAKE_COMMAND}"
            -DKERNEL_DIR=${GGML_HRX_KERNEL_DIR}
            -DGENERATED_KERNEL_DIR=${generated_dir}
            -DOUTPUT=${GGML_HRX_KERNELS_CPP}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/embed-kernels.cmake"
    DEPENDS ${GGML_HRX_KERNEL_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/embed-kernels.cmake"
    COMMENT "Embedding HRX Loom kernels"
    VERBATIM
)

file(GLOB GGML_HRX_SOURCES CONFIGURE_DEPENDS "*.cpp" "ops/*.cpp")
file(GLOB GGML_HRX_HEADERS CONFIGURE_DEPENDS "*.h")

ggml_add_backend_library(ggml-hrx
    ${GGML_HRX_HEADERS}
    ${GGML_HRX_SOURCES}
    "${GGML_HRX_KERNELS_CPP}"
    ../../include/ggml-hrx.h
)

target_link_libraries(ggml-hrx PRIVATE hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx PRIVATE .)
target_compile_features(ggml-hrx PRIVATE cxx_std_17)
