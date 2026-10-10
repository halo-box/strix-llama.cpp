set(TARGET llama-gemm-prec)
add_executable(${TARGET} gemm-prec.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
