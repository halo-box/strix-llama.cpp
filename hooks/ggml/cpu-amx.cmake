# Included by HOOK(cpu-amx) in ggml/src/ggml-cpu/CMakeLists.txt, in ggml_add_cpu_backend_variant_impl, with or without
# the hooks build (Windows builds AMX with clang too). The fork lines of 704c814da, verbatim: clang 22 miscompiles the
# unrolled AMX mmq loops.

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22 AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23)
    set_source_files_properties(ggml-cpu/amx/mmq.cpp PROPERTIES COMPILE_OPTIONS "$<$<BOOL:${MSVC}>:/clang:>-fno-unroll-loops")
endif()

set_property(GLOBAL PROPERTY HOOKS_CPU_AMX_INCLUDED TRUE) # checked by hooks_setup (hooks/hooks.cmake)
