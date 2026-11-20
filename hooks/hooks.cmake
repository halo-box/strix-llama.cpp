# Entry point of the hooks build. No upstream CMakeLists.txt includes it, configure with:
#   cmake -B build -DCMAKE_PROJECT_INCLUDE=<repo>/hooks/hooks.cmake ...
# CMake includes this file after every project() call. The setup runs at the end of the top-level directory,
# when every upstream target exists. Without the option, the build is pure upstream.

include_guard(GLOBAL)

if (MSVC OR APPLE)
    message(WARNING "hooks: link-time wrapping needs a GNU compatible linker (ld, gold, lld, mold): hooks disabled")
    return()
endif()

set(HOOKS_DIR ${CMAKE_CURRENT_LIST_DIR})
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
