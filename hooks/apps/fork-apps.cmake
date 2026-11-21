# The fork tools (hooks/apps): common, mtmd, server, llama-bench and quantize as namespaced copies (ggml_fork), and the
# fork example speculative-prefill. The fork_tools switch (GGML_HOOKS=+fork_tools, off) runs the fork tools: their entry
# functions (llama_server, llama_bench, llama_quantize) are renamed *__upstream in their upstream files and
# fork-tools-dispatch-*.cpp calls one side. The fork tools call the fork libllama directly.
# Included by llama-hooks.cmake, with the fork libllama.

set(fa ${HOOKS_DIR}/apps)

# the sources of a copied tree, without the programs (a file with a main)
function(hooks_fork_sources out dir)
    file(GLOB_RECURSE srcs CONFIGURE_DEPENDS ${dir}/*.cpp)
    set(res "")
    foreach (src ${srcs})
        file(STRINGS ${src} mains REGEX "^int main\\(")
        if (NOT mains)
            list(APPEND res ${src})
        endif()
    endforeach()
    set(${out} ${res} PARENT_SCOPE)
endfunction()

# common
if (TARGET llama-common)
    hooks_fork_sources(fc_srcs ${fa}/fork-common)
    if (NOT GGML_EXL3)
        list(FILTER fc_srcs EXCLUDE REGEX "/exl3-dir\\.cpp$")
    endif()
    add_library(hooks-fork-common OBJECT ${fc_srcs})
    target_include_directories(hooks-fork-common PRIVATE ${fa}/fork-common ${HOOKS_DIR}/llama/fork-include
                                                         ${HOOKS_DIR}/llama/fork-src ${HOOKS_DIR}/llama/include)
    target_link_libraries(hooks-fork-common PRIVATE ggml vendor::nlohmann vendor::sheredom cpp-httplib)
    target_compile_definitions(hooks-fork-common PRIVATE $<TARGET_PROPERTY:llama-common,COMPILE_DEFINITIONS>)
    if (GGML_EXL3)
        target_link_libraries(hooks-fork-common PRIVATE vendor::hash)
        target_compile_definitions(hooks-fork-common PRIVATE LLAMA_USE_EXL3)
    endif()
    target_compile_features(hooks-fork-common PRIVATE cxx_std_17)
    set_target_properties(hooks-fork-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(llama-common PRIVATE $<TARGET_OBJECTS:hooks-fork-common>)
    target_link_libraries(llama-common PRIVATE vendor::hash) # the fork common hashes files
    set(fork_common_inc ${fa}/fork-common ${HOOKS_DIR}/llama/fork-include ${HOOKS_DIR}/llama/fork-src ${HOOKS_DIR}/llama/include)
endif()

# a fork tool: its sources in an object library linked into the upstream impl library, and its entry dispatched
function(hooks_fork_tool name impl upstream_src entry)
    if (NOT TARGET ${impl} OR NOT TARGET hooks-fork-common)
        return()
    endif()
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LIBS" ${ARGN})
    add_library(hooks-fork-${name} OBJECT ${ARG_SOURCES})
    target_include_directories(hooks-fork-${name} PRIVATE ${ARG_INCLUDES} ${fork_common_inc})
    target_link_libraries(hooks-fork-${name} PRIVATE ggml vendor::nlohmann vendor::sheredom cpp-httplib ${ARG_LIBS})
    target_compile_definitions(hooks-fork-${name} PRIVATE $<TARGET_PROPERTY:llama-common,COMPILE_DEFINITIONS>
                                                          $<TARGET_PROPERTY:${impl},COMPILE_DEFINITIONS>)
    target_compile_features(hooks-fork-${name} PRIVATE cxx_std_17)
    set_target_properties(hooks-fork-${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(${impl} PRIVATE $<TARGET_OBJECTS:hooks-fork-${name}> ${fa}/fork-tools-dispatch-${name}.cpp)
    hooks_rename(${impl} SOURCE ${upstream_src} SYMBOLS ${entry})
endfunction()

hooks_fork_sources(fs_srcs ${fa}/fork-server)
hooks_fork_sources(fm_srcs ${fa}/fork-mtmd)
if (TARGET llama-server-impl)
    # the stats dashboard of the fork server, embedded as in tools/server/CMakeLists.txt of the fork
    set(dash_hpp ${CMAKE_BINARY_DIR}/hooks-fork-server/dashboard.html.hpp)
    add_custom_command(OUTPUT ${dash_hpp}
        COMMAND ${CMAKE_COMMAND} "-DINPUT=${fa}/fork-server/dashboard/dashboard.html" "-DOUTPUT=${dash_hpp}"
                "-DNAME=dashboard_html" -P "${fa}/fork-server/dashboard/embed.cmake"
        DEPENDS ${fa}/fork-server/dashboard/dashboard.html ${fa}/fork-server/dashboard/embed.cmake
        COMMENT "hooks: embedding the fork dashboard.html" VERBATIM)
    hooks_fork_tool(server llama-server-impl ${CMAKE_SOURCE_DIR}/tools/server/server.cpp llama_server
        SOURCES ${fs_srcs} ${fm_srcs} ${dash_hpp}
        INCLUDES ${fa}/fork-server ${fa}/fork-mtmd ${CMAKE_BINARY_DIR}/hooks-fork-server ${CMAKE_SOURCE_DIR}
        LIBS llama-ui vendor::hash)
    if (TARGET hooks-fork-server)
        add_dependencies(hooks-fork-server llama-ui-assets)
        # the miniaudio / stb implementations stay in the upstream mtmd: the fork mtmd uses them
        target_include_directories(hooks-fork-server PRIVATE $<TARGET_PROPERTY:mtmd,INCLUDE_DIRECTORIES>)
    endif()
endif()

hooks_fork_sources(fb_srcs ${fa}/fork-llama-bench)
hooks_fork_tool(llama-bench llama-bench-impl ${CMAKE_SOURCE_DIR}/tools/llama-bench/llama-bench.cpp llama_bench
    SOURCES ${fb_srcs})
hooks_fork_sources(fq_srcs ${fa}/fork-quantize)
hooks_fork_tool(quantize llama-quantize-impl ${CMAKE_SOURCE_DIR}/tools/quantize/quantize.cpp llama_quantize
    SOURCES ${fq_srcs} INCLUDES ${fa}/fork-quantize)

# the example of the fork, a program of its own
if (TARGET hooks-fork-common AND LLAMA_BUILD_EXAMPLES)
    add_executable(llama-speculative-prefill ${fa}/fork-speculative-prefill/speculative-prefill.cpp)
    target_include_directories(llama-speculative-prefill PRIVATE ${fork_common_inc})
    target_link_libraries(llama-speculative-prefill PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-speculative-prefill PRIVATE cxx_std_17)
endif()
