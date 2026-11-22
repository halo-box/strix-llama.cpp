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

# the tests of the fork that are new or changed: test-<name>-fork, not built by default
# (cmake --build build --target hooks-fork-tests)
if (TARGET hooks-fork-common)
    file(GLOB fork_tests CONFIGURE_DEPENDS ${fa}/fork-tests/test-*.cpp)
    if (NOT GGML_EXL3)
        list(FILTER fork_tests EXCLUDE REGEX "/test-exl3-dir\\.cpp$")
    endif()
    # not built by the fork either (its tests/CMakeLists.txt has no rule for it)
    list(FILTER fork_tests EXCLUDE REGEX "/test-save-load-state\\.cpp$")
    add_custom_target(hooks-fork-tests)
    foreach (src ${fork_tests})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name}-fork ${src})
        if (NOT LLAMA_BUILD_TESTS)
            set_target_properties(${name}-fork PROPERTIES EXCLUDE_FROM_ALL ON) # as the fork: built with the tests
        endif()
        target_include_directories(${name}-fork PRIVATE ${fa}/fork-tests ${fork_common_inc} ${fa}/fork-server ${fa}/fork-mtmd
                                                        ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR})
        target_link_libraries(${name}-fork PRIVATE llama-common llama ggml vendor::hash)
        if (name MATCHES "server|chat" AND TARGET llama-server-impl)
            target_link_libraries(${name}-fork PRIVATE llama-server-impl) # the fork server objects
        endif()
        target_compile_features(${name}-fork PRIVATE cxx_std_17)
        add_dependencies(hooks-fork-tests ${name}-fork)
        # the reference data of the trellis tests, as tests/CMakeLists.txt of the fork
        if (name STREQUAL "test-tq2t")
            target_compile_definitions(${name}-fork PRIVATE TQ2T_REF_DEFAULT="${CMAKE_SOURCE_DIR}/tests/data/tq2t_ref.bin")
        elseif (name STREQUAL "test-tqk")
            target_compile_definitions(${name}-fork PRIVATE TQK_REF_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        elseif (name STREQUAL "test-tqp")
            target_compile_definitions(${name}-fork PRIVATE TQP_REF_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        endif()
    endforeach()
endif()

# ctest: the tests of the fork as tests/CMakeLists.txt of the fork registers them (names, arguments, labels, conditions),
# run on the fork code: GGML_HOOKS holds the fork switches of the backends of this build. The fork versions of upstream
# tests get the -fork suffix, next to the upstream test of the same name.
if (LLAMA_BUILD_TESTS AND TARGET hooks-fork-tests)
    set(HOOKS_FORK_TEST_SWITCHES "+fork_sched,+fork_llama")
    if (GGML_CUDA OR GGML_HIP)
        string(APPEND HOOKS_FORK_TEST_SWITCHES ",+fork_cuda_graph")
    endif()
    if (GGML_VULKAN)
        string(APPEND HOOKS_FORK_TEST_SWITCHES ",+fork_vk_graph")
    endif()

    # hooks_fork_test(<ctest name> <fork test> [LABEL <labels>] [WORKING_DIRECTORY <dir>] [ARGS <args>])
    function(hooks_fork_test name test)
        cmake_parse_arguments(T "" "WORKING_DIRECTORY" "LABEL;ARGS" ${ARGN})
        if (NOT TARGET ${test}-fork)
            return()
        endif()
        if (NOT T_LABEL)
            set(T_LABEL main) # the default label of llama_test
        endif()
        if (NOT T_WORKING_DIRECTORY)
            set(T_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests) # "." of llama_test, the tests directory of the build
        endif()
        add_test(NAME ${name} COMMAND $<TARGET_FILE:${test}-fork> ${T_ARGS} WORKING_DIRECTORY ${T_WORKING_DIRECTORY})
        set_tests_properties(${name} PROPERTIES LABELS "${T_LABEL}" ENVIRONMENT "GGML_HOOKS=${HOOKS_FORK_TEST_SWITCHES}")
    endfunction()

    # new in the fork
    if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
        hooks_fork_test(test-qsa-fast          test-qsa-fast)
        hooks_fork_test(test-qsa-recover       test-qsa-recover)
        hooks_fork_test(test-ple-disk          test-ple-disk)
        hooks_fork_test(test-qsa-prefix        test-qsa-prefix)
    endif()
    hooks_fork_test(test-qwen4exp-mtp-cache-only test-llama-archs ARGS --mtp-cache-only -s 1234)
    hooks_fork_test(test-qwen4exp-merged-weights test-llama-archs ARGS --qwen4exp-merged -s 1234)
    hooks_fork_test(test-qwen35-merged-weights   test-llama-archs ARGS --qwen35-merged -s 1234)
    hooks_fork_test(test-sched-view-scratch test-sched-view-scratch)
    hooks_fork_test(test-graph-src-type     test-graph-src-type)
    hooks_fork_test(test-spec-mtp-pcum      test-spec-mtp-pcum)
    hooks_fork_test(test-draft-vocab        test-draft-vocab)
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        hooks_fork_test(test-tq2t test-tq2t)
        hooks_fork_test(test-tqk  test-tqk)
        hooks_fork_test(test-tqp  test-tqp)
        if (GGML_EXL3)
            hooks_fork_test(test-exl3-dir test-exl3-dir)
        endif()
    endif()
    if (NOT LLAMA_USE_SYSTEM_GGML)
        hooks_fork_test(test-backend-sched-ring test-backend-sched-ring)
    endif()
    # test-mtp-draft-vocab (needs -m with an MTP model) and test-vk-exact (needs AMD Vulkan pipelines): built, not
    # registered, as in the fork

    # changed by the fork: the labels and working directories of the upstream tests
    hooks_fork_test(test-alloc-fork            test-alloc            LABEL behavior backend)
    hooks_fork_test(test-arg-parser-fork       test-arg-parser       LABEL behavior)
    hooks_fork_test(test-batch-alloc-fork      test-batch-alloc      LABEL behavior)
    hooks_fork_test(test-chat-fork             test-chat             LABEL behavior WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    hooks_fork_test(test-grammar-parser-fork   test-grammar-parser   LABEL behavior)
    hooks_fork_test(test-reasoning-budget-fork test-reasoning-budget LABEL behavior)
    hooks_fork_test(test-server-component-fork test-server-component LABEL behavior)
endif()
