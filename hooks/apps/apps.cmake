# The fork tools (hooks/apps): common, mtmd, server, llama-bench and quantize as namespaced copies (ggml_fork), and the
# fork example speculative-prefill. The tools switch (on, GGML_HOOKS=-tools: upstream) runs the fork tools: their entry
# functions (llama_server, llama_bench, llama_quantize) are renamed *__upstream in their upstream files and
# tools-dispatch-*.cpp calls one side. The fork tools call the fork libllama directly.
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
    hooks_fork_sources(fc_srcs ${fa}/common)
    if (NOT GGML_EXL3)
        list(FILTER fc_srcs EXCLUDE REGEX "/exl3-dir\\.cpp$")
    endif()
    add_library(hooks-common OBJECT ${fc_srcs})
    target_include_directories(hooks-common PRIVATE ${fa}/common ${HOOKS_DIR}/llama/lib-include
                                                         ${HOOKS_DIR}/llama/lib ${HOOKS_DIR}/llama/include)
    target_link_libraries(hooks-common PRIVATE ggml vendor::nlohmann vendor::sheredom cpp-httplib)
    target_compile_definitions(hooks-common PRIVATE $<TARGET_PROPERTY:llama-common,COMPILE_DEFINITIONS>)
    if (GGML_EXL3)
        target_link_libraries(hooks-common PRIVATE vendor::hash)
        target_compile_definitions(hooks-common PRIVATE LLAMA_USE_EXL3)
    endif()
    target_compile_features(hooks-common PRIVATE cxx_std_17)
    set_target_properties(hooks-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(llama-common PRIVATE $<TARGET_OBJECTS:hooks-common>)
    target_link_libraries(llama-common PRIVATE vendor::hash) # the fork common hashes files
    set(fork_common_inc ${fa}/common ${HOOKS_DIR}/llama/lib-include ${HOOKS_DIR}/llama/lib ${HOOKS_DIR}/llama/include)
endif()

# a fork tool: its sources in an object library linked into the upstream impl library, and its entry dispatched
# mtmd-helper.cpp holds an implementation of stb_image and miniaudio (C symbols), as the upstream file does: local to the
# file, for the static builds where both are in one program
set_source_files_properties(${fa}/mtmd/mtmd-helper.cpp PROPERTIES
    COMPILE_DEFINITIONS "STB_IMAGE_STATIC;MA_API=static;ma_atomic_global_lock=ma_atomic_global_lock_hooks"
    COMPILE_OPTIONS "-Wno-unused-function;-Wno-unused-variable")

function(hooks_fork_tool name impl upstream_src entry)
    if (NOT TARGET ${impl} OR NOT TARGET hooks-common)
        return()
    endif()
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LIBS" ${ARGN})
    add_library(hooks-${name} OBJECT ${ARG_SOURCES})
    target_include_directories(hooks-${name} PRIVATE ${ARG_INCLUDES} ${fork_common_inc})
    target_link_libraries(hooks-${name} PRIVATE ggml vendor::nlohmann vendor::sheredom cpp-httplib ${ARG_LIBS})
    target_compile_definitions(hooks-${name} PRIVATE $<TARGET_PROPERTY:llama-common,COMPILE_DEFINITIONS>
                                                          $<TARGET_PROPERTY:${impl},COMPILE_DEFINITIONS>)
    target_compile_features(hooks-${name} PRIVATE cxx_std_17)
    set_target_properties(hooks-${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_sources(${impl} PRIVATE $<TARGET_OBJECTS:hooks-${name}> ${fa}/tools-dispatch-${name}.cpp)
    hooks_rename(${impl} SOURCE ${upstream_src} SYMBOLS ${entry})
endfunction()

hooks_fork_sources(fs_srcs ${fa}/server)
hooks_fork_sources(fm_srcs ${fa}/mtmd)
if (TARGET llama-server-impl)
    # the stats dashboard of the fork server, embedded as in tools/server/CMakeLists.txt of the fork
    set(dash_hpp ${CMAKE_BINARY_DIR}/hooks-server/dashboard.html.hpp)
    add_custom_command(OUTPUT ${dash_hpp}
        COMMAND ${CMAKE_COMMAND} "-DINPUT=${fa}/server/dashboard/dashboard.html" "-DOUTPUT=${dash_hpp}"
                "-DNAME=dashboard_html" -P "${fa}/server/dashboard/embed.cmake"
        DEPENDS ${fa}/server/dashboard/dashboard.html ${fa}/server/dashboard/embed.cmake
        COMMENT "hooks: embedding the fork dashboard.html" VERBATIM)
    hooks_fork_tool(server llama-server-impl ${CMAKE_SOURCE_DIR}/tools/server/server.cpp llama_server
        SOURCES ${fs_srcs} ${fm_srcs} ${dash_hpp}
        INCLUDES ${fa}/server ${fa}/mtmd ${CMAKE_BINARY_DIR}/hooks-server ${CMAKE_SOURCE_DIR}
        LIBS llama-ui vendor::hash)
    if (TARGET hooks-server)
        add_dependencies(hooks-server llama-ui-assets)
        # the miniaudio / stb implementations stay in the upstream mtmd: the fork mtmd uses them
        target_include_directories(hooks-server PRIVATE $<TARGET_PROPERTY:mtmd,INCLUDE_DIRECTORIES>)
    endif()
endif()

hooks_fork_sources(fb_srcs ${fa}/llama-bench)
hooks_fork_tool(llama-bench llama-bench-impl ${CMAKE_SOURCE_DIR}/tools/llama-bench/llama-bench.cpp llama_bench
    SOURCES ${fb_srcs})
hooks_fork_sources(fq_srcs ${fa}/quantize)
hooks_fork_tool(quantize llama-quantize-impl ${CMAKE_SOURCE_DIR}/tools/quantize/quantize.cpp llama_quantize
    SOURCES ${fq_srcs} INCLUDES ${fa}/quantize)

# the example of the fork, a program of its own
if (TARGET hooks-common AND LLAMA_BUILD_EXAMPLES)
    add_executable(llama-speculative-prefill ${fa}/speculative-prefill/speculative-prefill.cpp)
    target_include_directories(llama-speculative-prefill PRIVATE ${fork_common_inc})
    target_link_libraries(llama-speculative-prefill PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-speculative-prefill PRIVATE cxx_std_17)
endif()

# the tests of the fork that are new or changed: test-<name>, or test-<name>-fork when upstream has a test of the
# same name; built with the tests (LLAMA_BUILD_TESTS), else by the target hooks-tests
# (cmake --build build --target hooks-tests)
if (TARGET hooks-common)
    file(GLOB fork_tests CONFIGURE_DEPENDS ${fa}/tests/test-*.cpp)
    if (NOT GGML_EXL3)
        list(FILTER fork_tests EXCLUDE REGEX "/test-exl3-dir\\.cpp$")
    endif()
    # not built by the fork either (its tests/CMakeLists.txt has no rule for it)
    list(FILTER fork_tests EXCLUDE REGEX "/test-save-load-state\\.cpp$")
    if (GGML_BACKEND_DL OR NOT GGML_CPU)
        # they call the CPU backend directly, as in tests/CMakeLists.txt of the fork
        list(FILTER fork_tests EXCLUDE REGEX "/test-(tq2t|tqk|tqp|exl3-dir)\\.cpp$")
    endif()
    # several files: below
    list(FILTER fork_tests EXCLUDE REGEX "/test-llama-context\\.cpp$")
    add_custom_target(hooks-tests)
    foreach (src ${fork_tests})
        get_filename_component(name ${src} NAME_WE)
        set(tgt ${name})
        if (EXISTS ${CMAKE_SOURCE_DIR}/tests/${name}.cpp)
            set(tgt ${name}-fork) # upstream has a test of this name
        endif()
        add_executable(${tgt} ${src})
        if (NOT LLAMA_BUILD_TESTS)
            set_target_properties(${tgt} PROPERTIES EXCLUDE_FROM_ALL ON) # as the fork: built with the tests
        endif()
        target_include_directories(${tgt} PRIVATE ${fa}/tests ${fork_common_inc} ${fa}/server ${fa}/mtmd
                                                        ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR})
        target_link_libraries(${tgt} PRIVATE llama-common llama ggml vendor::hash)
        if (name MATCHES "server|chat" AND TARGET llama-server-impl)
            target_link_libraries(${tgt} PRIVATE llama-server-impl) # the fork server objects
        endif()
        target_compile_features(${tgt} PRIVATE cxx_std_17)
        add_dependencies(hooks-tests ${tgt})
        # the reference data of the trellis tests, as tests/CMakeLists.txt of the fork
        if (name STREQUAL "test-tq2t")
            target_compile_definitions(${tgt} PRIVATE TQ2T_REF_DEFAULT="${CMAKE_SOURCE_DIR}/tests/data/tq2t_ref.bin")
        elseif (name STREQUAL "test-tqk")
            target_compile_definitions(${tgt} PRIVATE TQK_REF_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        elseif (name STREQUAL "test-tqp")
            target_compile_definitions(${tgt} PRIVATE TQP_REF_DIR="${CMAKE_SOURCE_DIR}/tests/data")
        endif()
    endforeach()
endif()

# test-llama-context of the fork, built from its files (hooks/tools/import-apps.py copies them when the fork changes one)
if (TARGET hooks-tests AND EXISTS ${fa}/tests/test-llama-context.cpp)
    file(GLOB lc_srcs CONFIGURE_DEPENDS ${fa}/tests/llama-context/*.cpp)
    add_executable(test-llama-context-fork ${fa}/tests/test-llama-context.cpp ${lc_srcs})
    if (NOT LLAMA_BUILD_TESTS)
        set_target_properties(test-llama-context-fork PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
    target_include_directories(test-llama-context-fork PRIVATE ${fa}/tests ${fork_common_inc} ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR})
    target_link_libraries(test-llama-context-fork PRIVATE llama-common llama ggml)
    target_compile_features(test-llama-context-fork PRIVATE cxx_std_17)
    add_dependencies(hooks-tests test-llama-context-fork)
endif()

# ctest: the tests of the fork as tests/CMakeLists.txt of the fork registers them (names, arguments, labels, conditions),
# run on the fork code: GGML_HOOKS holds the switches of the backends of this build. The fork versions of upstream tests
# are test-<name>-fork, next to the upstream test of the same name.
if (LLAMA_BUILD_TESTS AND TARGET hooks-tests)
    set(HOOKS_FORK_TEST_SWITCHES "+sched,+llama")
    if (GGML_CUDA OR GGML_HIP)
        string(APPEND HOOKS_FORK_TEST_SWITCHES ",+cuda_graph")
    endif()
    if (GGML_VULKAN)
        string(APPEND HOOKS_FORK_TEST_SWITCHES ",+vk_graph")
    endif()

    # hooks_fork_test(<ctest name> <fork test> [LABEL <labels>] [WORKING_DIRECTORY <dir>] [ARGS <args>])
    function(hooks_fork_test name test)
        cmake_parse_arguments(T "" "WORKING_DIRECTORY" "LABEL;ARGS" ${ARGN})
        if (TARGET ${test}-fork)
            set(tgt ${test}-fork) # the fork version of an upstream test
        elseif (TARGET ${test} AND EXISTS ${fa}/tests/${test}.cpp AND NOT EXISTS ${CMAKE_SOURCE_DIR}/tests/${test}.cpp)
            set(tgt ${test}) # new in the fork
        else()
            return()
        endif()
        if (NOT T_LABEL)
            set(T_LABEL main) # the default label of llama_test
        endif()
        if (NOT T_WORKING_DIRECTORY)
            set(T_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests) # "." of llama_test, the tests directory of the build
        endif()
        add_test(NAME ${name} COMMAND $<TARGET_FILE:${tgt}> ${T_ARGS} WORKING_DIRECTORY ${T_WORKING_DIRECTORY})
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
    hooks_fork_test(test-backend-sampler-fork  test-backend-sampler  LABEL model behavior backend)
    if (TARGET test-llama-context-fork)
        # as test-llama-context of tests/CMakeLists.txt: the generated models, abort on a graph reallocation
        add_test(NAME test-llama-context-fork COMMAND $<TARGET_FILE:test-llama-context-fork> --models ${CMAKE_BINARY_DIR}/tests/test-models/
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
        # on llvmpipe the same test takes 863 s in the fork on a slow CI machine, its limit there is 900 s
        set_tests_properties(test-llama-context-fork PROPERTIES LABELS "main;behavior;backend" FIXTURES_REQUIRED generate-models TIMEOUT 1800
                             ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1;GGML_HOOKS=${HOOKS_FORK_TEST_SWITCHES}")
    endif()
endif()
