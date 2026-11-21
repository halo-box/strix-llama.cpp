# Hooks above ggml: libllama, common, server. Included by hooks/hooks.cmake.
# Hook files are globbed: add a file in the right directory and re-run cmake.

# libllama: models (hooks/llama/models) and graph passes (hooks/llama/passes)
file(GLOB HOOKS_LLAMA_UNITS CONFIGURE_DEPENDS ${HOOKS_DIR}/llama/models/*.cpp ${HOOKS_DIR}/llama/passes/*.cpp)
target_include_directories(llama PRIVATE ${HOOKS_DIR}/llama/include)
hooks_link(llama UNITS ${HOOKS_DIR}/llama/src/llama-hooks.cpp ${HOOKS_LLAMA_UNITS})
# the callbacks of llama-hooks.h: internal C++ functions (Itanium names, libstdc++) and the public C API
hooks_wrap(llama SYMBOLS
    _Z18llama_model_createR18llama_model_loaderRK18llama_model_params
    _ZNK11llama_model11build_graphERK16llm_graph_params
    _ZNK11llama_model13create_memoryERK19llama_memory_paramsRK13llama_cparams
    _ZN18llama_model_loader13create_tensorERK13llama_hparamsPKSt6vectorISt4pairIP19ggml_backend_deviceP24ggml_backend_buffer_typeESaIS9_EESD_SD_SD_RK11LLM_TN_IMPLRKSt16initializer_listIlEi)
hooks_wrap(llama API SYMBOLS
    llama_model_load_from_file llama_model_load_from_splits llama_model_load_from_file_ptr llama_model_init_from_user llama_model_quantize llama_model_free
    llama_init_from_model llama_free
    llama_process llama_decode llama_encode
    llama_memory_clear llama_memory_seq_rm llama_memory_seq_cp llama_memory_seq_keep llama_memory_seq_add llama_memory_seq_div)

# common: CLI arguments (hooks/common/args)
if (TARGET llama-common)
    file(GLOB HOOKS_COMMON_UNITS CONFIGURE_DEPENDS ${HOOKS_DIR}/common/args/*.cpp)
    target_include_directories(llama-common PUBLIC ${HOOKS_DIR}/common/include)
    hooks_link(llama-common UNITS ${HOOKS_DIR}/common/src/common-hooks.cpp ${HOOKS_COMMON_UNITS})
    hooks_wrap(llama-common SYMBOLS
        _Z19common_sampler_initPK11llama_modelR22common_params_sampling
        _Z24common_speculative_draftP18common_speculative
        _Z25common_speculative_acceptP18common_speculativeit)
endif()

# server: callbacks and routes (hooks/server/handlers). The registry and the slot callbacks are in server-context;
# the handlers and the HTTP side are in llama-server-impl, which has server-http.cpp
if (TARGET server-context)
    target_include_directories(server-context PUBLIC ${HOOKS_DIR}/server/include)
    hooks_link(server-context UNITS ${HOOKS_DIR}/server/src/server-hooks.cpp)
    hooks_wrap(server-context SYMBOLS
        _ZN13server_schema22eval_llama_cmpl_schemaEPK11llama_vocabRK13common_paramsRKSt6vectorI16llama_logit_biasSaIS7_EERK11common_json
        _ZN14server_context10load_modelER13common_params
        _ZN14server_contextD1Ev)
endif()
if (TARGET llama-server-impl)
    file(GLOB HOOKS_SERVER_UNITS CONFIGURE_DEPENDS ${HOOKS_DIR}/server/handlers/*.cpp)
    hooks_link(llama-server-impl UNITS ${HOOKS_DIR}/server/src/server-hooks-http.cpp ${HOOKS_SERVER_UNITS})
    hooks_wrap(llama-server-impl SYMBOLS _ZN19server_http_context5startEv)
endif()

# tests, not built by default: cmake --build build --target test-llama-hooks
if (TARGET llama-common)
    add_executable(test-llama-hooks EXCLUDE_FROM_ALL ${HOOKS_DIR}/tests/test-llama-hooks.cpp)
    target_link_libraries(test-llama-hooks PRIVATE llama-common llama)
    target_include_directories(test-llama-hooks PRIVATE ${HOOKS_DIR}/llama/include ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_compile_features(test-llama-hooks PRIVATE cxx_std_17)
    # the test declares ggml kernels: they need the registrar of the executable
    hooks_link(test-llama-hooks UNITS)
endif()

# upstream tests that loop over every ggml type: built without GGML_HOOKS, they see the upstream types only.
# The fork types are tested by the fork versions in hooks/tests.
foreach (t test-quantize-fns test-quantize-perf)
    if (TARGET ${t})
        target_compile_options(${t} PRIVATE -UGGML_HOOKS)
    endif()
endforeach()

if (TARGET test-quantize-fns)
    add_executable(test-quantize-fns-hooks EXCLUDE_FROM_ALL ${HOOKS_DIR}/tests/test-quantize-fns.cpp)
    target_link_libraries(test-quantize-fns-hooks PRIVATE ggml)
    target_include_directories(test-quantize-fns-hooks PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
    target_compile_definitions(test-quantize-fns-hooks PRIVATE GGML_USE_TRELLIS)
    target_compile_features(test-quantize-fns-hooks PRIVATE cxx_std_17)
endif()
