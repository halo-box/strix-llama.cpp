if(NOT DEFINED DSV41_BUILD_DIR OR NOT DEFINED DSV41_INSTALL_ROOT OR
        NOT DEFINED DSV41_COMPONENT OR NOT DEFINED DSV41_EXECUTABLE OR
        NOT DEFINED DSV41_SOURCE_EXECUTABLE OR
        NOT DEFINED DSV41_REVISION OR NOT DEFINED DSV41_PYTHON OR
        NOT DEFINED DSV41_RECEIPT OR NOT DEFINED DSV41_CONTAINMENT_HELPER OR
        NOT DEFINED DSV41_CONTAINMENT_HELPER_RECEIPT OR NOT DEFINED DSV41_VERIFY_SCRIPT)
    message(FATAL_ERROR "DeepSeek V4.1 install smoke test arguments are incomplete")
endif()

file(REMOVE_RECURSE "${DSV41_INSTALL_ROOT}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${DSV41_BUILD_DIR}"
        --prefix "${DSV41_INSTALL_ROOT}"
        --component "${DSV41_COMPONENT}"
        --config "${DSV41_CONFIG}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "DeepSeek V4.1 component install failed:\n${install_output}${install_error}")
endif()

file(SHA256 "${DSV41_SOURCE_EXECUTABLE}" source_executable_sha256)
file(SHA256 "${DSV41_INSTALL_ROOT}/bin/${DSV41_EXECUTABLE}" installed_executable_sha256)
if(NOT source_executable_sha256 STREQUAL installed_executable_sha256)
    message(FATAL_ERROR "installed DeepSeek V4.1 trace executable differs from linked bytes")
endif()

get_filename_component(containment_helper_name "${DSV41_CONTAINMENT_HELPER}" NAME)
file(SHA256 "${DSV41_CONTAINMENT_HELPER}" source_helper_sha256)
file(SHA256 "${DSV41_INSTALL_ROOT}/bin/${containment_helper_name}" installed_helper_sha256)
if(NOT source_helper_sha256 STREQUAL installed_helper_sha256)
    message(FATAL_ERROR "installed DeepSeek V4.1 containment helper differs from linked bytes")
endif()
file(
    SHA256 "${DSV41_CONTAINMENT_HELPER_RECEIPT}"
    source_containment_helper_receipt_sha256)
file(
    SHA256
    "${DSV41_INSTALL_ROOT}/share/deepseek-v41-trace/dsv41-containment-helper-receipt.json"
    installed_containment_helper_receipt_sha256)
if(NOT source_containment_helper_receipt_sha256 STREQUAL
        installed_containment_helper_receipt_sha256)
    message(FATAL_ERROR "installed DeepSeek V4.1 containment helper receipt differs from build bytes")
endif()

execute_process(
    COMMAND "${DSV41_PYTHON}" "${DSV41_VERIFY_SCRIPT}"
        --receipt "${DSV41_RECEIPT}"
        --install-root "${DSV41_INSTALL_ROOT}"
    RESULT_VARIABLE receipt_result
    OUTPUT_VARIABLE receipt_output
    ERROR_VARIABLE receipt_error)
if(NOT receipt_result EQUAL 0)
    message(FATAL_ERROR
        "installed DeepSeek V4.1 runtime receipt differs from linked bytes:\n${receipt_output}${receipt_error}")
endif()

unset(ENV{DYLD_FALLBACK_LIBRARY_PATH})
unset(ENV{DYLD_FALLBACK_FRAMEWORK_PATH})
unset(ENV{DYLD_FRAMEWORK_PATH})
unset(ENV{DYLD_IMAGE_SUFFIX})
unset(ENV{DYLD_INSERT_LIBRARIES})
unset(ENV{DYLD_LIBRARY_PATH})
unset(ENV{DYLD_ROOT_PATH})
unset(ENV{DYLD_VERSIONED_FRAMEWORK_PATH})
unset(ENV{DYLD_VERSIONED_LIBRARY_PATH})
unset(ENV{GGML_BACKEND_PATH})
unset(ENV{LD_LIBRARY_PATH})
unset(ENV{LD_PRELOAD})
unset(ENV{PATH})

execute_process(
    COMMAND "${DSV41_INSTALL_ROOT}/bin/${DSV41_EXECUTABLE}" --version
    RESULT_VARIABLE smoke_result
    OUTPUT_VARIABLE smoke_output
    ERROR_VARIABLE smoke_error)
if(NOT smoke_result EQUAL 0)
    message(FATAL_ERROR "installed DeepSeek V4.1 trace --version failed:\n${smoke_output}${smoke_error}")
endif()
if(NOT smoke_output MATCHES "commit ${DSV41_REVISION}")
    message(FATAL_ERROR "installed DeepSeek V4.1 trace reported the wrong revision:\n${smoke_output}")
endif()

execute_process(
    COMMAND "${DSV41_INSTALL_ROOT}/bin/${containment_helper_name}" --version
    RESULT_VARIABLE helper_smoke_result
    OUTPUT_VARIABLE helper_smoke_output
    ERROR_VARIABLE helper_smoke_error)
if(NOT helper_smoke_result EQUAL 0)
    message(FATAL_ERROR
        "installed DeepSeek V4.1 containment helper --version failed:\n"
        "${helper_smoke_output}${helper_smoke_error}")
endif()
if(NOT helper_smoke_output MATCHES "${DSV41_REVISION}")
    message(FATAL_ERROR
        "installed DeepSeek V4.1 containment helper reported the wrong revision:\n"
        "${helper_smoke_output}")
endif()
