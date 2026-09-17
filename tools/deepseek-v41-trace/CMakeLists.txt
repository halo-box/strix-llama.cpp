set(TARGET llama-deepseek-v41-trace)
set(DSV41_INSTALL_COMPONENT DeepSeekV41Trace)

if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Skipping DeepSeek V4.1 trace tools because BUILD_SHARED_LIBS is disabled")
    return()
endif()

find_package(Git REQUIRED)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE DSV41_BUILD_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE DSV41_BUILD_REVISION_RESULT)
string(LENGTH "${DSV41_BUILD_REVISION}" DSV41_BUILD_REVISION_LENGTH)
if(NOT DSV41_BUILD_REVISION_RESULT EQUAL 0 OR
        NOT DSV41_BUILD_REVISION_LENGTH EQUAL 40 OR
        NOT DSV41_BUILD_REVISION MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "DeepSeek V4.1 trace exporter requires the exact Git revision")
endif()
set(DSV41_RECEIPT_TARGETS llama-common llama ggml ggml-base)
target_sources(llama-common PRIVATE runtime-anchor.cpp)
set(DSV41_BACKEND_TARGETS ${GGML_AVAILABLE_BACKENDS})
list(SORT DSV41_BACKEND_TARGETS)
foreach(backend IN LISTS DSV41_BACKEND_TARGETS)
    if(TARGET ${backend})
        list(APPEND DSV41_RECEIPT_TARGETS ${backend})
    endif()
endforeach()
list(REMOVE_DUPLICATES DSV41_RECEIPT_TARGETS)

set(CONTAINMENT_HELPER_TARGET llama-deepseek-v41-containment-helper)
add_executable(${CONTAINMENT_HELPER_TARGET} linux-containment-helper.cpp)
target_compile_features(${CONTAINMENT_HELPER_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(
    ${CONTAINMENT_HELPER_TARGET}
    PRIVATE DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}")
set(DSV41_CONTAINMENT_HELPER_RECEIPT
    "${CMAKE_CURRENT_BINARY_DIR}/dsv41-containment-helper-receipt.json")
add_custom_command(
    OUTPUT "${DSV41_CONTAINMENT_HELPER_RECEIPT}"
    COMMAND
        ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/generate-containment-helper-receipt.py"
        --output "${DSV41_CONTAINMENT_HELPER_RECEIPT}"
        --helper "$<TARGET_FILE:${CONTAINMENT_HELPER_TARGET}>"
        --revision "${DSV41_BUILD_REVISION}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/generate-containment-helper-receipt.py"
        ${CONTAINMENT_HELPER_TARGET}
    VERBATIM)
add_custom_target(
    dsv41-containment-helper-receipt
    DEPENDS "${DSV41_CONTAINMENT_HELPER_RECEIPT}")

if(WIN32)
    set(DSV41_RUNTIME_PROFILE co-located)
else()
    set(DSV41_RUNTIME_PROFILE sibling-lib)
    if(APPLE)
        set(DSV41_LIBRARY_RPATH "@loader_path")
        set(DSV41_EXECUTABLE_RPATH "@loader_path/../lib")
    else()
        set(DSV41_LIBRARY_RPATH "\$ORIGIN")
        set(DSV41_EXECUTABLE_RPATH "\$ORIGIN/../lib")
    endif()
endif()

set(DSV41_RECEIPT_ARGUMENTS)
set(DSV41_RECEIPT_FILES)
foreach(component IN LISTS DSV41_RECEIPT_TARGETS)
    if(NOT TARGET ${component})
        message(FATAL_ERROR "DeepSeek V4.1 receipt target is missing: ${component}")
    endif()
    get_target_property(component_type ${component} TYPE)
    if(NOT component_type STREQUAL "SHARED_LIBRARY" AND NOT component_type STREQUAL "MODULE_LIBRARY")
        message(FATAL_ERROR "DeepSeek V4.1 receipt target is not shared: ${component}")
    endif()
    if(component STREQUAL "llama-common" OR component STREQUAL "ggml-base")
        set(component_revision "${DSV41_BUILD_REVISION}")
    else()
        set(component_revision "-")
    endif()
    list(APPEND DSV41_RECEIPT_ARGUMENTS
        --entry "${component}" "$<TARGET_FILE:${component}>" "${component_revision}")
    list(APPEND DSV41_RECEIPT_FILES "$<TARGET_FILE:${component}>")
    if(NOT WIN32)
        set_target_properties(
            ${component}
            PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "${DSV41_LIBRARY_RPATH}")
        if(APPLE)
            set_target_properties(
                ${component}
                PROPERTIES
                    BUILD_WITH_INSTALL_NAME_DIR TRUE
                    INSTALL_NAME_DIR "@rpath")
        endif()
    endif()
endforeach()

set(DSV41_RECEIPT_HEADER "${CMAKE_CURRENT_BINARY_DIR}/dsv41-runtime-receipt.h")
set(DSV41_RECEIPT_JSON "${CMAKE_CURRENT_BINARY_DIR}/dsv41-runtime-receipt.json")
add_custom_command(
    OUTPUT "${DSV41_RECEIPT_HEADER}" "${DSV41_RECEIPT_JSON}"
    COMMAND
        ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/generate-runtime-receipt.py"
        --output "${DSV41_RECEIPT_HEADER}"
        --json-output "${DSV41_RECEIPT_JSON}"
        --revision "${DSV41_BUILD_REVISION}"
        --profile "${DSV41_RUNTIME_PROFILE}"
        ${DSV41_RECEIPT_ARGUMENTS}
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/generate-runtime-receipt.py"
        ${DSV41_RECEIPT_FILES}
    VERBATIM)
add_custom_target(
    dsv41-runtime-receipt
    DEPENDS "${DSV41_RECEIPT_HEADER}" "${DSV41_RECEIPT_JSON}")
set_source_files_properties(
    llama-trace.cpp
    PROPERTIES OBJECT_DEPENDS "${DSV41_RECEIPT_HEADER}")

add_executable(${TARGET} llama-trace.cpp)
add_dependencies(${TARGET} dsv41-runtime-receipt dsv41-containment-helper-receipt)
target_link_libraries(${TARGET} PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}")
if(NOT WIN32)
    set_target_properties(
        ${TARGET}
        PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${DSV41_EXECUTABLE_RPATH}")
endif()

set(PROMPT_TARGET llama-deepseek-v41-prompt-builder)
add_executable(${PROMPT_TARGET} prompt-builder.cpp)
add_dependencies(${PROMPT_TARGET} dsv41-runtime-receipt dsv41-containment-helper-receipt)
target_link_libraries(${PROMPT_TARGET} PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
target_include_directories(${PROMPT_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(${PROMPT_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${PROMPT_TARGET} PRIVATE DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}")
if(NOT WIN32)
    set_target_properties(
        ${PROMPT_TARGET}
        PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${DSV41_EXECUTABLE_RPATH}")
endif()

if(LLAMA_TOOLS_INSTALL)
    install(
        TARGETS ${CONTAINMENT_HELPER_TARGET}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(
        FILES "${DSV41_CONTAINMENT_HELPER_RECEIPT}"
        DESTINATION "share/deepseek-v41-trace"
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
    install(
        TARGETS ${TARGET}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(
        TARGETS ${DSV41_RECEIPT_TARGETS}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        LIBRARY DESTINATION lib
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        NAMELINK_COMPONENT ${DSV41_INSTALL_COMPONENT})
    install(
        TARGETS ${PROMPT_TARGET}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT ${DSV41_INSTALL_COMPONENT}
        PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()

if(LLAMA_BUILD_TESTS)
    set(MANIFEST_TEST_TARGET test-deepseek41-trace-manifest)
    add_executable(${MANIFEST_TEST_TARGET} llama-trace.cpp)
    add_dependencies(
        ${MANIFEST_TEST_TARGET}
        dsv41-runtime-receipt
        dsv41-containment-helper-receipt)
    target_link_libraries(
        ${MANIFEST_TEST_TARGET}
        PRIVATE llama-common llama vendor::hash ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
    target_include_directories(
        ${MANIFEST_TEST_TARGET}
        PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_features(${MANIFEST_TEST_TARGET} PRIVATE cxx_std_17)
    target_compile_definitions(
        ${MANIFEST_TEST_TARGET}
        PRIVATE
            DSV41_BUILD_REVISION="${DSV41_BUILD_REVISION}"
            DSV41_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
            DSV41_MANIFEST_TEST_HARNESS=1)
    if(NOT WIN32)
        set_target_properties(
            ${MANIFEST_TEST_TARGET}
            PROPERTIES
                BUILD_WITH_INSTALL_RPATH TRUE
                INSTALL_RPATH "${DSV41_EXECUTABLE_RPATH}")
    endif()

    set(INJECTED_TEST_TARGET test-deepseek41-trace-injected)
    add_library(${INJECTED_TEST_TARGET} SHARED test-injected-library.cpp)
    set_target_properties(
        ${INJECTED_TEST_TARGET}
        PROPERTIES
            PREFIX ""
            OUTPUT_NAME "libggml-injected"
            SUFFIX ".module")
    set(HOST_TEST_TARGET test-deepseek41-trace-host)
    add_executable(${HOST_TEST_TARGET} test-host-attestation.cpp)
    target_link_libraries(${HOST_TEST_TARGET} PRIVATE ggml)
    target_compile_features(${HOST_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${HOST_TEST_TARGET} COMMAND $<TARGET_FILE:${HOST_TEST_TARGET}>)
    set_property(TEST ${HOST_TEST_TARGET} PROPERTY LABELS main)

    if(LLAMA_TOOLS_INSTALL)
        add_test(
            NAME test-deepseek41-trace-install
            COMMAND
                ${CMAKE_COMMAND}
                -DDSV41_BUILD_DIR=${CMAKE_BINARY_DIR}
                -DDSV41_INSTALL_ROOT=${CMAKE_CURRENT_BINARY_DIR}/install-smoke
                -DDSV41_COMPONENT=${DSV41_INSTALL_COMPONENT}
                -DDSV41_EXECUTABLE=${TARGET}${CMAKE_EXECUTABLE_SUFFIX}
                -DDSV41_SOURCE_EXECUTABLE=$<TARGET_FILE:${TARGET}>
                -DDSV41_REVISION=${DSV41_BUILD_REVISION}
                -DDSV41_CONFIG=$<CONFIG>
                -DDSV41_PYTHON=${Python3_EXECUTABLE}
                -DDSV41_RECEIPT=${DSV41_RECEIPT_JSON}
                -DDSV41_CONTAINMENT_HELPER=$<TARGET_FILE:${CONTAINMENT_HELPER_TARGET}>
                -DDSV41_CONTAINMENT_HELPER_RECEIPT=${DSV41_CONTAINMENT_HELPER_RECEIPT}
                -DDSV41_VERIFY_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/verify-runtime-install.py
                -P ${CMAKE_CURRENT_SOURCE_DIR}/test-install.cmake)
        set_property(TEST test-deepseek41-trace-install PROPERTY LABELS main)
    endif()
endif()
