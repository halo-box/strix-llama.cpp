include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
    server-stream.cpp
    server-stream.h
    server-tools.cpp
    server-tools.h
    server-mcp.cpp
    server-mcp.h
    server-schema.cpp
    server-schema.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})

# stats dashboard page (GET /dashboard), embedded as a byte array

set(DASHBOARD_HTML     ${CMAKE_CURRENT_SOURCE_DIR}/dashboard/dashboard.html)
set(DASHBOARD_HTML_HPP ${CMAKE_CURRENT_BINARY_DIR}/dashboard.html.hpp)

add_custom_command(
    OUTPUT  ${DASHBOARD_HTML_HPP}
    COMMAND ${CMAKE_COMMAND}
        "-DINPUT=${DASHBOARD_HTML}"
        "-DOUTPUT=${DASHBOARD_HTML_HPP}"
        "-DNAME=dashboard_html"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/dashboard/embed.cmake"
    DEPENDS ${DASHBOARD_HTML} ${CMAKE_CURRENT_SOURCE_DIR}/dashboard/embed.cmake
    COMMENT "Embedding dashboard.html"
    VERBATIM
)

# llama-server-impl: server logic, reusable by app

set(TARGET llama-server-impl)

add_library(${TARGET}
    ${DASHBOARD_HTML_HPP}
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ../mtmd ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC server-context llama-ui cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

add_dependencies(${TARGET} llama-ui-assets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-server executable

set(TARGET llama-server)

add_executable(${TARGET} main.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE llama-server-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
