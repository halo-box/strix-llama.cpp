# The fork adds LLVM 22 to the OpenMP fetch of Windows builds (GGML_OPENMP_FETCH in ggml/src/CMakeLists.txt of the
# fork 1fe05fa45, after upstream lines 234-249). The block sits inside the upstream fetch logic and the hooks are
# off with MSVC, so it is kept here as the fork wrote it; it is not included by the build.
    if (GGML_OPENMP_COMPILER_VERSION_MAJOR STREQUAL "22")
        set(GGML_OPENMP_LLVM_VERSION "22.1.3")
        set(GGML_OPENMP_INSTALLER_SHA256_X64   "72ca54c3b7940615776d8994c0b918a1d8c4981bdcf13b439eaa86a1bd477001")
        set(GGML_OPENMP_INSTALLER_SHA256_ARM64 "c18d02a43ba31595ca99590faf3ac74c89577f3f49e903537c5bdec5100ed4d2")
    else()
        set(GGML_OPENMP_LLVM_VERSION "20.1.8")
        set(GGML_OPENMP_INSTALLER_SHA256_X64   "3197846a2b19063687dd56e93e34cd941e3548d907f23a6131571321bdf9fe7b")
        set(GGML_OPENMP_INSTALLER_SHA256_ARM64 "7c4ac97eb2ae6b960ca5f9caf3ff6124c8d2a18cc07a7840a4d2ea15537bad8e")
    endif()
    string(REGEX MATCH "^[0-9]+" GGML_OPENMP_LLVM_VERSION_MAJOR "${GGML_OPENMP_LLVM_VERSION}")
        set(GGML_OPENMP_INSTALLER_SHA256 "${GGML_OPENMP_INSTALLER_SHA256_X64}")
        set(GGML_OPENMP_INSTALLER_SHA256 "${GGML_OPENMP_INSTALLER_SHA256_ARM64}")
