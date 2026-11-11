cmake_minimum_required(VERSION 3.28)

# Must precede project(): with no build type CMake adds no -O flag and GCC defaults to -O0.
# Only when built on its own; a project that pulls this one in keeps its own build type.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT CMAKE_BUILD_TYPE
        AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
endif()

# 120 = compute capability 12.0 (RTX 50-series). "native" works on CUDA 12.6+.
# A default only: set before project(), which would otherwise cache its own (75
# with CUDA 13), so -DCMAKE_CUDA_ARCHITECTURES=... on the command line still wins.
# A build directory configured before this file existed may hold that 75; reset
# its cache once (CLion: Tools > CMake > Reset Cache and Reload Project).
# A pip install from source defaults to "native", the GPU of the machine building it.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (DEFINED SKBUILD)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to compile for")
    endif()
endif()

# The one place the version is written. version.h, the find_package version file,
# the Python package metadata and neraulgraphics.__version__ all read it from here.
project(neraulgraphics VERSION 0.5.0 LANGUAGES CXX)

# CUDA is optional. Without it the library still builds and renders, on the CPU, with nothing but a
# C++17 compiler. AUTO uses CUDA when a CUDA compiler is found; ON requires it; OFF skips it.
set(NERAULGRAPHICS_CUDA AUTO CACHE STRING "Build the CUDA (NVIDIA GPU) backend: AUTO, ON or OFF")
set_property(CACHE NERAULGRAPHICS_CUDA PROPERTY STRINGS AUTO ON OFF)
set(NERAULGRAPHICS_WITH_CUDA OFF)
if (NERAULGRAPHICS_CUDA STREQUAL "ON")
    enable_language(CUDA)
    set(NERAULGRAPHICS_WITH_CUDA ON)
elseif (NERAULGRAPHICS_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(NERAULGRAPHICS_WITH_CUDA ON)
    endif()
endif()
message(STATUS "neraulgraphics: CUDA backend ${NERAULGRAPHICS_WITH_CUDA}; the CPU backend is always built")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# On by default only when this is the top-level project, so a project that uses
# the library through add_subdirectory or FetchContent never needs SDL2.
option(NERAULGRAPHICS_BUILD_VIEWER "Build the interactive SDL2 viewer" ${PROJECT_IS_TOP_LEVEL})
option(NERAULGRAPHICS_BUILD_TESTS "Build the C++ tests (run with ctest)" ${PROJECT_IS_TOP_LEVEL})
option(NERAULGRAPHICS_INSTALL "Generate install rules and the find_package config" ${PROJECT_IS_TOP_LEVEL})
option(NERAULGRAPHICS_BUILD_PYTHON "Build the Python module (pip install . sets this)" OFF)

# Shared links libcudart.so, which the machine running the program must have.
# Static bakes the runtime in and needs only the NVIDIA driver, which is what lets
# a Python wheel run on a machine without the CUDA toolkit.
set(NERAULGRAPHICS_CUDA_RUNTIME Shared CACHE STRING "How to link the CUDA runtime: Shared or Static")
set_property(CACHE NERAULGRAPHICS_CUDA_RUNTIME PROPERTY STRINGS Shared Static)

find_package(Threads REQUIRED)
if (NERAULGRAPHICS_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------- library --

# The renderer on its own: no window, no SDL. Consumers link neraulgraphics::core.
add_library(neraulgraphics_core STATIC
    src/renderer.cpp
    src/batch.cpp
    src/validate.cpp
    src/environment.cpp
    src/cpu_backend.cpp
    src/instances.cpp
    src/bvh.cpp
    src/mesh.cpp
    src/mesh_build.cpp)
if (NERAULGRAPHICS_WITH_CUDA)
    target_sources(neraulgraphics_core PRIVATE src/cuda_backend.cu)
else()
    target_sources(neraulgraphics_core PRIVATE src/cuda_backend_stub.cpp)
endif()
add_library(neraulgraphics::core ALIAS neraulgraphics_core)

configure_file(cmake/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/neraulgraphics/version.h
    @ONLY)

# The public headers. BASE_DIRS makes these the include path, both here and once
# installed, so every consumer writes #include <neraulgraphics/renderer.h>.
# Everything in src/ (the BVH, the rendering core, the backends) stays private to the library.
target_sources(neraulgraphics_core PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include ${CMAKE_CURRENT_BINARY_DIR}/include
    FILES
        include/neraulgraphics/batch.h
        include/neraulgraphics/mesh.h
        include/neraulgraphics/renderer.h
        include/neraulgraphics/scene.h
        ${CMAKE_CURRENT_BINARY_DIR}/include/neraulgraphics/version.h)

target_include_directories(neraulgraphics_core PRIVATE src)

# PRIVATE: the public headers need nothing from CUDA or threads. A static library still
# carries them to whoever links it, as link-only dependencies.
target_link_libraries(neraulgraphics_core PRIVATE Threads::Threads)
if (NERAULGRAPHICS_WITH_CUDA)
    if (NERAULGRAPHICS_CUDA_RUNTIME STREQUAL "Static")
        target_link_libraries(neraulgraphics_core PRIVATE CUDA::cudart_static)
    else()
        target_link_libraries(neraulgraphics_core PRIVATE CUDA::cudart)
    endif()
endif()
target_compile_features(neraulgraphics_core PUBLIC cxx_std_17)

# Position-independent so the library can be linked into a shared object, such as
# the Python extension module.
set_target_properties(neraulgraphics_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME core)

target_compile_options(neraulgraphics_core PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>
    # Prints registers per thread and spill counts; a nonzero stack frame means
    # something fell out to local memory.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

# ----------------------------------------------------------------- viewer --

if (NERAULGRAPHICS_BUILD_VIEWER)
    find_package(SDL2 REQUIRED)

    # Links the library exactly the way an outside project would.
    add_executable(neraulgraphics-viewer src/main.cpp)
    target_link_libraries(neraulgraphics-viewer PRIVATE neraulgraphics::core SDL2::SDL2)

    # The COMPILE_LANGUAGE guard is required: nvcc rejects -march=native and -ffast-math.
    # -ffast-math lets the compiler assume NaN never occurs, which is why the
    # renderer's input checks live in render_gpu.cu, not in anything built here.
    target_compile_options(neraulgraphics-viewer PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native -ffast-math>)
endif()

# ------------------------------------------------------------------ tests --

if (NERAULGRAPHICS_BUILD_TESTS)
    enable_testing()
    # Plain C++ built with -ffast-math, like the viewer: the checks must survive it.
    add_executable(neraulgraphics-tests tests/test_renderer.cpp)
    target_link_libraries(neraulgraphics-tests PRIVATE neraulgraphics::core)
    # The BVH builder is library-private, but its tests want it directly.
    target_include_directories(neraulgraphics-tests PRIVATE src)
    target_compile_options(neraulgraphics-tests PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>)
    add_test(NAME renderer COMMAND neraulgraphics-tests)

    # Batched worlds, in their own program: they are large and run on every device available.
    add_executable(neraulgraphics-batch-tests tests/test_batch.cpp)
    target_link_libraries(neraulgraphics-batch-tests PRIVATE neraulgraphics::core)
    target_compile_options(neraulgraphics-batch-tests PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>)
    add_test(NAME batch COMMAND neraulgraphics-batch-tests)

    # The path tracer and its denoiser: closed-form checks of light transport.
    add_executable(neraulgraphics-path-tests tests/test_path.cpp)
    target_link_libraries(neraulgraphics-path-tests PRIVATE neraulgraphics::core)
    target_compile_options(neraulgraphics-path-tests PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-ffast-math>)
    add_test(NAME path COMMAND neraulgraphics-path-tests)
endif()

# ----------------------------------------------------------------- python --

if (NERAULGRAPHICS_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
        OPTIONAL_COMPONENTS Development.SABIModule)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    # STABLE_ABI: on Python 3.12+ one build serves every later version too.
    nanobind_add_module(_core STABLE_ABI NB_STATIC python/bindings.cpp)
    # __version__ comes from the generated version.h, like everything else.
    target_link_libraries(_core PRIVATE neraulgraphics::core)
    install(TARGETS _core LIBRARY DESTINATION neraulgraphics)

    # Type stubs, so editors can autocomplete and type-check against the module.
    nanobind_add_stub(_core_stub
        MODULE _core
        OUTPUT _core.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_core>
        DEPENDS _core)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi DESTINATION neraulgraphics)
endif()

# ---------------------------------------------------------------- install --

if (NERAULGRAPHICS_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(NERAULGRAPHICS_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/neraulgraphics)

    install(TARGETS neraulgraphics_core
        EXPORT neraulgraphicsTargets
        FILE_SET HEADERS)

    # Writes the imported target neraulgraphics::core that find_package hands back.
    install(EXPORT neraulgraphicsTargets
        NAMESPACE neraulgraphics::
        DESTINATION ${NERAULGRAPHICS_CMAKE_DIR})

    configure_package_config_file(cmake/neraulgraphicsConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/neraulgraphicsConfig.cmake
        INSTALL_DESTINATION ${NERAULGRAPHICS_CMAKE_DIR})

    # Before 1.0 a minor version may change the API, so only the same minor matches.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/neraulgraphicsConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/neraulgraphicsConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/neraulgraphicsConfigVersion.cmake
        DESTINATION ${NERAULGRAPHICS_CMAKE_DIR})

    if (NERAULGRAPHICS_BUILD_VIEWER)
        install(TARGETS neraulgraphics-viewer RUNTIME)
    endif()
endif()
