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
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures to compile for")
endif()

project(raytracer VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# On by default only when this is the top-level project, so a project that uses
# the library through add_subdirectory or FetchContent never needs SDL2.
option(RAYTRACER_BUILD_VIEWER "Build the interactive SDL2 viewer" ${PROJECT_IS_TOP_LEVEL})
option(RAYTRACER_INSTALL "Generate install rules and the find_package config" ${PROJECT_IS_TOP_LEVEL})

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------- library --

# The renderer on its own: no window, no SDL. Consumers link raytracer::core.
add_library(raytracer_core STATIC src/render_gpu.cu)
add_library(raytracer::core ALIAS raytracer_core)

# The public headers. BASE_DIRS makes include/ the include path, both here and
# once installed, so every consumer writes #include <raytracer/render_gpu.h>.
target_sources(raytracer_core PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/raytracer/render_gpu.h
        include/raytracer/scene.h
        include/raytracer/geom_gpu.cuh)

# PUBLIC: the headers include cuda_runtime.h, so consumers need its include path,
# and they link the CUDA runtime that the kernel launches go through.
target_link_libraries(raytracer_core PUBLIC CUDA::cudart)
target_compile_features(raytracer_core PUBLIC cxx_std_17)

# Position-independent so the library can be linked into a shared object later,
# such as a Python extension module.
set_target_properties(raytracer_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    EXPORT_NAME core)

target_compile_options(raytracer_core PRIVATE
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-use_fast_math>
    # Prints registers per thread and spill counts; a nonzero stack frame means
    # something fell out to local memory.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

# ----------------------------------------------------------------- viewer --

if (RAYTRACER_BUILD_VIEWER)
    find_package(SDL2 REQUIRED)

    # Links the library exactly the way an outside project would.
    add_executable(raytracer src/main.cpp)
    target_link_libraries(raytracer PRIVATE raytracer::core SDL2::SDL2)

    # The COMPILE_LANGUAGE guard is required: nvcc rejects -march=native and -ffast-math.
    # -ffast-math lets the compiler assume NaN never occurs, which is why the
    # renderer's input checks live in render_gpu.cu, not in anything built here.
    target_compile_options(raytracer PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native -ffast-math>)
endif()

# ---------------------------------------------------------------- install --

if (RAYTRACER_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(RAYTRACER_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/raytracer)

    install(TARGETS raytracer_core
        EXPORT raytracerTargets
        FILE_SET HEADERS)

    # Writes the imported target raytracer::core that find_package hands back.
    install(EXPORT raytracerTargets
        NAMESPACE raytracer::
        DESTINATION ${RAYTRACER_CMAKE_DIR})

    configure_package_config_file(cmake/raytracerConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/raytracerConfig.cmake
        INSTALL_DESTINATION ${RAYTRACER_CMAKE_DIR})

    # Before 1.0 a minor version may change the API, so only the same minor matches.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/raytracerConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)

    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/raytracerConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/raytracerConfigVersion.cmake
        DESTINATION ${RAYTRACER_CMAKE_DIR})

    if (RAYTRACER_BUILD_VIEWER)
        install(TARGETS raytracer RUNTIME)
    endif()
endif()
