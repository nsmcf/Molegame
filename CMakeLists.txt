cmake_minimum_required(VERSION 3.24)
project(MoleGame VERSION 0.1.0 LANGUAGES CXX)

# Correct a CMake encoding mismatch when detecting Chinese MSVC include output.
if(MSVC AND CMAKE_CL_SHOWINCLUDES_PREFIX MATCHES "娉ㄦ剰")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件: ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(MOLE_WITH_CAMERA "Build the Basler and YOLO preview" ON)
option(MOLE_WITH_TENSORRT "Enable the TensorRT YOLO backend" ON)
set(BASLER_ROOT "" CACHE PATH "Basler pylon installation")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn)
add_executable(MoleGame src/main.cpp src/Game.cpp src/CameraInput.cpp)
target_include_directories(MoleGame PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(MoleGame PRIVATE ${OpenCV_LIBS})
if(WIN32)
    target_compile_definitions(MoleGame PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_compile_options(MoleGame PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
    $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
    $<$<CXX_COMPILER_ID:MSVC>:/permissive->)

if(MOLE_WITH_CAMERA)
    if(NOT WIN32)
        message(FATAL_ERROR "Basler preview currently targets Windows; use -DMOLE_WITH_CAMERA=OFF otherwise.")
    endif()
    find_package(pylon CONFIG REQUIRED HINTS "${BASLER_ROOT}/Development/CMake/pylon")
    target_sources(MoleGame PRIVATE src/vision/YoloHandTracker.cpp)
    target_link_libraries(MoleGame PRIVATE pylon::pylon)
    target_compile_definitions(MoleGame PRIVATE MOLE_WITH_CAMERA)
    if(MOLE_WITH_TENSORRT)
        find_package(CUDAToolkit REQUIRED)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED)
        find_library(TENSORRT_LIBRARY NAMES nvinfer HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
        # The reused detector retains its original compilation switch.
        target_compile_definitions(MoleGame PRIVATE RHYTHM_USE_TENSORRT_YOLO)
        target_include_directories(MoleGame PRIVATE "${TENSORRT_INCLUDE_DIR}")
        target_link_libraries(MoleGame PRIVATE "${TENSORRT_LIBRARY}" CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
    endif()
endif()
