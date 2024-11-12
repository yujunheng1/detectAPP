cmake_minimum_required(VERSION 3.22.1)
project(myapplication)

# 开启 OpenMP 支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -frtti -fexceptions")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif()

# 设置 NCNN 的路径
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240820-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 创建 yolo8 共享库
add_library(
        yolo8
        SHARED
        YoloLayer.cpp
        yolo8.cpp
)

# 创建 yolo5 共享库


# 链接 ncnn 库到 yolo8
target_link_libraries(
        yolo8
        ncnn
        # 你可以在这里添加其他需要的库
)


