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


# 创建共享库
add_library(
        Cyolo5
        SHARED
        YoloLayer.cpp
        Cyolo5.cpp
)

# 链接 ncnn 库
target_link_libraries(
        Cyolo5
        ncnn
        # 你可以在这里添加其他需要的库
)

# 清理 CMake 配置缓存，确保一切重新构建
