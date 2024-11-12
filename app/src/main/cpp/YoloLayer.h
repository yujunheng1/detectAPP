//
// Created by junhengyu on 22/10/2024.
//

#ifndef MY_APPLICATION_YOLOLAYER_H
#define MY_APPLICATION_YOLOLAYER_H
#include <string>
#include "ncnn-20240820-android-vulkan/x86/include/ncnn/net.h"
#include "ncnn-20240820-android-vulkan/armeabi-v7a/include/ncnn/layer.h"
struct Object_box {
    float x;
    float y;
    float w;
    float h;
    float score;
    int label;
};

struct YoloLayer {
    const char *name;
    int stride;
    ncnn::Mat anchors;
    std::vector<Object_box> objects;

    YoloLayer(const char *name, int stride, ncnn::Mat anchors)
            : name(name), stride(stride), anchors(anchors) {
    }
};
extern YoloLayer layer_801;
extern YoloLayer layer_781;
extern YoloLayer layer_out;

void initializeAnchors();
#endif //MY_APPLICATION_YOLOLAYER_H