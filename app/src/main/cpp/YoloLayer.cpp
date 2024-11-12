#include "YoloLayer.h"

// Define global YoloLayer instances
YoloLayer layer_801("801", 32, ncnn::Mat(6));
YoloLayer layer_781("781", 16, ncnn::Mat(6));

// Define layer_out with appropriate parameters
YoloLayer layer_out("output", 8, ncnn::Mat(6)); // Ensure layer_out is defined
YoloLayer layer_out_8("output", 8, ncnn::Mat(6));
YoloLayer layer_out_16("output", 16, ncnn::Mat(6));
YoloLayer layer_out_32("output", 32, ncnn::Mat(6));
void initializeAnchors() {
    ncnn::Mat anchors_32(6);
    anchors_32[0] = 116.f;
    anchors_32[1] = 90.f;
    anchors_32[2] = 156.f;
    anchors_32[3] = 198.f;
    anchors_32[4] = 373.f;
    anchors_32[5] = 326.f;

    layer_801.anchors = anchors_32; // Initialize anchors for layer_801
    layer_out_32.anchors=anchors_32;

    ncnn::Mat anchors_16(6);
    anchors_16[0] = 30.f;
    anchors_16[1] = 61.f;
    anchors_16[2] = 62.f;
    anchors_16[3] = 45.f;
    anchors_16[4] = 59.f;
    anchors_16[5] = 119.f;

    layer_781.anchors = anchors_16; // Initialize anchors for layer_781
    layer_out_16.anchors=anchors_16;

    ncnn::Mat anchors_8(6);
    anchors_8[0] = 10.f;
    anchors_8[1] = 13.f;
    anchors_8[2] = 16.f;
    anchors_8[3] = 30.f;
    anchors_8[4] = 33.f;
    anchors_8[5] = 23.f;

    layer_out.anchors = anchors_8; // Initialize anchors for layer_out
    layer_out_8.anchors=anchors_8;
}
