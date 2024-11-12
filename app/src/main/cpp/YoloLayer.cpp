#include "YoloLayer.h"

// Define global YoloLayer instances
YoloLayer layer_801("801", 32, ncnn::Mat(6));
YoloLayer layer_781("781", 16, ncnn::Mat(6));

// Define layer_out with appropriate parameters
YoloLayer layer_out("output", 8, ncnn::Mat(6)); // Ensure layer_out is defined

void initializeAnchors() {
    ncnn::Mat anchors_801(6);
    anchors_801[0] = 116.f;
    anchors_801[1] = 90.f;
    anchors_801[2] = 156.f;
    anchors_801[3] = 198.f;
    anchors_801[4] = 373.f;
    anchors_801[5] = 326.f;

    layer_801.anchors = anchors_801; // Initialize anchors for layer_801

    ncnn::Mat anchors_781(6);
    anchors_781[0] = 30.f;
    anchors_781[1] = 61.f;
    anchors_781[2] = 62.f;
    anchors_781[3] = 45.f;
    anchors_781[4] = 59.f;
    anchors_781[5] = 119.f;

    layer_781.anchors = anchors_781; // Initialize anchors for layer_781

    ncnn::Mat anchors_out(6);
    anchors_out[0] = 10.f;
    anchors_out[1] = 13.f;
    anchors_out[2] = 16.f;
    anchors_out[3] = 30.f;
    anchors_out[4] = 33.f;
    anchors_out[5] = 23.f;

    layer_out.anchors = anchors_out; // Initialize anchors for layer_out
}