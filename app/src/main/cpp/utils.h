//
// Created by 余君珩 on 25/10/2024.
//

#ifndef MY_APPLICATION_UTILS_H
#define MY_APPLICATION_UTILS_H
#include "YoloLayer.h"

namespace utils {

    //void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object_box>& objects);
    void nms_sorted_bboxes(const std::vector<Object_box>& faceobjects, std::vector<int>& picked, float nms_threshold);
    void qsort_descent_inplace(std::vector<Object_box>& faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object_box>& faceobjects);
//    inline float fast_exp(float x);
    //inline float sigmoid(float x);
    static inline float intersection_area(const Object_box& a, const Object_box& b);

    //void extract_output_layer(ncnn::Extractor &ex, ncnn::Mat& out, YoloLayer yoloLayer,std::vector<Object_box> &result,const ncnn::Mat &in_pad,const float &threshold);
    //void detect_infer(ncnn::Extractor& ex, ncnn::Mat &in_net, std::vector<Object_box>&results, const float threshold, const float nms_threshold,const ncnn::Mat &in_pad);

} // namespace util
#endif //MY_APPLICATION_UTILS_H
