//
// Created by 余君珩 on 21/09/2024.
//

#ifndef MY_APPLICATION_CYOLO5_H
#define MY_APPLICATION_CYOLO5_H

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "net.h"
#include "layer.h"
#include "benchmark.h"
#include <string>
#include <vector>
#include "YoloLayer.h"

namespace yolocv {
    typedef struct {
        int width;
        int height;
    }YoloSize;
}

struct YoloLayerData {
    std::string name;
    int stride;
    ncnn::Mat anchors;
    YoloLayerData(const std::string& name, int stride, const ncnn::Mat& anchors)
            : name(name), stride(stride), anchors(anchors) {}
};

struct imageinfo{
    int image_wid;
    int image_height;
};

struct image_scaled_info{
    float scale;
    int wpad;
    int hpad;
    int height_scale;
    int wid_scale;
};


class Cyolo5 {
    public:
        // Constructor
        Cyolo5(AAssetManager *mgr, const char *param, const char *bin, bool useGPU);

        // Destructor
        ~Cyolo5();

        // Detection method
        std::vector<Object_box> detect(JNIEnv *env, jobject image, float threshold, float nms_threshold);

        // Class labels
        std::vector<std::string> labels{
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                "hair drier", "toothbrush"
        };

    private:
        void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object_box>& objects);
        void nms_sorted_bboxes(const std::vector<Object_box>& faceobjects, std::vector<int>& picked, float nms_threshold);
        void qsort_descent_inplace(std::vector<Object_box>& faceobjects, int left, int right);
        void qsort_descent_inplace(std::vector<Object_box>& faceobjects);
        void extract_output_layer(ncnn::Extractor &ex, ncnn::Mat& out, YoloLayer yoloLayer,std::vector<Object_box> &result,const ncnn::Mat &in_pad,const float &threshold);
        void detect_infer(ncnn::Extractor& ex, ncnn::Mat &in_net, std::vector<Object_box>&results, const float threshold, const float nms_threshold,const ncnn::Mat &in_pad);
    // NCNN Network
        ncnn::Net *yolo5;
        int input_size = 640;  // Input size for the model
        int num_class = 80;     // Number of classes
        bool useGPU;
        std::vector<Object_box> objects;
        imageinfo imageinfo;
        image_scaled_info scale_info;

public:
    // Static variables for GPU usage
    static Cyolo5 *detector;
    static bool toUseGPU;

};

#endif // MY_APPLICATION_CYOLO5_H
