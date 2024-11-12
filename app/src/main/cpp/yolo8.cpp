#include <cpu.h>
#include "yolo8.h"

#include "jni.h"
#include "YoloLayer.h"
#include "utils.h"


extern "C" JNIEXPORT jlong JNICALL
Java_com_example_myapplication_yolo8_createNativeInstance(JNIEnv *env, jobject thiz, jobject assetManager, jboolean useGPU) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolo8* detector = new yolo8(mgr, "yolov8n.param", "yolov8n.bin", useGPU);

    return reinterpret_cast<jlong>(detector);
}

// 检测方法
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_myapplication_yolo8_detectNative(JNIEnv *env, jobject thiz, jlong nativePtr, jobject image, jfloat threshold, jfloat nmsThreshold) {
    yolo8* detector = reinterpret_cast<yolo8*>(nativePtr);
    std::vector<Object_box> detections = detector->detect(env, image, threshold, nmsThreshold);
    jclass objectClass = env->FindClass("com/example/myapplication/Object_box"); // 根据实际路径修改
    jmethodID constructor = env->GetMethodID(objectClass, "<init>", "(FFFFFI)V"); // 构造函数签名

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jobject arrayList = env->NewObject(arrayListClass, arrayListConstructor);

    // 将检测结果转换为 Java 对象并添加到结果数组中
    for (size_t i = 0; i < detections.size(); ++i) {
        Object_box &det = detections[i];
        jobject objectBox = env->NewObject(objectClass, constructor, det.x, det.y, det.w, det.h, det.score, det.label);

        // 将 Object_box 添加到 ArrayList 中
        env->CallBooleanMethod(arrayList, arrayListAdd, objectBox);

        // 删除局部引用
        env->DeleteLocalRef(objectBox);
    }

    return arrayList; // 返回包含检测结果的数组
}

// 释放资源
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_yolo8_releaseNativeInstance(JNIEnv *env, jobject thiz, jlong nativePtr) {
    delete reinterpret_cast<yolo8*>(nativePtr);
}

yolo8 *yolo8::detector = nullptr;
//static ncnn::Net yolo5;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;



yolo8::yolo8(AAssetManager *mgr, const char *param, const char *bin, bool useGPU)
{

    initializeAnchors();
    yolo8_detect = new ncnn::Net();
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    this->useGPU=useGPU;
    this->objects.clear();

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = ncnn::get_big_cpu_count();
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    yolo8_detect->opt = opt;
    //yolo8_detect->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    int ret_param = yolo8_detect->load_param(mgr, param);
    if (ret_param != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Failed to load param: %d", ret_param);
    }

    int ret_model = yolo8_detect->load_model(mgr, bin);
    if (ret_model !=0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Failed to load model: %d", ret_model);
    }

}

yolo8::~yolo8() {
    yolo8_detect->clear();
    delete yolo8_detect;
}
//void yolo8::qsort_descent_inplace(std::vector<Object_box>& faceobjects)
//{
//    if (faceobjects.empty())
//        return;
//
//    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
//}
//void yolo8::qsort_descent_inplace(std::vector<Object_box>& faceobjects, int left, int right)
//{
//    int i = left;
//    int j = right;
//    float p = faceobjects[(left + right) / 2].score;
//
//    while (i <= j)
//    {
//        while (faceobjects[i].score > p)
//            i++;
//
//        while (faceobjects[j].score < p)
//            j--;
//
//        if (i <= j)
//        {
//            // swap
//            std::swap(faceobjects[i], faceobjects[j]);
//
//            i++;
//            j--;
//        }
//    }
//
//#pragma omp parallel sections
//    {
//#pragma omp section
//        {
//            if (left < j) qsort_descent_inplace(faceobjects, left, j);
//        }
//#pragma omp section
//        {
//            if (i < right) qsort_descent_inplace(faceobjects, i, right);
//        }
//    }
//}

float scale_image(int* wid, int* height, int input_size) {

    float scale;
    if (*wid > *height) {
        scale = static_cast<float>(input_size) / *wid;
        *wid = input_size;
        *height = static_cast<int>(*height * scale);
    } else {
        scale = static_cast<float>(input_size) / *height;
        *height = input_size;
        *wid = static_cast<int>(*wid * scale);
    }
    return scale;
}
void yolo8::extract_output_layer(ncnn::Extractor& ex, ncnn::Mat& out,std::vector<Object_box> &result,const ncnn::Mat &in_pad,const float &threshold) {

    int issu_output = ex.extract(layer_out.name, out);
    if (issu_output != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Failed to extract output from output layer!");
    }
    if (out.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Output from output layer is empty!");
    }


    std::vector<Object_box> proposals;
    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, threshold, proposals);
    result.insert(result.begin(), proposals.begin(), proposals.end());

}
void yolo8::detect_infer(ncnn::Extractor& ex, ncnn::Mat &in_net, std::vector<Object_box> &result, const float threshold, const float nms_threshold,const ncnn::Mat &in_pad) {

    if (in_net.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Input net is empty!");
    }

    int issu_input=ex.input("images", in_pad);

    if (issu_input==-1) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO8", "Input net is fail!");
    }
    // stride 8
    {
        ncnn::Mat out;
        extract_output_layer(ex,out,result,in_pad,threshold);
    }


    utils::qsort_descent_inplace(result);
    std::vector<int> picked;
    utils::nms_sorted_bboxes(result, picked, nms_threshold);

    int count = picked.size();
    this->objects.clear();
    objects.resize(count);
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = result[picked[i]];

        // adjust offset to original unpadded
        float x1 = (objects[i].x - (scale_info.wpad / 2)) / scale_info.scale;
        float y1 = (objects[i].y - (scale_info.hpad / 2)) / scale_info.scale;
        float x2 = (objects[i].x + objects[i].w - (scale_info.wpad / 2)) / scale_info.scale;
        float y2 = (objects[i].y + objects[i].h - (scale_info.hpad / 2)) / scale_info.scale;



        // clip
        x1 = std::max(std::min(x1, (float)(imageinfo.image_wid - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(imageinfo.image_height - 1)), 0.f);
        x2 = std::max(std::min(x2, (float)(imageinfo.image_wid - 1)), 0.f);
        y2 = std::max(std::min(y2, (float)(imageinfo.image_height - 1)), 0.f);

        objects[i].x = x1;
        objects[i].y = y1;
        objects[i].w = x2 - x1;
        objects[i].h = y2 - y1;
    }

}
std::vector<Object_box> yolo8::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
    AndroidBitmapInfo img_size;
    AndroidBitmap_getInfo(env, image, &img_size);
    imageinfo.image_wid=img_size.width;
    imageinfo.image_height=img_size.height;

    scale_info.wid_scale=imageinfo.image_wid;
    scale_info.height_scale=imageinfo.image_height;

    scale_info.scale = scale_image(&scale_info.wid_scale, &scale_info.height_scale, input_size);


    ncnn::Mat in_net = ncnn::Mat::from_android_bitmap_resize(env, image, ncnn::Mat::PIXEL_RGB, scale_info.wid_scale ,
                                                             scale_info.height_scale );

    scale_info.wpad= (scale_info.wid_scale + 31) / 32 * 32 - scale_info.wid_scale;
    scale_info.hpad = (scale_info.height_scale + 31) / 32 * 32 - scale_info.height_scale;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in_net, in_pad, scale_info.hpad / 2, scale_info.hpad - scale_info.hpad / 2, scale_info.wpad / 2, scale_info.wpad - scale_info.wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    float mean[3] = {0, 0, 0};
    in_net.substract_mean_normalize(mean, norm);
    ncnn::Extractor ex = yolo8_detect->create_extractor();
    ex.set_light_mode(true);
    ex.set_vulkan_compute(this->useGPU);
    std::vector<Object_box>results;
    detect_infer(ex,in_net,results,threshold,nms_threshold,in_pad);

    return this->objects;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

void yolo8::generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
//void yolo8::generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object_box>& objects)
//void yolo8::generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object_box>& objects)
//{
//    const int num_points = grid_strides.size();
//    const int num_class = 80;
//    const int reg_max_1 = 16;
//
//    for (int i = 0; i < num_points; i++)
//    {
//        const float* scores = pred.row(i) + 4 * reg_max_1;
//
//        // find label with max score
//        int label = -1;
//        float score = -FLT_MAX;
//        for (int k = 0; k < num_class; k++)
//        {
//            float confidence = scores[k];
//            if (confidence > score)
//            {
//                label = k;
//                score = confidence;
//            }
//        }
//        float box_prob = sigmoid(score);
//        if (box_prob >= prob_threshold)
//        {
//            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
//            {
//                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
//
//                ncnn::ParamDict pd;
//                pd.set(0, 1); // axis
//                pd.set(1, 1);
//                softmax->load_param(pd);
//
//                ncnn::Option opt;
//                opt.num_threads = 1;
//                opt.use_packing_layout = false;
//
//                softmax->create_pipeline(opt);
//
//                softmax->forward_inplace(bbox_pred, opt);
//
//                softmax->destroy_pipeline(opt);
//
//                delete softmax;
//            }
//
//            float pred_ltrb[4];
//            for (int k = 0; k < 4; k++)
//            {
//                float dis = 0.f;
//                const float* dis_after_sm = bbox_pred.row(k);
//                for (int l = 0; l < reg_max_1; l++)
//                {
//                    dis += l * dis_after_sm[l];
//                }
//
//                pred_ltrb[k] = dis * grid_strides[i].stride;
//            }
//
//            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
//            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;
//
//            float x0 = pb_cx - pred_ltrb[0];
//            float y0 = pb_cy - pred_ltrb[1];
//            float x1 = pb_cx + pred_ltrb[2];
//            float y1 = pb_cy + pred_ltrb[3];
//
//            Object_box obj;
//            obj.x = x0;
//            obj.y = y0;
//            obj.w = x1 - x0;
//            obj.h= y1 - y0;
//            obj.label = label;
//            obj.score = box_prob;
//
//            objects.push_back(obj);
//
//        }
//    }
//}
void yolo8::generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object_box>& objects)
{
    const int num_points = grid_strides.size();
    const int reg_max_1 = 16;

    // 定义删除器
    auto softmax_deleter = [](ncnn::Layer* layer) {
        if (layer) {
            ncnn::Option opt;
            layer->destroy_pipeline(opt);
            delete layer;
        }
    };

    // 创建 Softmax 层并设置参数
    auto create_softmax_layer = []() -> ncnn::Layer* {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        pd.set(0, 1); // axis
        pd.set(1, 1);
        softmax->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        softmax->create_pipeline(opt);

        return softmax;
    };

    // 使用自定义删除器的 std::unique_ptr
    std::unique_ptr<ncnn::Layer, decltype(softmax_deleter)> softmax_layer(create_softmax_layer(), softmax_deleter);

    for (int i = 0; i < num_points; ++i)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float max_score = -FLT_MAX;
        for (int k = 0; k < num_class; ++k)
        {
            if (scores[k] > max_score)
            {
                max_score = scores[k];
                label = k;
            }
        }

        float box_prob = sigmoid(max_score);
        if (box_prob >= prob_threshold)
        {
            // Softmax on bounding box prediction
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            softmax_layer->forward_inplace(bbox_pred, ncnn::Option());

            float pred_ltrb[4];
            for (int k = 0; k < 4; ++k)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; ++l)
                {
                    dis += l * dis_after_sm[l];
                }
                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            // 计算中心点坐标
            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            // 计算边界框
            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            // 创建对象并添加到输出中
            Object_box obj;
            obj.x = x0;
            obj.y = y0;
            obj.w = x1 - x0;
            obj.h = y1 - y0;
            obj.label = label;
            obj.score = box_prob;

            objects.push_back(obj);
        }
    }
}

//static inline float intersection_area(const Object_box& a, const Object_box& b)
//{
//    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
//    {
//
//        return 0.f;
//    }
//
//    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
//    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);
//
//    return inter_width * inter_height;
//}
//void yolo8::nms_sorted_bboxes(const std::vector<Object_box>& faceobjects, std::vector<int>& picked, float nms_threshold)
//{
//    picked.clear();
//
//    const int n = faceobjects.size();
//
//    std::vector<float> areas(n);
//    for (int i = 0; i < n; i++)
//    {
//        areas[i] = faceobjects[i].w * faceobjects[i].h;
//    }
//
//    for (int i = 0; i < n; i++)
//    {
//        const Object_box& a = faceobjects[i];
//
//        int keep = 1;
//        for (int j = 0; j < (int)picked.size(); j++)
//        {
//            const Object_box& b = faceobjects[picked[j]];
//
//            // intersection over union
//            float inter_area = intersection_area(a, b);
//            float union_area = areas[i] + areas[picked[j]] - inter_area;
//            // float IoU = inter_area / union_area
//            if (inter_area / union_area > nms_threshold)
//                keep = 0;
//        }
//
//        if (keep)
//            picked.push_back(i);
//    }
//}


