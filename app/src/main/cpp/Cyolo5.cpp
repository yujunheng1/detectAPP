#include <cpu.h>
#include "Cyolo5.h"

#include "jni.h"
#include "YoloLayer.h"


extern "C" JNIEXPORT jlong JNICALL
Java_com_example_myapplication_YOLO5_createNativeInstance(JNIEnv *env, jobject thiz, jobject assetManager, jboolean useGPU) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    Cyolo5* detector = new Cyolo5(mgr, "yolov5s.param", "yolov5s.bin", useGPU);

    return reinterpret_cast<jlong>(detector);
}

// 检测方法
extern "C" JNIEXPORT jobject JNICALL
Java_com_example_myapplication_YOLO5_detectNative(JNIEnv *env, jobject thiz, jlong nativePtr, jobject image, jfloat threshold, jfloat nmsThreshold) {
    Cyolo5* detector = reinterpret_cast<Cyolo5*>(nativePtr);
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
Java_com_example_myapplication_YOLO5_releaseNativeInstance(JNIEnv *env, jobject thiz, jlong nativePtr) {
    delete reinterpret_cast<Cyolo5*>(nativePtr);
}

Cyolo5 *Cyolo5::detector = nullptr;
//static ncnn::Net yolo5;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

Cyolo5::Cyolo5(AAssetManager *mgr, const char *param, const char *bin, bool useGPU)
{

    initializeAnchors();
    yolo5 = new ncnn::Net();
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

    yolo5->opt = opt;
    yolo5->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
    int ret_param = yolo5->load_param(mgr, param);
    if (ret_param != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Failed to load param: %d", ret_param);
    }

    int ret_model = yolo5->load_model(mgr, bin);
    if (ret_model !=0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Failed to load model: %d", ret_model);
    }

}

Cyolo5::~Cyolo5() {
    yolo5->clear();
    delete yolo5;
}
void Cyolo5::qsort_descent_inplace(std::vector<Object_box>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}
void Cyolo5::qsort_descent_inplace(std::vector<Object_box>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].score;

    while (i <= j)
    {
        while (faceobjects[i].score > p)
            i++;

        while (faceobjects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

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
void Cyolo5::extract_output_layer(ncnn::Extractor& ex, ncnn::Mat& out, YoloLayer yoloLayer,std::vector<Object_box> &result,const ncnn::Mat &in_pad,const float &threshold) {

    int issu_output = ex.extract(yoloLayer.name, out);
    if (issu_output != 0) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Failed to extract output from output layer!");
    }
    if (out.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Output from output layer is empty!");
    }

    generate_proposals(yoloLayer.anchors, yoloLayer.stride, in_pad, out, threshold,
                       yoloLayer.objects);
    result.insert(result.begin(), yoloLayer.objects.begin(), yoloLayer.objects.end());
}
void Cyolo5::detect_infer(ncnn::Extractor& ex, ncnn::Mat &in_net, std::vector<Object_box> &result, const float threshold, const float nms_threshold,const ncnn::Mat &in_pad) {

    if (in_net.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Input net is empty!");
    }

    int issu_input=ex.input("images", in_pad);

    if (issu_input==-1) {
        __android_log_print(ANDROID_LOG_ERROR, "YOLO5", "Input net is fail!");
    }

    {
        ncnn::Mat out;
        extract_output_layer(ex,out,layer_out,result,in_pad,threshold);
    }
    // stride 16
    {
        ncnn::Mat out;
        extract_output_layer(ex,out,layer_781,result,in_pad,threshold);
    }

    // stride 32
    {
        ncnn::Mat out;
        extract_output_layer(ex,out,layer_801,result,in_pad,threshold);
    }

    qsort_descent_inplace(result);
    std::vector<int> picked;
    nms_sorted_bboxes(result, picked, nms_threshold);

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
std::vector<Object_box> Cyolo5::detect(JNIEnv *env, jobject image, float threshold, float nms_threshold) {
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
    ncnn::Extractor ex = yolo5->create_extractor();
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


void Cyolo5::generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object_box>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object_box obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h= y1 - y0;
                    obj.label = class_index;
                    obj.score = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}
static inline float intersection_area(const Object_box& a, const Object_box& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {

        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}
void Cyolo5::nms_sorted_bboxes(const std::vector<Object_box>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object_box& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object_box& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


