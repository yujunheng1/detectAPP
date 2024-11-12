#include "utils.h"
namespace utils {
    void qsort_descent_inplace(std::vector<Object_box>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }
    void qsort_descent_inplace(std::vector<Object_box>& faceobjects, int left, int right)
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

//    inline float fast_exp(float x) {
//        union {
//            uint32_t i;
//            float f;
//        } v{};
//        v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
//        return v.f;
//    }

//    inline float sigmoid(float x) {
//        return 1.0f / (1.0f + fast_exp(-x));
//    }

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

    void nms_sorted_bboxes(const std::vector<Object_box>& faceobjects, std::vector<int>& picked, float nms_threshold)
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

}