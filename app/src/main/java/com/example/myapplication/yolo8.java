package com.example.myapplication;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.util.ArrayList;
import java.util.List;

public class yolo8 {
    static {
        System.loadLibrary("yolo8");
    }
    private long nativePtr; // 指向 C++ 的 Cyolo5 实例的指针

    // 构造函数，初始化 Cyolo5
    public yolo8(AssetManager assetManager, boolean useGPU) {
        nativePtr = createNativeInstance(assetManager,  useGPU);
    }

    // JNI 方法，用于创建 Cyolo5 实例
    private native long createNativeInstance(AssetManager assetManager, boolean useGPU);

    // 检测方法
    public List<Object_box> detect(Bitmap image, float threshold, float nmsThreshold) {
        Object[] resultArray = new List[]{detectNative(nativePtr, image, threshold, nmsThreshold)};
        return convertToObjectList(resultArray);
    }
    private List<Object_box> convertToObjectList(Object[] resultArray) {
        ArrayList<Object_box> reresult = (ArrayList<Object_box>) resultArray[0];
        List<Object_box>result=(List<Object_box>) reresult;

        return result;
    }

    private Object_box getObjectFromJObject(Object jObj) {
        try {
            Class<?> objectClass = jObj.getClass();
            float x1 = (float) objectClass.getField("x1").get(jObj);
            float y1 = (float) objectClass.getField("y1").get(jObj);
            float x2 = (float) objectClass.getField("x2").get(jObj);
            float y2 = (float) objectClass.getField("y2").get(jObj);
            float score = (float) objectClass.getField("score").get(jObj);
            int label = (int) objectClass.getField("label").get(jObj);

            return new Object_box(x1, y1, x2, y2, score, label);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    // JNI 方法，用于调用 C++ 的 detect 方法
    private native List<Object> detectNative(long nativePtr, Bitmap image, float threshold, float nmsThreshold);

    // 释放资源
    public void release() {
        if (nativePtr != 0) {
            releaseNativeInstance(nativePtr);
            nativePtr = 0;
        }
    }

    // JNI 方法，用于释放 Cyolo5 实例
    private native void releaseNativeInstance(long nativePtr);
}
