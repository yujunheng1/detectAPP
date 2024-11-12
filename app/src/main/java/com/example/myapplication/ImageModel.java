package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;

import androidx.camera.core.ImageProxy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ImageModel {
    private ImageProxy image;
    private Bitmap tranbitmap;
    private Bitmap rotateBitmap;
    private int width;
    private int height;

    public ImageModel() {
        this.image = null;
        this.tranbitmap = null;
        this.rotateBitmap = null;
    }

    public ImageModel(ImageProxy image) {
        this.image=image;
        this.tranbitmap = imageToBitmap();
        this.rotateBitmap=rotateBitmap();
        this.width=image.getWidth();
        this.height=image.getHeight();
    }

    public void setImage(ImageProxy image) {
        release(); // 先释放当前的资源
        this.image = image; // 设置新的 ImageProxy
        if (this.image != null) {
            this.tranbitmap = imageToBitmap(); // 转换为 Bitmap
            this.rotateBitmap = rotateBitmap(); // 旋转 Bitmap
            this.width=image.getWidth();
            this.height=image.getHeight();
        }
    }
    public int getWidth(){
        return this.width;
    }
    public int getHeight(){
        return this.height;
    }

    public Bitmap getTranbitmap(){
        return this.tranbitmap;
    }
    public Bitmap getRotatebitmap(){
        return this.rotateBitmap;
    }
    private Bitmap imageToBitmap() {
        byte[] nv21 = imagetToNV21(this.image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, this.image.getWidth(), this.image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();
        try {
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
    private byte[] imagetToNV21(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ImageProxy.PlaneProxy y = planes[0];
        ImageProxy.PlaneProxy u = planes[1];
        ImageProxy.PlaneProxy v = planes[2];
        ByteBuffer yBuffer = y.getBuffer();
        ByteBuffer uBuffer = u.getBuffer();
        ByteBuffer vBuffer = v.getBuffer();
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        byte[] nv21 = new byte[ySize + uSize + vSize];
        // U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        return nv21;
    }
    private Bitmap rotateBitmap(){
        int imageRotationDegrees= this.image.getImageInfo().getRotationDegrees();
        Matrix matrix = new Matrix();
        matrix.postRotate(imageRotationDegrees);
        return Bitmap.createBitmap(tranbitmap, 0, 0, tranbitmap.getWidth(), tranbitmap.getHeight(), matrix, false);

    }
    public void clear() {
        if (tranbitmap != null) {
            tranbitmap.recycle(); // 回收 Bitmap 资源
            tranbitmap = null;    // 清空引用
        }
        if (rotateBitmap != null) {
            rotateBitmap.recycle(); // 回收 Bitmap 资源
            rotateBitmap = null;    // 清空引用
        }
        if(image!=null) {
            image.close(); // 关闭 ImageProxy，释放资源
            image=null;
        }
    }

    // 释放资源
    public void release() {
        clear(); // 清空 Bitmap
        if (image != null) {
            image.close(); // 关闭 ImageProxy，释放资源
            image = null;  // 清空引用
        }
    }
}
