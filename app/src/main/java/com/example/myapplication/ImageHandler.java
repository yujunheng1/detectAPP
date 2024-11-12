package com.example.myapplication;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import java.io.IOException;

public class ImageHandler {
    private Bitmap bitmap;

    // 构造函数：通过图像路径加载 Bitmap
    public ImageHandler(String imagePath) {
        this.bitmap = BitmapFactory.decodeFile(imagePath);
        this.bitmap = rotateBitmap(this.bitmap, getRotationAngle(imagePath));
    }

    // 获取旋转角度
    private int getRotationAngle(String imagePath) {
        int rotation = 0;
        try {
            ExifInterface exif = new ExifInterface(imagePath);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotation = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotation = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotation = 270;
                    break;
                default:
                    rotation = 0;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rotation;
    }


    // 旋转 Bitmap 的函数
    private Bitmap rotateBitmap(Bitmap bitmap, int angle) {
        if (bitmap == null || angle == 0) {
            return bitmap;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    // 获取旋转后的 Bitmap
    public Bitmap getBitmap() {
        return bitmap;
    }

    // 可选：释放 Bitmap 资源
    public void recycle() {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
            bitmap = null;
        }
    }
}