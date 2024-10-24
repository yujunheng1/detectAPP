package com.example.myapplication;

import static androidx.camera.core.CameraX.*;

import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ImageFormat;
import android.graphics.Matrix;

import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraX;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import com.example.myapplication.databinding.ActivityMainBinding;

import android.graphics.Paint;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;
import android.graphics.YuvImage;
import android.graphics.Rect;
import android.graphics.BitmapFactory;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCase;
import androidx.camera.core.impl.ImageAnalysisConfig;
import androidx.camera.lifecycle.ProcessCameraProvider;



import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;


import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.File;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding activityMainBinding;

    private static final int CAMERA_REQUEST_CODE = 100;
    private PreviewView textureView;
    private YOLO5 yoloDetector =null;
    private long lastProcessedTime = 0;
    private static final long PROCESS_INTERVAL_MS = 1000; // 1秒

    private ImageReader imageReader;
    private long lastAnalysisTime = 0;
    AssetManager assetManager;
    private boolean useGPU=false;
    private List<Object_box> ObjectLists= new ArrayList<>();
    private objectResult_view Object_box_view;

    ExecutorService detectService = Executors.newSingleThreadExecutor();

    private int frameCount = 10;
    private int frameCounter = 0;
    private long lastFpsTimestamp = 0;
    protected Bitmap mutableBitmap;
    private Button startCameraButton;
    private Button stopCameraButton;
    private Button buttonTakePhoto;
    //private static final int CAMERA_REQUEST_CODE = 1001; // 用于打开相机
    private static final int VIDEO_REQUEST_CODE = 1002;  // 用于录像
    private int type_camera=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        activityMainBinding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(activityMainBinding.getRoot());

        setContentView(R.layout.activity_main);
        assetManager = getAssets();
        textureView = findViewById(R.id.textureView);
        Object_box_view = findViewById(R.id.Object_box_view);
         startCameraButton = findViewById(R.id.startCameraButton);
        stopCameraButton = findViewById(R.id.stopCameraButton);
        buttonTakePhoto=findViewById(R.id.buttonTakePhoto);

        requestCameraPermission();

    }

    private boolean checkAndUseGPU() {
        if (isGPUSupported()) {
            Toast.makeText(this, "Using GPU for processing.", Toast.LENGTH_SHORT).show();
            return true;
        } else {
            Toast.makeText(this, "GPU is not supported. Using CPU.", Toast.LENGTH_SHORT).show();
            return false;
        }
    }

    private boolean isGPUSupported() {
        try {
            ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
            return activityManager.getDeviceConfigurationInfo().reqGlEsVersion >= 0x00030000; // OpenGL ES 3.0 及以上
        } catch (Exception e) {
            return false;
        }
    }

    private void requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        } else {
            startCameraPreview(type_camera);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCameraPreview(type_camera);
            } else {
                startCameraButton.setBackgroundColor(getResources().getColor(R.color.button_default_color));
                Toast.makeText(this, "Camera permission is declined", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void loadYOLOModelInBackground() {
        new Thread(() -> {
            // 模型加载可以在这里进行，独立于相机启动
            long modelStartTime = System.currentTimeMillis();

            // 初始化YOLO模型
            yoloDetector = new YOLO5(assetManager, useGPU);

            long modelEndTime = System.currentTimeMillis();
            long modelDuration = modelEndTime - modelStartTime;
            Log.d("YOLOModel", "YOLO model loaded in " + modelDuration + " ms");

        }).start();
    }


    private void takePhoto(ImageCapture imageCapture,boolean isSave) {
        File photoFile = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "photo.jpg");
        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        // 拍照成功后处理图像
                        Bitmap imageBitmap = BitmapFactory.decodeFile(photoFile.getAbsolutePath());

                        // 显示定格的图片在 ImageView
                        ImageView photoPreview = findViewById(R.id.photoPreview);
                        photoPreview.setImageBitmap(imageBitmap);
                        photoPreview.setVisibility(View.VISIBLE);

                        findViewById(R.id.buttonTakePhoto).setVisibility(View.GONE);
                        findViewById(R.id.startCameraButton).setVisibility(View.GONE);
                        findViewById(R.id.stopCameraButton).setVisibility(View.GONE);

                        // 隐藏拍照按钮，显示确认按钮
                        findViewById(R.id.confirmButton).setVisibility(View.VISIBLE);
                        findViewById(R.id.cancelButton).setVisibility(View.VISIBLE);

                        // 点击确认按钮
                        findViewById(R.id.confirmButton).setOnClickListener(v -> {
                            // 返回上一个页面并附加图片
                            Intent resultIntent = new Intent();
                            if (isSave) {
                                // 如果 isSave 为 true，保存图片并返回 URI
                                resultIntent.putExtra("photo_uri", Uri.fromFile(photoFile).toString());
                            } else {
                                // 如果不保存，仅返回图片 URI
                                photoFile.delete(); // 删除临时文件
                                resultIntent.putExtra("photo_uri", Uri.fromFile(photoFile).toString());
                            }

                            setResult(RESULT_OK, resultIntent);
                            finish(); // 结束当前页面并返回
                        });

                        findViewById(R.id.cancelButton).setOnClickListener(v -> {
                            // 取消操作并回到拍照状态
                            photoPreview.setVisibility(View.GONE); // 隐藏照片预览
                            findViewById(R.id.confirmButton).setVisibility(View.GONE); // 隐藏确认按钮
                            findViewById(R.id.cancelButton).setVisibility(View.GONE); // 隐藏取消按钮

                            // 显示原来的拍照和视频按钮
                            findViewById(R.id.buttonTakePhoto).setVisibility(View.VISIBLE);
                            findViewById(R.id.startCameraButton).setVisibility(View.VISIBLE);
                            findViewById(R.id.stopCameraButton).setVisibility(View.VISIBLE);
                        });
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        // 错误处理
                        Log.e("CameraPreview", "Error taking picture: " + exception.getMessage());
                    }
                });
    }


    private void startCameraPreview(int type_camera) {
        long startTime = System.currentTimeMillis();
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider processCameraProvider = cameraProviderFuture.get();


                PreviewView viewFinder = (PreviewView)findViewById(R.id.textureView);
                Preview preview = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setTargetRotation(textureView.getDisplay().getRotation())
                        //.setTargetResolution(new Size(480, 640))
                        .build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());

                //select camera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();
                //use for saving image
                ImageCapture imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();
                processCameraProvider.unbindAll();
                if (type_camera == 0) {
                    // type_camera 为 0 时，初始化 ImageAnalysis、加载 YOLO 模型并设置定时分析
                    useGPU = checkAndUseGPU();
                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            //.setTargetResolution(new Size(480, 640))
                            .setTargetRotation(textureView.getDisplay().getRotation())
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            //.setTargetRotation(Surface.ROTATION_90)
                            .build();

                    // 设置定时分析任务
                    imageAnalysis.setAnalyzer(detectService, new DetectAnalyzer());

                    // 绑定 Preview 和 ImageAnalysis
                    processCameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis,imageCapture);


                    loadYOLOModelInBackground();

                    // 打印启动时间
                    long endTime = System.currentTimeMillis();
                    long duration = endTime - startTime;
                    Log.d("CameraPreview", "Camera preview with analysis and YOLO started in " + duration + " ms");
                }else if(type_camera == 1){


                    // 绑定 Preview 和 ImageCapture
                    processCameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);

                    // 打印启动时间
                    long endTime = System.currentTimeMillis();
                    long duration = endTime - startTime;
                    Log.d("CameraPreview", "Camera preview with capture mode started in " + duration + " ms");

                    // 设置拍照按钮监听器

                }
                Object_box_view.setTextureViewSize(textureView.getLeft(),textureView.getTop(),textureView.getRight(),textureView.getBottom());

                buttonTakePhoto.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        v.setBackgroundColor(getResources().getColor(R.color.button_pressed_color));
                        takePhoto(imageCapture, false);  // 拍照方法
                        Log.d("CameraPreview", "Take a picture");
                    }
                });


                startCameraButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        v.setBackgroundColor(getResources().getColor(R.color.button_pressed_color));
                    }
                });

                stopCameraButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        stopCamera();
                    }
                });

            } catch (ExecutionException | InterruptedException e) {

                throw new RuntimeException(e);
            }
        }, ContextCompat.getMainExecutor(this));
        loadYOLOModelInBackground();
    }


    private class DetectAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(@NonNull ImageProxy image) {
            int videoWidth = image.getWidth();
            int videoHeight = image.getHeight();
            Log.d("CameraX", "Video resolution: " + videoWidth + "x" + videoHeight);
            detectbymodel(image);
        }
    }
    public Bitmap rotateBitmapClockwise90(Bitmap tranbitmap) {
        // 创建一个新的 Matrix 对象
        Matrix matrix = new Matrix();

        // 设置旋转90度
        matrix.postRotate(90);

        // 使用 matrix 创建新的 Bitmap
        Bitmap rotatedBitmap = Bitmap.createBitmap(tranbitmap, 0, 0, tranbitmap.getWidth(), tranbitmap.getHeight(), matrix, true);

        return rotatedBitmap;
    }
    private void detectbymodel(ImageProxy image) {
        final int imageRotationDegrees= image.getImageInfo().getRotationDegrees();
        final Bitmap tranbitmap = imageToBitmap(image);
        if (detectService == null) {
            return;
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(imageRotationDegrees);
        Bitmap bitmap = Bitmap.createBitmap(tranbitmap, 0, 0, tranbitmap.getWidth(), tranbitmap.getHeight(), matrix, false);
        //Bitmap roBitmap=rotateBitmapClockwise90(bitmap);
        logFPS(tranbitmap);
        int videoWidth = image.getWidth();
        int videoHeight = image.getHeight();
        Log.d("CameraX", "Video resolution: " + videoWidth + "x" + videoHeight);
        boolean draw=detectDraw(bitmap);
        if (draw==false){
            Log.d("DetectionResult", "No detect object no draw ");
        }
        else{
            Log.d("DetectionResult", "Detected objects count ");
        }
        //updateUIWithDetectionResults();
        image.close();
    }
    private boolean detectDraw(Bitmap bitmap){
        List<Object_box> result = yoloDetector.detect(bitmap, 0.6f, 0.6f);
        int resultSize = result.size();
        if(resultSize==0){
            return false;
        }
//        int videoWidth=bitmap.getWidth();
//        int videoHeight=bitmap.getHeight();
        int videoWidth=bitmap.getWidth();
        int videoHeight=bitmap.getHeight();

        Object_box_view.setDetectionResults(result,videoWidth,videoHeight);
        return true;
    }

//    private void updateUIWithDetectionResults() {
//        runOnUiThread(() -> {
//            //activityCameraBinding.textPrediction.setText("Detected: Object");
//        });
//    }


    private void logFPS(Bitmap bitmap) {

        if (++frameCounter % frameCount == 0) {
            frameCounter = 0;


            long now = System.currentTimeMillis();
            long delta = now - lastFpsTimestamp;
            float fps = 1000f * frameCount / (float) delta;

            Log.d("TAG", String.format("FPS: %.02f with tensorSize: %d x %d", fps, bitmap.getWidth(), bitmap.getHeight()));


            lastFpsTimestamp = now;
        }
    }






    private Bitmap imageToBitmap(ImageProxy image) {
        byte[] nv21 = imagetToNV21(image);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
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




    private void stopCamera() {

        try {
            if (yoloDetector != null) {
                yoloDetector.release();
                yoloDetector = null;
            }
            if (detectService != null) {
                detectService.shutdown();
                detectService = null; // 释放资源
            }
        } catch (Exception e) {
            Log.e("CameraStop", "Error stopping camera: " + e.getMessage());
            e.printStackTrace();
        }

    }
}