package com.example.myapplication;


import android.Manifest;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import android.content.res.ColorStateList;
import android.graphics.Bitmap;

import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;



import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import android.view.GestureDetector;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.myapplication.databinding.ActivityMainBinding;
import com.google.common.util.concurrent.ListenableFuture;


import java.io.File;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding activityMainBinding;
    private static final int CAMERA_REQUEST_CODE = 100;
    private PreviewView textureView;// use for preview the video stream
    private yolo8 yoloDetector =null; //use for yolo model

    AssetManager assetManager;//use for get the resource of model,point to the asset
    private boolean useGPU=false;// detect gpu
    private objectResult_view Object_box_view;
    private ImageModel imageHandlerFromProxy; //use for transfer the image form the video

    ExecutorService detectService = Executors.newSingleThreadExecutor();// thread for detect

    // Constant to define the number of frames to be processed (used for FPS calculations)
    private final int frameCount = 10;

    // Counter to keep track of the number of frames processed
    private int frameCounter = 0;

    // Timestamp for the last frame-per-second (FPS) calculation, used to determine when to update the FPS display
    private long lastFpsTimestamp = 0;

    // Button to stop the camera feed
    private Button stopCameraButton;

    // Button to take a photo from the camera
    private Button buttonTakePhoto;

    //private static final int CAMERA_REQUEST_CODE = 1001;
    private int type_camera=0;

    private TextToSpeech textToSpeech;
    private GestureDetector gestureDetector; //use for take picture double click
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ProcessCameraProvider processCameraProvider;
    private int isInPreviewMode=1; //flag to previewModel 1: take photo view 0: confirm this picture view
    private File photoFile;
    private ImageHandler imageHandlerFromPath;
    private boolean isConfirm=false;
    private Uri imagePath;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        activityMainBinding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(activityMainBinding.getRoot());
        long startTime = System.currentTimeMillis();

        assetManager = getAssets();
        textureView = findViewById(R.id.textureView);
        Object_box_view = findViewById(R.id.Object_box_view);
        stopCameraButton = findViewById(R.id.stopCameraButton);
        buttonTakePhoto=findViewById(R.id.buttonTakePhoto);
        textToSpeech = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = textToSpeech.setLanguage(Locale.US);
                    if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e("TTS", "Language not supported");
                    }
                } else {
                    Log.e("TTS", "Initialization failed");
                }
            }
        });




        Intent intent = getIntent();
        type_camera = intent.getIntExtra("typecamera", 0);

        requestCameraPermission();


        long elapsedTime = System.currentTimeMillis() - startTime;
        Log.d("StartupTime", "App startup time: " + elapsedTime + " ms");
    }
    // Method to check if GPU support is available and use it if supported
    private boolean checkAndUseGPU() {
        return isGPUSupported();
    }
    // Method to determine if the device supports OpenGL ES 3.0 or higher
    private boolean isGPUSupported() {
        try {
            ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
            return activityManager.getDeviceConfigurationInfo().reqGlEsVersion >= 0x00030000; // OpenGL ES 3.0
        } catch (Exception e) {
            return false;
        }
    }
    // Method to request camera permission from the user
    private void requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        } else {
            startCameraPreview(type_camera);
        }
    }
    // Callback method when the user responds to the camera permission request
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCameraPreview(type_camera);
            } else {
                //startCameraButton.setBackgroundColor(getResources().getColor(R.color.button_default_color));
//                if (toast != null) {
//                    toast.cancel();
//                }
//                Toast.makeText(this, "Camera permission is declined", Toast.LENGTH_SHORT).show();
            }
        }
    }
    //a thread for loading model to speedup
    private void loadYOLOModelInBackground() {
        new Thread(() -> {
            long modelStartTime = System.currentTimeMillis();
            imageHandlerFromProxy = new ImageModel();
            yoloDetector = new yolo8(assetManager, useGPU);

            long modelEndTime = System.currentTimeMillis();
            long modelDuration = modelEndTime - modelStartTime;

//            runOnUiThread(() -> {
//                speak("Start, Take phone, by double-tap the screen, confirm and swap left to ask AI of this photo, cancel to undo");
//
//            });

            Log.d("YOLOModel", "YOLO model loaded in " + modelDuration + " ms");

        }).start();
    }


    /**
     * Captures a photo using the provided ImageCapture instance.
     *
     * @param imageCapture The ImageCapture instance used to take the photo.
     * @param isSave A boolean flag indicating whether to save the captured photo to storage.
     */
    private void takePhoto(ImageCapture imageCapture,boolean isSave) {
        speak("Finish");

        photoFile = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "photo.jpg");
        ImageCapture.OutputFileOptions outputOptions = new ImageCapture.OutputFileOptions.Builder(photoFile).build();

        imageCapture.takePicture(outputOptions, ContextCompat.getMainExecutor(this),
                new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {

                        imageHandlerFromPath = new ImageHandler(photoFile.getAbsolutePath());

                        ImageView photoPreview = findViewById(R.id.photoPreview);
                        photoPreview.setImageBitmap(imageHandlerFromPath.getBitmap());
                        photoPreview.setVisibility(View.VISIBLE);

                        findViewById(R.id.buttonTakePhoto).setVisibility(View.GONE);
                        //findViewById(R.id.startCameraButton).setVisibility(View.GONE);
                        findViewById(R.id.stopCameraButton).setVisibility(View.GONE);

                        findViewById(R.id.confirmButton).setVisibility(View.VISIBLE);
                        findViewById(R.id.cancelButton).setVisibility(View.VISIBLE);
                        isInPreviewMode=0;

                        //check confirm button
                        findViewById(R.id.confirmButton).setOnClickListener(v -> {
                            confirmPhone(isSave);

                            finishPhoto();
                        });

                        findViewById(R.id.cancelButton).setOnClickListener(v -> {
                            isConfirm=false;
                            finishPhoto();
                        });
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException exception) {
                        Log.e("CameraPreview", "Error taking picture: " + exception.getMessage());
                    }
                });
    }

    private void confirmPhone(boolean isSave ){
        speak("confirm");
        Intent resultIntent = new Intent();
        isConfirm=true;
        imageHandlerFromPath.saveBitmapToFile(this,"Bitmap.jpg");
        // Retrieve the URI of the saved image
        imagePath=imageHandlerFromPath.getImageUri();
        //if set up this activity form the chat activity, then return the image url
        if (type_camera == 1) {
            speak("Jump");

            resultIntent.putExtra("photo_uri", imagePath);

            setResult(RESULT_OK, resultIntent);
            finish();
        }


    }



    private void finishPhoto(){
        ImageView photoPreview = findViewById(R.id.photoPreview);
        photoPreview.setVisibility(View.GONE);
        findViewById(R.id.confirmButton).setVisibility(View.GONE);

        findViewById(R.id.cancelButton).setVisibility(View.GONE);


        findViewById(R.id.buttonTakePhoto).setVisibility(View.VISIBLE);
        buttonTakePhoto.setBackgroundColor(getResources().getColor(R.color.button_default_color)); // recover color

        findViewById(R.id.stopCameraButton).setVisibility(View.VISIBLE);
    }
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        return gestureDetector.onTouchEvent(event);
    }
    // refer cameraX :https://github.com/android/camera-samples/
    private void startCameraPreview(int type_camera) {
        long startTime = System.currentTimeMillis();
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                processCameraProvider = cameraProviderFuture.get();
                PreviewView viewFinder = (PreviewView)findViewById(R.id.textureView);
                //select camera
                CameraSelector cameraSelector = new CameraSelector.Builder()
                        .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                        .build();
                int rotation = viewFinder.getDisplay() != null ? viewFinder.getDisplay().getRotation() : Surface.ROTATION_0;

                Preview preview = new Preview.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .setTargetRotation(rotation)
                        .build();
                preview.setSurfaceProvider(viewFinder.getSurfaceProvider());


                //use for saving image
                ImageCapture imageCapture = new ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                        .build();
                processCameraProvider.unbindAll();
                if (type_camera == 0) {
                    // type_camera = 0 ，loading model add analyiss
                    useGPU = checkAndUseGPU();
                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .setTargetRotation(rotation)
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();


                    imageAnalysis.setAnalyzer(detectService, new DetectAnalyzer());


                    processCameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis,imageCapture);


                    loadYOLOModelInBackground();

                    speak("Start, Take phone, by double-tap the screen, confirm and swap left to ask AI of this photo, cancel to undo");
                    long endTime = System.currentTimeMillis();
                    long duration = endTime - startTime;
                    Log.d("CameraPreview", "Camera preview with analysis and YOLO started in " + duration + " ms");
                }else if(type_camera == 1){

                    processCameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture);
                    //
                    long endTime = System.currentTimeMillis();
                    long duration = endTime - startTime;
                    speak("By double-tap the screen");
                    Log.d("CameraPreview", "Camera preview with capture mode started in " + duration + " ms");


                }
                Object_box_view.setTextureViewSize(textureView.getLeft(),textureView.getTop(),textureView.getRight(),textureView.getBottom());

                buttonTakePhoto.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        v.setBackgroundColor(getResources().getColor(R.color.button_pressed_color));
                        takePhoto(imageCapture, false);
                        Log.d("CameraPreview", "Take a picture");
                    }
                });

                //overwrite gesture
                gestureDetector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {

                    @Override
                    public boolean onDoubleTap(MotionEvent e) {
                        handleDoubleTap(imageCapture);
                        return true;
                    }
                    @Override
                    public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {

                        float diffX = e2.getX() - e1.getX();
                        float diffY = e2.getY() - e1.getY();

                        final int SWIPE_THRESHOLD = 100;
                        final int SWIPE_VELOCITY_THRESHOLD = 100;

                        if (Math.abs(diffX) > Math.abs(diffY)) {
                            if (Math.abs(diffX) > SWIPE_THRESHOLD && Math.abs(velocityX) > SWIPE_VELOCITY_THRESHOLD) {
                                if (diffX > 0) {
                                    handleSwipeRight();
                                }
                                //else {
//                                    handleSwipeLeft();
//                                }
                                return true;
                            }
                        }
                        return false;
                    }

                });


                stopCameraButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        v.setBackgroundColor(getResources().getColor(R.color.button_pressed_color));
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
            imageHandlerFromProxy.setImage(image);
            detectBymModel();
        }
    }

    private void detectBymModel() {
        if (detectService == null) {
            return;
        }
        logFPS(imageHandlerFromProxy.getTranbitmap());

        int videoWidth = imageHandlerFromProxy.getWidth();
        int videoHeight = imageHandlerFromProxy.getHeight();
        Log.d("CameraX", "Video resolution: " + videoWidth + "x" + videoHeight);
        boolean draw=detectDraw(imageHandlerFromProxy.getRotatebitmap());
        if (!draw){
            Log.d("DetectionResult", "No detect object no draw ");
        }
        else{
            Log.d("DetectionResult", "Detected objects count ");
        }
        //updateUIWithDetectionResults();
        imageHandlerFromProxy.clear();
    }

    //draw the bounding box
    private boolean detectDraw(Bitmap bitmap){

        List<Object_box> result = yoloDetector.detect(bitmap, 0.6f, 0.6f);

        int videoWidth=bitmap.getWidth();
        int videoHeight=bitmap.getHeight();

        Object_box_view.setDetectionResults(result,videoWidth,videoHeight);
        return true;
    }



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

    void speak(String text) {
        if (textToSpeech != null) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        }
    }

    /**
     * Interface to other activities, send images url
     */
//    private void handleSwipeLeft() {
//
//
//
//        if(isConfirm){
////            imageHandlerFromPath.saveBitmapToFile(this,"Bitmap.jpg");
////            Uri imagePath=imageHandlerFromPath.getImageUri();
//
//            speak("Jump");
//            Intent resultIntent = new Intent(CameraActivity.this, ChatActivity.class);
//            if(imagePath!=null) {
//                resultIntent.putExtra("send_image", true);
//                resultIntent.putExtra("photo_uri", imagePath);
//            }
//            else{
//                resultIntent.putExtra("send_image", false);
//            }
//            startActivity(resultIntent);
//            finish();
//        }
//        else{
//            Intent intent = new Intent(CameraActivity.this, ChatActivity.class);
//            intent.putExtra("send_image", false);
//            startActivity(intent);
//            finish();
//        }
//    }

    // deal the different gesture event: swipe right return previous page
    private void handleSwipeRight() {
        if(isInPreviewMode==0){
            isInPreviewMode=1;
            isConfirm=false;
            finishPhoto();
        }
        else{
            isConfirm=false;
            stopCamera();
        }
    }
    //double tap for take photo or confirm the image
    private void handleDoubleTap(ImageCapture imageCapture){
        if(isInPreviewMode==1) {
            Log.d("Gesture", "Double tap detected");
            buttonTakePhoto.setBackgroundTintList(ColorStateList.valueOf(ContextCompat.getColor(getApplicationContext(), R.color.button_pressed_color)));
            takePhoto(imageCapture, false);
            isInPreviewMode=0;
        }
        else {
            confirmPhone(false);
            finishPhoto();
            isInPreviewMode=1;
        }
    }



    private void stopCameraInstance() {
        speak("return");
        if (processCameraProvider != null) {
            processCameraProvider.unbindAll();  //unbind all the instance
        }
    }

    private void stopCamera() {
        stopCameraInstance();
        finish();
    }
    @Override
    protected void onPause() {
        super.onPause();
        stopCameraInstance();
    }
    protected void onResume() {
        super.onResume();
        if (processCameraProvider == null) {
            startCameraPreview(type_camera);
        }
        else {
            stopCameraInstance();
            startCameraPreview(type_camera);
        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (yoloDetector != null) {
            yoloDetector.release();
            yoloDetector = null;
        }
        if (detectService != null) {
            detectService.shutdown();
            detectService = null;
        }
        if(textToSpeech!=null){
            textToSpeech.stop();
            textToSpeech.shutdown();
        }

        stopCamera();
    }
}