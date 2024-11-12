package com.example.myapplication;// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.



import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.example.myapplication.databinding.ActivityMainBinding;

import java.util.ArrayList;
import java.util.List;


public class objectResult_view extends View {

    private List<Object_box> currentDetectionResults;
    private List<Object_box> previousDetectionResults;
    private ActivityMainBinding activityMainBinding;
    private final int[] colors;
    private int videoHeight;
    private int videoWidth;
    private float scale;
    private float offset_x;
    private float offset_y;
    private int textureViewLeft;
    private int textureViewTop ;
    private int textureViewRight ;
    private int textureViewBottom;


    // 构造函数：接受 Context
    public objectResult_view(Context context,ActivityMainBinding activityMainBinding) {
        super(context);

        this.colors = new int[] {
                Color.rgb( 54,  67, 244),
                Color.rgb( 99,  30, 233),
                Color.rgb(176,  39, 156),
                Color.rgb(183,  58, 103),
                Color.rgb(181,  81,  63),
                Color.rgb(243, 150,  33),
                Color.rgb(244, 169,   3),
                Color.rgb(212, 188,   0),
                Color.rgb(136, 150,   0),
                Color.rgb( 80, 175,  76),
                Color.rgb( 74, 195, 139),
                Color.rgb( 57, 220, 205),
                Color.rgb( 59, 235, 255),
                Color.rgb(  7, 193, 255),
                Color.rgb(  0, 152, 255),
                Color.rgb( 34,  87, 255),
                Color.rgb( 72,  85, 121),
                Color.rgb(158, 158, 158),
                Color.rgb(139, 125,  96)
        };
        init();
    }

    // 构造函数：接受 Context 和 AttributeSet
    public objectResult_view(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.colors = new int[] {
                Color.rgb( 54,  67, 244),
                Color.rgb( 99,  30, 233),
                Color.rgb(176,  39, 156),
                Color.rgb(183,  58, 103),
                Color.rgb(181,  81,  63),
                Color.rgb(243, 150,  33),
                Color.rgb(244, 169,   3),
                Color.rgb(212, 188,   0),
                Color.rgb(136, 150,   0),
                Color.rgb( 80, 175,  76),
                Color.rgb( 74, 195, 139),
                Color.rgb( 57, 220, 205),
                Color.rgb( 59, 235, 255),
                Color.rgb(  7, 193, 255),
                Color.rgb(  0, 152, 255),
                Color.rgb( 34,  87, 255),
                Color.rgb( 72,  85, 121),
                Color.rgb(158, 158, 158),
                Color.rgb(139, 125,  96)
        };

        init();
    }

    // 初始化方法
    private void init() {

        currentDetectionResults = new ArrayList<>();
        previousDetectionResults=new ArrayList<>();
        this.videoWidth =0;
        this.videoHeight=0;

    }
    public void setTextureViewSize(int left, int top,int Right,int Bottom) {
        this.textureViewLeft = left;
        this.textureViewTop = top;
        this.textureViewRight = Right;
        this.textureViewBottom = Bottom;
    }
    public void updateViewSize(int newWidth, int newHeight) {
        // 获取当前的布局参数
        ViewGroup.LayoutParams layoutParams = getLayoutParams();
        // 更新宽度和高度
        layoutParams.width = newWidth;
        layoutParams.height = newHeight;
        // 应用新的布局参数
        setLayoutParams(layoutParams);
    }
    public void setVideoinfo(int videoWidth,int videoHeight){
        this.videoWidth=videoWidth;
        this.videoHeight=videoHeight;
        setScaleOffset();
    }
    private void setScaleOffset(){
        int textureViewWidth = getWidth();
        int textureViewHeight = getHeight();
        // 计算缩放比例和偏移量
        float scaleX = textureViewWidth / (float) this.videoWidth;
        float scaleY = textureViewHeight / (float) this.videoHeight;
        float scale = Math.min(scaleX, scaleY);
        this.scale=scale;

        float dx = (textureViewWidth - this.videoWidth * scale) / 2;
        float dy = (textureViewHeight - this.videoHeight * scale) / 2;
        this.offset_x=dx;
        this.offset_y=dy;

    }
    private boolean isDetectionResultsEqual(Object_box currentBox) {
        // true for redraw false for noredraw
        for (int j = 0; j < previousDetectionResults.size(); j++) {
            Object_box previousBox=previousDetectionResults.get(j);
            if(currentBox.getLabel()!=previousBox.getLabel()){
                continue;
            }
            if(isBoxPositionChanged(currentBox, previousBox)==true){
                return true;
            }
        }

        return false; // do not find the same result form previous
    }

    private boolean isBoxPositionChanged(Object_box currentBox, Object_box previousBox) {
        float THRESHOLD=5;
        float deltaX = Math.abs(currentBox.x - previousBox.x);
        float deltaY = Math.abs(currentBox.y - previousBox.y);
        float deltaWidth = Math.abs(currentBox.w - previousBox.w);
        float deltaHeight = Math.abs(currentBox.h - previousBox.h);

        // 检查坐标和尺寸变化是否都在阈值范围内
        return deltaX > THRESHOLD || deltaY > THRESHOLD || deltaWidth > THRESHOLD || deltaHeight > THRESHOLD;
    }

    public void setDetectionResults(List<Object_box> objects,int videoWidth,int videoHeight) {

        this.videoWidth=videoWidth;
        this.videoHeight=videoHeight;
        this.previousDetectionResults=currentDetectionResults;
        this.currentDetectionResults.clear();
        this.currentDetectionResults = objects;
        invalidate();  // 重新绘制
    }
    public void clearCanvas() {
        // 清空检测结果
        if (currentDetectionResults != null) {
            currentDetectionResults.clear();
        }
        // 触发重绘以清除画布
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        // 设置用于绘制边框的 Paint
        if (currentDetectionResults == null || currentDetectionResults.isEmpty()) {
            return;
        }
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        // 设置用于绘制文本背景的 Paint
        Paint textbgpaint = new Paint();
        textbgpaint.setColor(Color.WHITE);
        textbgpaint.setStyle(Paint.Style.FILL);

        // 设置用于绘制文本的 Paint
        Paint textpaint = new Paint();
        textpaint.setColor(Color.BLACK);
        textpaint.setTextSize(26);
        textpaint.setTextAlign(Paint.Align.LEFT);
       // setScaleOffset();
        // 遍历 objects，绘制每个对象的边框和标签
        for (int i = 0; i < currentDetectionResults.size(); i++) {
            // set different color
//            if(isDetectionResultsEqual(currentDetectionResults.get(i))==true){
//                continue;
//            }

            paint.setColor(colors[i % colors.length]);

            Object_box obj = currentDetectionResults.get(i);
            float scaleX = (float) (this.textureViewRight - this.textureViewLeft) / this.videoWidth; // X轴缩放比例
            float scaleY = (float) (this.textureViewBottom - this.textureViewTop) / this.videoHeight; // Y轴缩放比例


            float left = obj.x +this.textureViewLeft;
            float top = obj.y +this.textureViewTop;
            float right = obj.w*scaleX+left;
            float bottom = obj.h *scaleY+top;

            // 在 Canvas 上绘制矩形框
            canvas.drawRect(left, top, right, bottom, paint);

//            // 根据缩放比例和偏移量调整坐标
//            float left = rect.left * this.scale + this.offset_x;
//            float top = rect.top * this.scale + this.offset_y;
//            float right = rect.right * this.scale + this.offset_x;
//            float bottom = rect.bottom * this.scale + this.offset_y;
//            // draw rect
//            canvas.drawRect(left, top, right, bottom, paint);

            // lebal and confidence
            String text = obj.getLabel() + " = " + String.format("%.1f", obj.getScore() * 100) + "%";

            // 计算文本宽度和高度
            float text_width = textpaint.measureText(text);
            float text_height = -textpaint.ascent() + textpaint.descent();

            // 计算文本的位置，避免超出图片边界
            float x = left;
            float y = top - text_height;
            if (y < 0)
                y = 0; // 如果超出顶部，将 y 设置为 0
            if (x + text_width > canvas.getWidth())
                x = canvas.getWidth() - text_width; // 如果超出右边界，将 x 设置为最大宽度

            // 绘制文本背景
            canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);

            // 绘制文本
            canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
        }
    }
}
