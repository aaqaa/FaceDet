/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.aaq.facedet;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

import com.aaq.facedet.env.ImageUtils;

import java.nio.ByteBuffer;

public abstract class CameraActivity extends AppCompatActivity
    implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {


  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private boolean debug = false;
  private Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView;
  protected ImageView bottomSheetArrowImageView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView threadsTextView;
  private RadioGroup faceRadioGroup;
  SharedPreferences sharedpreferences;
  private RadioButton faceRadioBtn;
  private RadioButton faceMaskRadioBtn;
  private CheckBox FaceAlarmCheckBox ;
  private CheckBox FaceMaskAlarmCheckBox;
  private CheckBox FaceWithoutMaskAlarmCheckBox;
  private ImageButton SwitchBtn;
  public static boolean toggleCamera = false;
  @Override
  protected void onCreate(final Bundle savedInstanceState) {

    Log.i("facedet","onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    setContentView(R.layout.tfe_od_activity_camera);
//    getSupportActionBar().hide();

    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }

//    threadsTextView = findViewById(R.id.threads);
//    plusImageView = findViewById(R.id.plus);
//    minusImageView = findViewById(R.id.minus);
    apiSwitchCompat = findViewById(R.id.api_info_switch);
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
    faceRadioBtn = findViewById(R.id.faceRadioBtn);
    faceRadioGroup = findViewById(R.id.radioGroup);
    faceMaskRadioBtn = findViewById(R.id.faceMaskRadioBtn);

    FaceAlarmCheckBox = findViewById(R.id.faceAlarm);
    FaceMaskAlarmCheckBox = findViewById(R.id.faceMaskAlarm);
    FaceWithoutMaskAlarmCheckBox = findViewById(R.id.faceWithoutMaskAlarm);

    SwitchBtn = findViewById(R.id
            .switchCamera);
    SwitchBtn.setOnClickListener(view -> {
      CameraActivity.toggleCamera = ! CameraActivity.toggleCamera;

      this.chooseCamera();
      this.setFragment();
    });

    FaceAlarmCheckBox.setOnCheckedChangeListener((compoundButton, b) -> {
      Log.i("face","isFaceOnly : "+ b);
      DetectorActivity.playAlarmFace = b;
    });
    FaceMaskAlarmCheckBox.setOnCheckedChangeListener((compoundButton, b) -> {
      DetectorActivity.playAlarmMask = b;
    });
    FaceWithoutMaskAlarmCheckBox.setOnCheckedChangeListener((compoundButton, b) -> {
      DetectorActivity.playAlarmWithoutMask = b;
    });


    sharedpreferences = getSharedPreferences("MODE", Context.MODE_PRIVATE);
    String MODE = sharedpreferences.getString("MODE",null);
    Log.i("face","isFaceOnly : "+ MODE);
//    MODE = null;
    if(MODE!=null && MODE.equals("FACE")){
      faceRadioGroup.clearCheck();
      faceRadioBtn.setActivated(true);
      faceRadioBtn.setChecked(true);
       DetectorActivity.TF_OD_API_LABELS_FILE = "file:///android_asset/coco_face.txt";
      DetectorActivity.TF_OD_API_MODEL_FILE = "faceDetect.tflite";
    }
    else if(MODE!=null && MODE.equals("FACEMASK")){
      Log.i("face","isFaceOnly : "+ "HAHA FACE IN MASK");

      faceRadioGroup.clearCheck();
      faceMaskRadioBtn.setActivated(true);
      faceMaskRadioBtn.setChecked(true);
      DetectorActivity.TF_OD_API_LABELS_FILE = "file:///android_asset/coco.txt";
      DetectorActivity.TF_OD_API_MODEL_FILE = "faceMask.tflite";
    }
    else{
      faceRadioGroup.clearCheck();
      faceRadioBtn.setActivated(true);
      faceRadioBtn.setChecked(true);
      DetectorActivity.TF_OD_API_LABELS_FILE = "file:///android_asset/coco_face.txt";
      DetectorActivity.TF_OD_API_MODEL_FILE = "faceDetect.tflite";
    }

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
              gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
            } else {
              gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            }
            //                int width = bottomSheetLayout.getMeasuredWidth();
            int height = gestureLayout.getMeasuredHeight();

            sheetBehavior.setPeekHeight(height);
          }
        });
    sheetBehavior.setHideable(false);

    sheetBehavior.addBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            Log.i("FACE","DRAWWE");
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.arrow_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.arrow_up);
                }
                break;
              case BottomSheetBehavior.STATE_DRAGGING:
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.arrow_down);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

//    frameValueTextView = findViewById(R.id.frame_info);
//    cropValueTextView = findViewById(R.id.crop_info);
    inferenceTimeTextView = findViewById(R.id.inference_info);

    apiSwitchCompat.setOnCheckedChangeListener(this);

//    plusImageView.setOnClickListener(this);
//    minusImageView.setOnClickListener(this);

    faceRadioGroup.setOnCheckedChangeListener((radioGroup, checkedId) -> {

      if(checkedId == R.id.faceRadioBtn) {
        Toast.makeText(getApplicationContext(), "Detecting Face",
                Toast.LENGTH_SHORT).show();


        SharedPreferences.Editor editor = sharedpreferences.edit();
        editor.putString("MODE", "FACE");
        editor.commit();


        finish();
        overridePendingTransition(0, 0);
        startActivity(getIntent());
        overridePendingTransition(0, 0);

      } else if(checkedId == R.id.faceMaskRadioBtn) {

        Toast.makeText(getApplicationContext(), "Detecting Masked Face",
                Toast.LENGTH_SHORT).show();
        SharedPreferences.Editor editor = sharedpreferences.edit();
        editor.putString("MODE", "FACEMASK");
        editor.commit();

        finish();
        overridePendingTransition(0, 0);
        startActivity(getIntent());
        overridePendingTransition(0, 0);
      }
    });

  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  /** Callback for android.hardware.Camera API */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      Log.i("facedet","Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      Log.e("facedet", "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
        new Runnable() {
          @Override
          public void run() {
            ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
          }
        };

    postInferenceCallback =
        new Runnable() {
          @Override
          public void run() {
            camera.addCallbackBuffer(bytes);
            isProcessingFrame = false;
          }
        };
    processImage();
  }

  /** Callback for Camera2 API */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
          new Runnable() {
            @Override
            public void run() {
              ImageUtils.convertYUV420ToARGB8888(
                  yuvBytes[0],
                  yuvBytes[1],
                  yuvBytes[2],
                  previewWidth,
                  previewHeight,
                  yRowStride,
                  uvRowStride,
                  uvPixelStride,
                  rgbBytes);
            }
          };

      postInferenceCallback =
          new Runnable() {
            @Override
            public void run() {
              image.close();
              isProcessingFrame = false;
            }
          };

      processImage();
    } catch (final Exception e) {
      Log.e("facedet", "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public synchronized void onStart() {
    Log.d("facedet","onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    Log.d("facedet","onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    Log.d("facedet","onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      Log.e("facedet", "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    Log.d("facedet","onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    Log.d("facedet","onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
          final int requestCode, final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        Toast.makeText(
                com.aaq.facedet.CameraActivity.this,
                "Camera permission is required for this demo",
                Toast.LENGTH_LONG)
            .show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
          CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }

  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for ( String cameraId : manager.getCameraIdList()) {
        String rearCameraId = manager.getCameraIdList()[0];
        String frontCameraId = manager.getCameraIdList()[1];

        if(!CameraActivity.toggleCamera){ cameraId = rearCameraId; }
        else{ cameraId = frontCameraId; }

        Log.i("face","isFace Camera ID : " + cameraId);
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
//        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
//          continue;
//        }

        final StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
            (facing == CameraCharacteristics.LENS_FACING_BACK)
                || isHardwareLevelSupported(
                    characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
        Log.i("facedet","Camera API lv2?: %s"+ useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
//      Log.e("facedet", e);
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();

    Fragment fragment;
    if (useCamera2API) {
      CameraConnectionFragment camera2Fragment =
          CameraConnectionFragment.newInstance(
              new CameraConnectionFragment.ConnectionCallback() {
                @Override
                public void onPreviewSizeChosen(final Size size, final int rotation) {
                  previewHeight = size.getHeight();
                  previewWidth = size.getWidth();
                  com.aaq.facedet.CameraActivity.this.onPreviewSizeChosen(size, rotation);
                }
              },
              this,
              getLayoutId(),
              getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
          new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        Log.d("facedet","Initializing buffer %d at size %d"+ i+ buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @Override
  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
    setUseNNAPI(isChecked);
    if (isChecked) apiSwitchCompat.setText("NNAPI");
    else apiSwitchCompat.setText("TFLITE");
  }

  @Override
  public void onClick(View v) {
//    if (v.getId() == R.id.plus) {
//      String threads = threadsTextView.getText().toString().trim();
//      int numThreads = Integer.parseInt(threads);
//      if (numThreads >= 9) return;

//      numThreads++;
//      threadsTextView.setText(String.valueOf(numThreads));
//      setNumThreads(numThreads);
//    } else if (v.getId() == R.id.minus) {
//      String threads = threadsTextView.getText().toString().trim();
//      int numThreads = Integer.parseInt(threads);
//      if (numThreads == 1) {
//        return;
//      }
//      numThreads--;
//      threadsTextView.setText(String.valueOf(numThreads));
//      setNumThreads(numThreads);
//    }
  }

  protected void showFrameInfo(String frameInfo) {
//    frameValueTextView.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
//    cropValueTextView.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    inferenceTimeTextView.setText(inferenceTime);
  }

  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}
