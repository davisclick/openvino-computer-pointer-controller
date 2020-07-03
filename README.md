# Computer Pointer Controller

In this project, it is use a gaze detection model to control the mouse pointer of the computer.
It is use the InferenceEngine API from Intel's OpenVino ToolKit. The gaze estimation model requires three inputs:
 - The head pose
 - The left eye image
 - The right eye image.
 
The flow of data used is:

## Project Set Up and Installation

#### Step 1
You need to install openvino successfully.
See this guide for installing [OpenVino ToolKit](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html).

#### Step 2
Clone the repository from this URL: https://github.com/davisclick/openvino-eye-gaze-estimation

#### Step 3
After you clone the repo, you need to install the dependecies

	pip3 install -r requirements.txt

#### Step 4
Create Virtual Enviorment in working directory.

 	python3 -m venv venv

#### Step 5
Activate Virtual Enviorment

 	source venv/bin/activate

#### Step 6
Initialize the openVINO environment:-

	source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

#### Step 7
Download the following models by using openVINO model downloader:

	cd models
 ####Face Detection Model
	```
 	python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-0001"
	```
####Facial Landmarks Detection Model
	```
	python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"
	```
####Head Pose Estimation Model
	```
 	python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"
	```
####Gaze Estimation Model
	```
	python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
	```

## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation

### Used Models

1. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
2. [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
3. [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
4. [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### Command Line Arguments for Running the app

Argument|Type|Description
| ------------- | ------------- | -------------
-fd | Required | Path to .xml file of Face Detection model.
-fl | Required | Path to .xml file of Facial Landmark Detection model.
-hp| Required | Path to .xml file of Head Pose Estimation model.
-ge| Required | Path to .xml file of Gaze Estimation model.
-i| Required | Specify the path of input video file or enter cam for taking input video from webcam.
-l| Optional | 
-d | Optional | Provide the target device: CPU / GPU / MYRIAD / FPGA
-pt  | Optional | Probability threshold for detections filtering.
-pof | Optional | Specify the flags from fd, fl, hp, ge if you want to visualize the output of corresponding models of each frame (write flags with space seperation. Ex:- -flags fd fld hp).

 ### Directory Structure of the project
  ```bash
computer-pointer-controller  
|
|--media
|   |--demo.mp4
|--models
|   |--face-detection-adas-binary-0001
|   |--gaze-estimation-adas-0002
|   |--head-pose-estimation-adas-0001
|   |--landmarks-regression-retail-0009
|--README.md
|--requirements.txt
|--src
    |--face_detection.py
    |--facial_landmarks_detection.py
    |--gaze_estimation.py
    |--head_pose_estimation.py
    |--input_feeder.py
    |--main.py
    |--mouse_controller.py
```
- <b>media</b> Folder with the media files
- <b>models</b> Folder with the pre-trained models from Open Model Zoo
    - intel
        1. face-detection-adas-binary-0001
        2. gaze-estimation-adas-0002
        3. head-pose-estimation-adas-0001
        4. landmarks-regression-retail-0009
- <b>src</b> Folder with the python files of the app
    + [face_detection.py](./src/face_detection.py) : Face Detection related inference code
    + [facial_landmarks_detection.py](./src/facial_landmarks_detection.py) : Take the deteted face as input, preprocessed it, perform inference on it and detect the eye landmarks, postprocess the outputs.
    + [gaze_estimation.py](./src/gaze_estimation.py) : Gaze Estimation related inference code
    + [head_pose_estimation.py](./src/head_pose_estimation.py) : Head Pose Estimation related inference code
    + [input_feeder.py](./src/input_feeder.py) : Contains InputFeeder class which initialize VideoCapture as per the user argument and return the frames one by one.
	+ [main.py](./src/driver.py) : Main script to run the app
    + [mouse_controller.py](./src/mouse_controller.py) : Contains MouseController class which take x, y coordinates value, speed, precisions and according these values it moves the mouse pointer by using pyautogui library.
    + [profiling.py](./src/profiling.py) : To check performance of script line by line
    
- <b>.gitignore</b> Listing of files that should not be uploaded to GitHub
- <b>README.md</b> File that you are reading right now.
- <b>requirements.txt</b> All the dependencies of the project listed here

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
