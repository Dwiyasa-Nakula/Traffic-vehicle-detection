# Yolo Vehicle Counter

## Overview
You Only Look Once (YOLO) is a CNN architecture for performing real-time object detection. The algorithm applies a single neural network to the full image, and then divides the image into regions and predicts bounding boxes and probabilities for each region. 

This project aims to count every vehicle (motorcycle, bus, car, cycle, truck, train) detected in the input video using YOLOv3 object-detection algorithm.

* The pre-trained yolov4 weight file should be downloaded by following these steps:
```
cd yolo-coco
https://github.com/kiyoshiiriemon/yolov4_darknet?tab=readme-ov-file#pre-trained-models
``` 

## Dependencies for using CPU for computations
```
* OpenCV
```
pip3 install opencv-python
```
* Imutils 
```
pip3 install imutils
```
* Scipy
```
pip3 install scipy
```

## Dependencies for using GPU for computations
* Installing GPU appropriate drivers by following Step #2 in the following post:
https://www.pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/
* Installing OpenCV for GPU computations:
Pip installable OpenCV does not support GPU computations for `dnn` module. Therefore, this post walks through installing OpenCV which can leverage the power of a GPU-
https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/
## Usage
* `--input` or `-i` argument requires the path to the input video
* `--output` or `-o` argument requires the path to the output video
* `--yolo` or `-y` argument requires the path to the folder where the configuration file, weights and the coco.names file is stored
* `--confidence` or `-c` is an optional argument which requires a float number between 0 to 1 denoting the minimum confidence of detections. By default, the confidence is 0.5 (50%).
* `--threshold` or `-t` is an optional argument which requires a float number between 0 to 1 denoting the threshold when applying non-maxima suppression. By default, the threshold is 0.3 (30%).
* `--use-gpu` or `-u` is an optional argument which requires 0 or 1 denoting the use of GPU. By default, the CPU is used for computations
```
python yolo_video.py --input <input video path> --output <output video path> --yolo yolo-coco [--confidence <float number between 0 and 1>] [--threshold <float number between 0 and 1>] [--use-gpu 1]
```
Examples: 
* Running with defaults
```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco 
```
* Specifying confidence
```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --confidence 0.3
```
* Using GPU
```
python yolo_video.py --input inputVideos/highway.mp4 --output outputVideos/highwayOut.avi --yolo yolo-coco --use-gpu 1
```
## Implementation details
* The detections are performed on each frame by using YOLOv3 object detection algorithm and displayed on the screen with bounding boxes.
* The detections are filtered to keep all vehicles like motorcycle, bus, car, cycle, truck, train. The reason why trains are also counted is because sometimes, the longer vehicles like a bus, is detected as a train; therefore, the trains are also taken into account.
* The center of each box is taken as a reference point (denoted by a green dot when performing the detections) when track the vehicles.   
* Also, in order to track the vehicles, the shortest distance to the center point is calculated for each vehicle in the last 10 frames. 
* If `shortest distance < max(width, height) / 2`, then the vehicles is not counted in the current frame. Else, the vehicle is counted again. Usually, the direction in which the vehicle moves is bigger than the other one. 
* For example, if a vehicle moves from North to South or South to North, the height of the vehicle is most likely going to be greater than or equal to the width. Therefore, in this case, `height/2` is compared to the shortest distance in the last 10 frames. 
* As YOLO misses a few detections for a few consecutive frames, this issue can be resolved by saving the detections for the last 10 frames and comparing them to the current frame detections when required. The size of the vehicle does not vary too much in 10 frames and has been tested in multiple scenarios; therefore, 10 frames was chosen as an optimal value.

## Reference
* https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

Repo:
* https://github.com/Dwiyasa-Nakula/Traffic-vehicle-detection.git
