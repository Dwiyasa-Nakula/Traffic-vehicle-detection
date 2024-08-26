# import the necessary packages
import numpy as np
import imutils
import time
import os
import cv2
from input_retrieval import *

# All these classes will be counted as 'vehicles'
list_of_vehicles = ["Mobil", "Sepeda Motor", "Bis", "Truk"]
inputWidth, inputHeight = 416, 416

# Parse command line arguments and extract the values required
LABELS, weightsPath, configPath, inputVideoPath, outputVideoPath, \
    preDefinedConfidence, preDefinedThreshold, USE_GPU = parseCommandLineArguments()

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# PURPOSE: Displays the FPS of the detected video
# PARAMETERS: Start time of the frame, number of frames within the same second
# RETURN: New start time, new number of frames 
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    if current_time > start_time:
        os.system('clear')  # Equivalent of CTRL+L on the terminal
        print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames

# PURPOSE: Draw all the detection boxes with a label and a green dot at the center
# PARAMETERS: Indexes of valid detections, bounding boxes, class IDs, confidences, frame
# RETURN: N/A
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw a green dot in the middle of the box
            cv2.circle(frame, (x + (w // 2), y + (h // 2)), 2, (0, 255, 0), thickness=2)

# PURPOSE: Count vehicles in the current frame by type
# PARAMETERS: Indexes of valid detections, bounding boxes, class IDs
# RETURN: A dictionary with vehicle types and their counts in the current frame
def count_vehicles_per_frame(idxs, boxes, classIDs):
    vehicle_count_dict = {vehicle: 0 for vehicle in list_of_vehicles}

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            label = LABELS[classIDs[i]]
            if label in list_of_vehicles:
                vehicle_count_dict[label] += 1

            # Add Manusia count to Sepeda Motor
            if label == "Manusia":
                vehicle_count_dict["Sepeda Motor"] += 1

    return vehicle_count_dict


# PURPOSE: Display vehicle count per type on the frame
# PARAMETERS: Frame on which the counts are displayed, the count of each vehicle type
# RETURN: N/A
def displayVehicleCountPerType(frame, vehicle_count_dict):
    y_offset = 100
    for vehicle, count in vehicle_count_dict.items():
        text = f'{vehicle}: {count}'
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        y_offset += 20

# PURPOSE: Initializing the video writer with the output video path and the same number of fps, width and height as the source video 
# PARAMETERS: Width of the source video, Height of the source video, the video stream
# RETURN: The initialized video writer
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(outputVideoPath, fourcc, sourceVideofps,
                           (video_width, video_height), True)

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Using GPU if flag is passed
if USE_GPU:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Get the output layer names in the YOLO model
ln = net.getLayerNames()

# The following line needs to be modified to handle the scalar case
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
videoStream = cv2.VideoCapture(inputVideoPath)
if not videoStream.isOpened():
    print(f"Error: Could not open video {inputVideoPath}")
    exit()

video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialization
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())

# loop over frames from the video file stream
while True:
    num_frames += 1
    print("================NEW FRAME================")
    print("FRAME:\t", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], []

    # Calculating fps each second
    start_time, num_frames = displayFPS(start_time, num_frames)
    # read the next frame from the file
    grabbed, frame = videoStream.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start_detection = time.time()
    layerOutputs = net.forward(ln)
    end_detection = time.time()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for i, detection in enumerate(output):
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > preDefinedConfidence:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, preDefinedConfidence,
                            preDefinedThreshold)

    # Draw detection boxes
    drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)

    # Count vehicles per frame
    vehicle_count_dict = count_vehicles_per_frame(idxs, boxes, classIDs)

    # Display vehicle counts per type on the frame
    displayVehicleCountPerType(frame, vehicle_count_dict)

    # display total detected vehicles in the current frame
    total_vehicles = sum(vehicle_count_dict.values())
    cv2.putText(
        frame,
        f'Total Vehicles: {total_vehicles}',
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # write the output frame to disk
    writer.write(frame)

    # Display the frame to the screen
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
videoStream.release()
cv2.destroyAllWindows()
