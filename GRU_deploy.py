import torch
import torch.nn as nn
import numpy as np
import json
import cv2
import sys
from server_test import  Server
from pose_estimator import PoseEstimator
import time
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/decamargo/Documents/uni')
sys.path.insert(1, '/home/decamargo/Documents/uni/yolov7')
from yolov7 import utils
from yolov7 import YOLODetector
from system_streamer import Streamer
import math
from PIL import Image


use_video = False
number_of_object_detect_frames = 10
#gru_path = "GRU_trained_2layers.pt"
gru_path = "GRU_2layer_256dims_val.pt"
#gru_path = "GRU_ext_data_2layers.pt"
# Change hidden dims in model to 512 to use last model
#gru_path = "GRU_ext_data_2layers_512hidden_dims.pt"

# Server for communication
server = Server()
# TODO: use the recived robo_image
robo_image = server.get_robo_pose_image(vis=True)

pose_estimator = PoseEstimator()
cap_pose_estimator = cv2.VideoCapture(0)

if not use_video:
    streamer = Streamer()
    img = None
    ret = False
    while True:
        if not ret:
            ret, img, gaze_point = streamer.get_frame_and_gaze()
        else:
            break
    print("Recv img")
    print(gaze_point)
    #cv2.imshow("test", img)
    #cv2.waitKey(0)


# BIG TODOS
# 1. Train GRU with more data
# 2. Create object detector class and add it to the GRU deploy DONE
# 3. Translate GRU output into object -> Object into x,y of robot cam DONE
# 4. Make streaming work with HOLOLENS and ubuntu code DONE
# 5. Add networking stuff to send and recv data DONE


# HOW TO RUN:
# 1. Enable device portal on HOLOLENS and run hl2ss app
# 2. use IP shown in top right corner and adapt ip in system_streamer.Streamer
# 3. Adapt server ip/port in test_server.Server and client_test.Client (on remote device)
# 4. Run this script and then the client script on the remote machine

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


# Set hyperparameters
input_dim = 3
hidden_dim = 256
output_dim = 2
print("Load GRU")
# Load the trained model
model = GRUModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(gru_path))
model.eval()
print("Init YOLO")
# Init Object detector
detector = YOLODetector.YOLODetector(weights='best.pt', imgsz=640)
print("Detect on robot image")
# Detection on robot camera
#img_path = '/home/decamargo/Documents/uni/yolov7/test_orig/images/valid_image_0000.jpg'
#robo_image = cv2.imread(img_path)
#cv2.imshow('Frame', robo_image)
#cv2.waitKey(0)
cv2.circle(robo_image, (167, 246), 5, (255, 255, 255), -1)
img, img0 = detector.prepare_image_robo_cam(robo_image)
# Detect bounding boxes in the image
rob_bounding_boxes, rob_labels = detector.detect(img, img0, vis=False)
print("Detected objects: ", rob_labels)
rob_object_centroids = {}
for b, l in zip(rob_bounding_boxes, rob_labels):
    rob_object_centroids[l] = (b[1], b[2])

# Read video using OpenCV
if use_video:
    video_path = 'experiments/p2/rec_id00004/recording4.mp4'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Open JSON file containing gaze data
    json_path = 'experiments/p2/rec_id00004/fixation_points.json'
    with open(json_path, 'r') as json_file:
        gaze_data = json.load(json_file)

# Initialize variables for input tensor
sample_data = []
sequence_length = 10

# Iterate over frames in the video
frame_index = 0
frame_boxes = []
frame_labels = []
# while cap.isOpened():
while True:
    if use_video:
        ret, frame = cap.read()
        gaze_info = next((g for g in gaze_data if g['video_frame_index'] == frame_index), None)
        if gaze_info is not None:
            eye_gaze_point = gaze_info['eye_gaze_point:']
            corresponding_frame_index = gaze_info['video_frame_index']
            print(f"Frame Index: {frame_index}, Gaze Point: {eye_gaze_point}, Frame index: {corresponding_frame_index}")
            x, y = eye_gaze_point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            # Add gaze point to sample_data
            sample_data.append(eye_gaze_point)
        else:
            continue
        if not ret:
            break
    else:
        ret, frame, eye_gaze_point = streamer.get_frame_and_gaze()
        if not gaze_point:
            break
        sample_data.append(eye_gaze_point)

    # Create input tensor for the model with the last 10 gaze points and last column as 0
    if len(sample_data) > sequence_length:
        sample_data = sample_data[-sequence_length:]  # Keep only the last sequence_length gaze points
        last_column_value = 0  # Default value of the last column
        # TODO if objects detected in current frame -> set true
        # Object Detector
        img, img0 = detector.prepare_image(frame)
        # Detect bounding boxes in the image
        bounding_boxes, labels = detector.detect(img, img0, vis=False)
        # Print the bounding boxes
        for box, label in zip(bounding_boxes, labels):
         #   print("Object: " + str(label) + " | " + str(box))
            pass
        if bounding_boxes:
         #   print("Object Detected")
            last_column_value = 1  # Set value of the last column based on the condition
        frame_labels.append(labels)
        frame_boxes.append(bounding_boxes)
        if len(frame_boxes) >= number_of_object_detect_frames:
            frame_boxes.pop(0)
            frame_labels.pop(0)
        input_tensor = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0)
        last_column_tensor = torch.full((1, sequence_length, 1), last_column_value, dtype=torch.float32)
        input_tensor = torch.cat((input_tensor, last_column_tensor), dim=2)
        # print("Input Tensor:", input_tensor)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        # print("Output:", output)
        output_point = (int(output[0][0]), int(output[0][1]))
        cv2.circle(frame, output_point, 5, (0, 255, 0), -1)
        if bounding_boxes:
            j = 0
            min = 10000
            min_ind_frame_boxes = 0
            min_ind_boxes = 0
            #print("Output point: " + str(output_point))
            # TODO incorporate bounding boxes of last 5 frames
            for f_b in frame_boxes:
                i = 0
                for b in f_b:
                    bb_center = (int(b[1] * frame.shape[1]), int(b[2] * frame.shape[0]))
                    cv2.circle(frame, bb_center, 5, (255, 255, 255), -1)
                    dist = calculate_distance(bb_center, output_point)
                    # print("Bounding box: " + str(frame_labels[j][i]) + " | Dist: " + str(dist) + " | Center: " + str(bb_center))
                    if dist < min:
                        min = dist
                        min_ind_boxes = i
                        min_ind_frame_boxes = j
                    i = i + 1
                j = j + 1
            intention_object = frame_labels[min_ind_frame_boxes][min_ind_boxes]
            print("Closest Bounding box: " + str(frame_labels[min_ind_frame_boxes][min_ind_boxes]) + " | Dist: " + str(
                min))
            if not intention_object in rob_object_centroids:
                print("Object not found on robot camera. Skipping.")
                continue
            rob_obj_centroid = (int(rob_object_centroids[intention_object][0] * 640), int(rob_object_centroids[intention_object][1] * 480))
            #print("Object centroid from robot: ", rob_obj_centroid)
            # TODO: insert pose estimation here
            ret, frame = cap_pose_estimator.read()
            if not ret:
                print("Can't receive frame from Pose Estimator. Exiting ...")
                break
            is_intention = pose_estimator.estimate_pose(frame)
            # TODO: Send data here: GazeX, GazeY, IsIntentionBool
            server.send_data(rob_obj_centroid[0], rob_obj_centroid[1], is_intention)
    # Display the frame
    #cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()

# Sample data for inference
# sample_data = torch.rand(10, 1, input_dim-1) * 100 + 200
# sample_data = torch.cat((sample_data, torch.zeros(10, 1, 1)), dim=2)
# print("Input:", sample_data)
# Run inference
# with torch.no_grad():
#    output = model(sample_data)
# print("Output:", output)
