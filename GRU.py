import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import re
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

image_width, image_height = 960, 540
split_dataset = True

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available, using CUDA.')
else:
    device = torch.device('cpu')
    print('GPU not available, using CPU.')

print("asd")


def object_detected_percentage(sequence):
    object_detected_count = sum([1 for seq in sequence if seq[-1]])
    percentage = object_detected_count / len(sequence) * 100
    return percentage


def index_to_class_name(index):
    class_names = ["Yellow_tool", "Multimeter", "Tape", "Ninja", "Screwdriver", "Hot_glue"]
    if 0 <= index < len(class_names):
        return class_names[index]
    else:
        raise ValueError(f"Invalid index: {index}. Must be between 0 and {len(class_names) - 1}.")


def class_name_to_index(class_name):
    class_names = ["Yellow_tool", "Multimeter", "Tape", "Ninja", "Screwdriver", "Hot_glue"]
    if class_name in class_names:
        return class_names.index(class_name)
    else:
        raise ValueError(f"Invalid class name: {class_name}. Must be one of {class_names}.")


def get_bounding_box_centroid(bbox, image_width, image_height):
    x_center, y_center = (bbox[1] * image_width) + 2, bbox[2] * image_height
    return x_center, y_center


def interpolate_coordinates(first_pair, second_pair, gain):
    fx = first_pair[0]
    fy = first_pair[1]
    x3 = (1.0 - gain) * float(fx) + gain * float(second_pair[0])
    y3 = (1.0 - gain) * fy + gain * second_pair[1]
    x3 = int(round(x3))
    y3 = int(round(y3))
    return torch.Tensor([x3, y3])


def get_bounding_boxes_for_frame(folder_path, frame_index):
    bbox_dict = {}
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return bbox_dict
    # Look for the label file with the specified frame index at the end of its name
    label_file_path = None
    for file_name in os.listdir(folder_path):
        if re.search(f"_{frame_index}\.txt$", file_name):
            label_file_path = os.path.join(folder_path, file_name)
            break
    if not label_file_path:
        print(f"Error: Label file containing frame index {frame_index} at the end of its name not found.")
        return bbox_dict
    # Read the label file and extract bounding box information
    with open(label_file_path, "r") as label_file:
        for line in label_file:
            line_parts = line.strip().split()
            # Check if the line has the expected number of elements
            if len(line_parts) != 5:
                print(f"Error: Invalid line format in '{label_file_path}'.")
                continue
            class_id, x_center, y_center, width, height = map(float, line_parts)
            class_id = int(class_id)
            if class_id not in bbox_dict:
                bbox_dict[class_id] = []
            bbox_dict[class_id].append((x_center, y_center, width, height))
    return bbox_dict

def get_data(r, train_data, train_labels, labels_path):
    object_of_interest = r[3]["object_of_interest"]
    r = r[2]
    # break
    object_detected = False
    index_of_obj_of_interest = class_name_to_index(object_of_interest)
    fixation_points = []
    video_frame_idx = []
    for e in r:
        print(e.keys())
        fixation_points.append(e["eye_gaze_point:"])
        video_frame_idx.append(e["video_frame_index"])
    bbs = []
    for i in range(len(fixation_points)):
        has_object_of_interest_bb = False
        bb = get_bounding_boxes_for_frame(labels_path,
                                          video_frame_idx[i])
        if index_of_obj_of_interest in bb:
            has_object_of_interest_bb = True
        t = (fixation_points[i], bb, has_object_of_interest_bb)
        bbs.append(t)
    # TODO: get bool / bounding boxes incorporated
    data = torch.tensor(fixation_points, dtype=torch.float32).view(-1, 1, 2)

    seq_length = 10
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        c = 0
        for h in range(i, i + seq_length):
            if bbs[i][2]:
                c = c + 1
                current_bb = bbs[i][1][index_of_obj_of_interest]

        if float(c) / float(seq_length) >= 0.3:
            object_detected = True
        else:
            object_detected = False

        # TODO: make check based on percentage of frames in sequence where object is detected. If > 30 -> object detected
        if not object_detected:
            # add False boolean for whole sequence
            bool_sequence = torch.full((seq_length, 1), False, dtype=torch.bool)
            sequence = torch.cat((sequence.squeeze(), bool_sequence), dim=1)
            train_data.append(sequence)
            # Calculate the average coordinates of the sequence and append it to train_labels
            avg_coordinates = sequence.mean(dim=0)
            avg_coordinates = torch.round(avg_coordinates).to(dtype=torch.int)
            train_labels.append(avg_coordinates[0:2])
        # TODO: get correct bounding box for the object of interest for the sequence: last frame in which bounding box was detected?
        else:
            # add True boolean for whole sequence
            bool_sequence = torch.full((seq_length, 1), True, dtype=torch.bool)
            sequence = torch.cat((sequence.squeeze(), bool_sequence), dim=1)
            train_data.append(sequence)
            # Object bounding box detected
            # get centroid of bounding box:
            avg_coordinates = sequence.mean(dim=0)
            # avg_coordinates = torch.round(avg_coordinates).to(dtype=torch.int)
            bbox = [0, 0.5, 0.5, 0.1, 0.2]  # Example bounding box in YOLOv7 format
            current_bb = list(current_bb[0])
            current_bb.insert(0, index_of_obj_of_interest)
            centroid = get_bounding_box_centroid(current_bb, image_width, image_height)
            adjusted_label = interpolate_coordinates(avg_coordinates.tolist(), centroid, 0.75)
            train_labels.append(adjusted_label)
            print("Average coordinates: ", avg_coordinates)
            print("Centroid:", centroid)
            print("Adjusted label: ", adjusted_label)
    return train_data, train_labels


# Base directory
base_dir = './experiments'
data = []
# Iterate through person directories
for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if os.path.isdir(person_dir) and person.startswith('p'):
        # Iterate through recording directories
        for recording in os.listdir(person_dir):
            recording_dir = os.path.join(person_dir, recording)
            if os.path.isdir(recording_dir) and recording.startswith('rec_id'):

                json_file = os.path.join(recording_dir, 'fixation_points.json')
                meta_file = os.path.join(recording_dir, 'meta.json')

                if os.path.isfile(json_file) and os.path.isfile(meta_file):
                    # Read the fixation points JSON file
                    with open(json_file, 'r') as f:
                        fixation_points_data = json.load(f)

                    # Read the meta JSON file
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)

                    data.append((person, recording, fixation_points_data, meta_data))

                else:
                    if not os.path.isfile(json_file):
                        print(f"fixation_points.json not found in {recording_dir}")
                    if not os.path.isfile(meta_file):
                        print(f"meta.json not found in {recording_dir}")

train_data = []
train_labels = []
# TODO: load this properly if time
for r in data:
    # p1
    if r[0] == 'p1' and r[1] == "rec_id00000":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp22/labels")
    if r[0] == 'p1' and r[1] == "rec_id00001":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp23/labels")
    if r[0] == 'p1' and r[1] == "rec_id00002":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp24/labels")
    if r[0] == 'p1' and r[1] == "rec_id00003":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp25/labels")
    if r[0] == 'p1' and r[1] == "rec_id00004":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp26/labels")
    if r[0] == 'p1' and r[1] == "rec_id00005":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp27/labels")
    if r[0] == 'p1' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp28/labels")
    if r[0] == 'p1' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp29/labels")
    # p2
    if r[0] == 'p2' and r[1] == "rec_id00002":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp2/labels")
    if r[0] == 'p2' and r[1] == "rec_id00003":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp3/labels")
    if r[0] == 'p2' and r[1] == "rec_id00004":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp4/labels")
    if r[0] == 'p2' and r[1] == "rec_id00005":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp5/labels")
    if r[0] == 'p2' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp6/labels")
    if r[0] == 'p2' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp7/labels")
    # p3
    if r[0] == 'p3' and r[1] == "rec_id00003":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp9/labels")
    if r[0] == 'p3' and r[1] == "rec_id00004":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp10/labels")
    if r[0] == 'p3' and r[1] == "rec_id00005":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp11/labels")
    if r[0] == 'p3' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp13/labels")
    if r[0] == 'p3' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp14/labels")
    if r[0] == 'p3' and r[1] == "rec_id00008":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp15/labels")
    # p4
    if r[0] == 'p4' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp16/labels")
    if r[0] == 'p4' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp17/labels")
    if r[0] == 'p4' and r[1] == "rec_id00008":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp18/labels")
    if r[0] == 'p4' and r[1] == "rec_id000010":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp19/labels")
    if r[0] == 'p4' and r[1] == "rec_id00013":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp20/labels")
    if r[0] == 'p4' and r[1] == "rec_id000015":
        train_data, train_labels = get_data(r, train_data, train_labels,
                                            labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp21/labels")
    # p5
    if r[0] == 'p5' and r[1] == "rec_id00005":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp30/labels")
    if r[0] == 'p5' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp31/labels")
    if r[0] == 'p5' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp32/labels")
    if r[0] == 'p5' and r[1] == "rec_id00008":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp33/labels")
    if r[0] == 'p5' and r[1] == "rec_id00009":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp34/labels")
    if r[0] == 'p5' and r[1] == "rec_id00010":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp35/labels")
    # p6
    if r[0] == 'p6' and r[1] == "rec_id00006":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp36/labels")
    if r[0] == 'p6' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp37/labels")
    if r[0] == 'p6' and r[1] == "rec_id00008":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp38/labels")
    if r[0] == 'p6' and r[1] == "rec_id00009":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp39/labels")
    if r[0] == 'p6' and r[1] == "rec_id00014":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp40/labels")
    # p7
    if r[0] == 'p7' and r[1] == "rec_id00007":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp41/labels")
    if r[0] == 'p7' and r[1] == "rec_id00008":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp42/labels")
    if r[0] == 'p7' and r[1] == "rec_id00009":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp43/labels")
    if r[0] == 'p7' and r[1] == "rec_id00011":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp44/labels")
    if r[0] == 'p7' and r[1] == "rec_id00016":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp45/labels")
    # p8
    if r[0] == 'p8' and r[1] == "rec_id00012":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp46/labels")
    if r[0] == 'p8' and r[1] == "rec_id00013":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp47/labels")
    if r[0] == 'p8' and r[1] == "rec_id00014":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp48/labels")
    if r[0] == 'p8' and r[1] == "rec_id00015":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp49/labels")
    if r[0] == 'p8' and r[1] == "rec_id00016":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp50/labels")
    if r[0] == 'p8' and r[1] == "rec_id00017":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp51/labels")
    if r[0] == 'p8' and r[1] == "rec_id00018":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp52/labels")
    if r[0] == 'p8' and r[1] == "rec_id00019":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp53/labels")
    if r[0] == 'p8' and r[1] == "rec_id00020":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp54/labels")
    if r[0] == 'p8' and r[1] == "rec_id00021":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp55/labels")
    if r[0] == 'p8' and r[1] == "rec_id00022":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp56/labels")
    if r[0] == 'p8' and r[1] == "rec_id00023":
        train_data, train_labels = get_data(r, train_data, train_labels, labels_path="/home/decamargo/Documents/uni/yolov7/runs/detect/exp57/labels")


train_data = torch.stack(train_data).squeeze()
train_labels = torch.stack(train_labels).squeeze()

if split_dataset:
    train_data, temp_data, train_labels, temp_labels = train_test_split(train_data, train_labels, test_size=0.3, random_state=42)
    # Split the temporary set into validation and test sets
    validation_data, test_data, validation_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)

print("test")


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
num_epochs = 5000
learning_rate = 0.01

model = GRUModel(input_dim, hidden_dim, output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_data = train_data.to(device)
train_labels = train_labels.to(device)
save_name = "GRU_2layer_256dims_val.pt"

if split_dataset:
    # Assume that validation_data and validation_labels are your validation datasets
    validation_data = validation_data.to(device)
    validation_labels = validation_labels.to(device)

    # Similarly for test_data and test_labels
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    # Initialize the best validation loss to a high value
    best_val_loss = float('inf')
    train_losses = []
    validation_losses = []

    # Training the model
    for epoch in range(num_epochs):
        # Train on training data
        model.train()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Validate on validation data
        if (epoch + 1) % 10 == 0:
            train_losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                val_outputs = model(validation_data)
                val_loss = criterion(val_outputs, validation_labels)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

            validation_losses.append(val_loss.item())
            # If the validation loss is at a new minimum, save the model
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), save_name)
                best_val_loss = val_loss
                print('Model saved!')

    # Load the best model
    model = GRUModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(save_name))

    # Test the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data)
        test_loss = criterion(test_outputs, test_labels)
    print(f'Test Loss: {test_loss.item()}')

    # Plot the training and validation loss over each epoch
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  # Save the figure
    plt.show()  # Display the figure

else:
    # Create the model, loss function, and optimizer

    # Train the model
    for epoch in range(num_epochs):
        outputs = model(train_data.to(device))
        loss = criterion(outputs, train_labels.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    torch.save(model.state_dict(), save_name)

