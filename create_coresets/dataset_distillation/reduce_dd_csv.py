import csv
import os

'''
This script creates a label-csv file for the frames saved with filter_dd_frames.py.
Take as input a csv file with frame names (first column) and class (second column) without header.
Loop through frame folder and copy the row with the frame to new csv.
Input:
1. Original label CSV path
2. New label CSV path
3. Video path
'''

# Define paths
original_csv_path = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/ED_small_distillation_labels.csv'
new_csv_path = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/ED_small_distillation_labels_1_percent.csv'
frame_folder_path = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/1_percent'

# Read old CSV into a dictionary for fast lookup
frame_dict = {}
with open(original_csv_path, "r") as file1:
    reader = csv.reader(file1)
    for row in reader:
        frame_dict[row[0]] = row  # Using frame name (row[0]) as the key

# List all files in the frame folder
frame_files = set(os.listdir(frame_folder_path))

# Open the new CSV to write
with open(new_csv_path, "w", newline='') as file2:
    writer = csv.writer(file2)

    # Optionally write a header (if your original CSV had one)
    # writer.writerow(['video_name', 'label'])  # You can change column names as needed

    # Loop through the keys in the dictionary and write matching rows
    for frame_name, row in frame_dict.items():
        if frame_name in frame_files:  # Check if the frame is in the folder
            writer.writerow(row)