import csv
import shutil 
import os
import time

'''
This script filters frames from the coreset saved using the dataset distillation-based coreset construction method from Sotangkur, 2024.
The code can be accessed via: https://github.com/Jack47744/ultrasound_subset.
The prerequisite to run this script is to include the following code into process_video_ultrasound.py:
# Save to CSV
with open("/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_1_percent/frames_metadata_class2.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([args.video_name, ",".join(map(str, top_image_indices))])  # Store indices as comma-separated string

To filter frames in this script, frames_per_video has to be set.
'''

frames_per_video = 1

# # # # # # # # # # # #

start_time = time.time()

# Load from CSV
filtered_frames = []

# set csv file and folder depending on the current class
frames_metadata = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/frames_metadata_class2.csv"
frames_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/all_generated_frames/class_2_finished"
destination_folder = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/distillation_frames_smaller/1_percent"

with open(frames_metadata, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        video_name = row[0]
        indices = list(map(int, row[1].split(",")))  # Convert second row entry back to list of ints
        first_frames = indices[:frames_per_video]  # Get first two indices
        for idx in first_frames:
            filtered_frames.append(f"{video_name}_{idx}.png")

# now I have a list with filtered frames
# print(filtered_frames)
for frame in filtered_frames:
    frame_file_path = os.path.join(frames_folder, frame)
    shutil.copy(frame_file_path, destination_folder)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for whole process: {elapsed_time} seconds")