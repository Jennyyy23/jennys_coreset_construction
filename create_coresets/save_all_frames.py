import os
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
import random
import csv

class SaveAllFrames():
    """ 
    This class saves all frames as png from all videos of a given directory.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                     CONSTRUCTOR                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self, source_path, destination_path, set_csv, set, set_column, dataset):

        # initialize time to start measuring
        start_time = time.time()

        self.source = source_path
        self.destination_path = destination_path
        self.set_csv = set_csv
        self.set = set # either "TRAIN", "TEST", or "VAL"
        self.set_column = set_column
        self.dataset = dataset

        # Create the output folder if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # count processed videos
        self.count = 0

        # count all videos:
        # Count only the .mp4 and .avi files
        num_video_files = sum(1 for item in os.listdir(source_path) if item.lower().endswith('.mp4') or item.lower().endswith('.avi'))

        # loop through all videos in source location:
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)

            if not item.lower().endswith('.mp4') and not item.lower().endswith('.avi'):
                # Skip non-MP4 and non-avi files and move to the next file
                print(f'The file {item} is not an mp4 or avi video.')
                continue

            # check for EchoNet Pediatric and LVH only (uncomment if not necessary)
            if self.check_value(item, set_csv, set) == False:
                # print(f'The file {item} is not a video belonging to the current split.')
                continue

            # process one video
            if os.path.isfile(item_path):

                start_time_one_video = time.time()

                # print information
                self.count+=1
                print(f'Video {self.count} out of {num_video_files}: {item}')

                # split the string for the image name later (based on video name)
                split_string = item.split('.') # split off the .mp4 or .avi ending
                self.image_name = split_string[0]

                # import movie
                self.bgr_frames = self.ImportMovie(source_path + '/' + item)

                # save frames (regular distances)
                self.save_frames()

                # print time for video processing
                end_time_one_video = time.time()
                elapsed_time_one_video = end_time_one_video - start_time_one_video
                print(f"Elapsed time per video processing: {elapsed_time_one_video} seconds")

            elif os.path.isdir(item_path):
                print(f'Directory: {item}')

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for whole process: {elapsed_time} seconds")

                 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                   ALL MY METHODS                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # based on train, test and val split in csv for EchoNet Pediatric
    def check_value(self, item, csv_filename, target_value):
        """This function checks for the target value TRAIN, TEST or VAL"""

        # Remove the ".avi" extension from the item
        item_base = item.replace('.avi', '')
        
        # Open and read the CSV file
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)

            if self.dataset == "EchoNet":
                # Skip the header row
                next(reader)  
            
            # Loop through the rows in the CSV file
            for row in reader:
                # Check if the first column (or relevant column) matches the item_base
                if row[0] == item_base:  # Assuming the item is in the first column
                    # Check if column contains the target value (TRAIN, TEST or VAL)
                    if row[self.set_column] == target_value:
                        return True  # Condition met
                    else:
                        return False
        
        return False  # Condition not met

    def rgb2gray(self, rgb):
        """This function converts RGB images to grayscale"""
        # we need greyscale images to have only one channel
        # such that a video can be represented in a 2D matrix istead of 3 x 2D
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def ImportMovie(self, fname):
        """Main movie import function."""
        # Open video
        vidcap = cv2.VideoCapture(fname)
        # ATTENTION! OpenCV uses BGR format. Needs to be converted back later before saving image.
        # Import first video frame
        success,image = vidcap.read()
        success = True
        bgr_frames = []
        # Import other frames until end of video
        while success:
            bgr_frames.append(image)
            success, image = vidcap.read()
        return bgr_frames
    
    def save_frames(self):
        """This function saves all frames from the video"""

        for idx, frame in enumerate(self.bgr_frames):
            count = "{:02d}".format(idx)
            bgr_image = frame
            # convert to rgb
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            mpimg.imsave(os.path.join(self.destination_path, f"{self.image_name}_{count}.png"), image)

if __name__ == "__main__":

    print("Main block in save_all_frames.py executed!")

    # *********************************************************************************************************************

    # execute (ImageNet VidVRD)
    # source_paths = ["/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos"]
    # destination_path = "/vol/ideadata/ep56inew/ImageNet_VidVRD/again_official_split/train_all_frames"
    # set_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv"
    # set = "TRAIN"
    # set_column = 2
    # dataset = "ImageNet"

    # execute (EchoNet-Pediatric-A4C)
    # source_paths = ['/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/videos']
    # destination_path = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_all_frames_1'
    # set_csv = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv'
    # set = 'TRAIN'
    # set_column = 6
    # dataset = 'EchoNet'

    # execute (EchoNet-Dynamic)
    # SaveAllFrames(source_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/Videos", \
    #               destination_path="/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames")

    # execute (EchoNet-LVH)
    source_paths = ["/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch1", 
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch2",
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch3",
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch4"]

    destination_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/all_frames"
    set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv"
    set = "train"
    set_column = 5
    dataset = 'EchoNet'
    
    # *********************************************************************************************************************

    # total_start_time = time.time()
    for source_path in source_paths:
        SaveAllFrames(source_path, destination_path, set_csv, set, set_column, dataset)
        print(f"completed for {source_path}")
    # total_end_time = time.time()
    # total_elapsed_time = total_end_time - total_start_time
    # print(f"Elapsed time for total process over all Batches: {total_elapsed_time} seconds")
    