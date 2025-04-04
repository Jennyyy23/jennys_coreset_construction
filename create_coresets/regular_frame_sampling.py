import os
import time
import cv2
import numpy as np
import matplotlib.image as mpimg
import csv

class SampleFrames():
    """
    Subsamples every i-th frame from videos in a given directory, where `i` is the sampling rate specified by the user.

    Parameters:
        source_path (str): Path to the source directory containing video files.
        destination_path (str): Path where the sampled frames will be saved.
        sample_rate (int): Sampling rate (every i-th frame is selected).
        set_csv (str): Path to the CSV file listing videos and their train/test/validation split.
        set_name (str): Name of the split to use ("train", "validation", or "test").
        set_column (str): Column name in `set_csv` that indicates the data split.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                     CONSTRUCTOR                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self, source_path, destination_path, sample_rate, set_csv, set_name, set_column):

        # initialize time to start measuring
        start_time = time.time()

        # Create the output folder if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        self.source = source_path
        self.destination_path = destination_path
        self.sample_rate = sample_rate
        self.set_csv = set_csv
        self.set_name = set_name
        self.set_column = set_column

        # Check if destination path exists; if not, create it
        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)  # Create the directory

        # count processed videos
        self.count = 0

        # count all videos:
        # Count only the .mp4 and .avi files
        num_video_files = sum(1 for item in os.listdir(source_path) if item.lower().endswith('.mp4') or item.lower().endswith('.avi'))

        # loop through all videos in source location:
        for item in os.listdir(source_path):

            if not item.lower().endswith('.mp4') and not item.lower().endswith('.avi'):
                # Skip non-MP4 and non-avi files and move to the next file
                print(f'The file {item} is not an mp4 or avi video.')
                continue

            # check for EchoNet Pediatric only (uncomment if not necessary)
            if self.check_value(item, self.set_csv, self.set_name) == False:
                # print(f'The file {item} is not a video belonging to the current split.')
                continue

            item_path = os.path.join(source_path, item)

            # process one video
            if os.path.isfile(item_path):
                start_time_one_video = time.time()

                # print information
                self.count+=1
                print(f'Video {self.count} out of {num_video_files}: {item}')

                # split the string for the image name later (based on video name)
                split_string = item.split('.')
                self.image_name = split_string[0]

                # import movie
                movie, FrameRate, self.bgr_frames = self.ImportMovie(source_path + '/' + item)

                movie_shape = movie.shape
                self.frames = movie_shape[0]

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

    # based on train, test and val split in csv
    def check_value(self, item, csv_filename, target_value):
        """This function checks for the target value TRAIN, TEST or VAL"""

        # Remove the ".avi" extension from the item
        item_base = item.replace('.avi', '')
        
        # Open and read the CSV file
        with open(csv_filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            
            # Loop through the rows in the CSV file
            for row in reader:
                # Check if the first column (or relevant column) matches the item_base
                if row[0] == item_base:  # Assuming the item is in the first column
                    # Check if the 7th column contains the target value
                    if row[self.set_column] == target_value:  # 7th column (index 6)
                        return True  # Condition met
        
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
        FrameRate = vidcap.get(cv2.CAP_PROP_FPS)
        # Import first video frame
        success,image = vidcap.read()
        count = 0
        success = True
        movie = []
        bgr_frames = []
        # Import other frames until end of video
        while success:
            bgr_frames.append(image)
            movie.append(self.rgb2gray(image))
            # still save rgb values for later
            success, image = vidcap.read()
            count += 1
        # Convert to array
        movie = np.array(movie)
        # Display some summary information
        # print("==========================================")
        # print("           Video Import Summary           ")
        # print("------------------------------------------")
        # print("   Imported movie: ", fname)
        # print(" Frames extracted: ", count)
        # print("Frames per second: ", FrameRate)
        # print("      data shape = ", movie.shape)
        # print("==========================================")
        return movie, FrameRate, bgr_frames
    
    def save_frames(self):
        """This function saves every i-th image (where i corresponds to sample_rate)"""
        for frame in range(self.frames):
            # Use the modulus operator to check if the current frame correstponds to the sample-rate (e.g. 10th one)
            if frame % self.sample_rate == 0:
                count = "{:02d}".format(frame)
                # bgr
                bgr_image = self.bgr_frames[frame]
                # convert to rgb
                image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                mpimg.imsave(self.destination_path + '/' + self.image_name + '_' + count + '.png', image)

# execute
if __name__ == "__main__":

    # ***************************************************************************************************************
    # EchoNet LVH
    source_paths = ["/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch1",
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch2",
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch3",
                    "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/Batch4"]

    destination_path = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/test"
    sample_rate = 5
    set_csv = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/MeasurementsList_new_clean_ASH_official_split_more_test.csv"
    set_name = "test"
    set_column = 5
    dataset = "EchoNet"

    # ImageNet VidVRD
    # source_paths = ["/vol/ideadata/ep56inew/ImageNet_VidVRD/all_videos"]
    # destination_path = "/vol/ideadata/ep56inew/ImageNet_VidVRD/again_official_split/test_subset"
    # sample_rate = 5
    # set_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/one_class_labels_final_split_removed_classes_VAL_num.csv"
    # set_name = "TEST"
    # set_column = 2

    # EchoNet Pediatric
    # source_paths = ['/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/videos']
    # destination_path = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/val_frames_1'
    # sample_rate = 5
    # set_csv = '/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/FileList_EF_categories.csv'
    # set_name = 'VAL'
    # set_column = 6

    # ***************************************************************************************************************
    total_start_time = time.time()
    for source_path in source_paths:
        SampleFrames(source_path, destination_path, sample_rate, set_csv, set_name, set_column)
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Elapsed time for total process over all Batches: {total_elapsed_time} seconds")

    print(f"EchoNet-LVH {set_name} set frame sampling finished.")