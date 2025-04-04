import os
import random
import shutil
import time

def select_and_copy_images(png_directory, num_images, destination_directory):
    """
    Selects a random number of images from a source directory and copies them to a destination directory.

    :param png_directory: Path to the directory containing PNG images.
    :param num_images: Number of random images to select.
    :param destination_directory: Directory where the selected images will be copied.
    """
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Get a list of all .png files in the directory
    all_png_files = [f for f in os.listdir(png_directory) if f.lower().endswith('.png')]
    print(f"all png files: {len(all_png_files)}")

    # Check if there are enough images to select from
    if len(all_png_files) < num_images:
        raise ValueError(f"Not enough images in {png_directory}. Found {len(all_png_files)}, but need {num_images}.")

    # Randomly select the specified number of images
    selected_images = random.sample(all_png_files, num_images)

    # Copy each selected image to the destination directory
    for image in selected_images:
        src_path = os.path.join(png_directory, image)
        dst_path = os.path.join(destination_directory, image)
        shutil.copy(src_path, dst_path)

    print(f"Copied {num_images} images to {destination_directory}.")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EchoNet-Dynamic
png_directory = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames"
num_images = 120251
destination_directory = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/random_reduced_10percent"

# EchoNet-Pediatric
# png_directory = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_all_frames"
# num_images = 22175
# destination_directory = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/random/train_random_frames_DB2reduced"

# Execute
start = time.time()
select_and_copy_images(png_directory, num_images, destination_directory)
end = time.time()
total_time = end -start
print(f"Elapsed time for random subset creation: {total_time} seconds")