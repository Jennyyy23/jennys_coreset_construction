import numpy as np
from PIL import Image
import os

def calculate_mean_std_from_multiple_dirs(dirs):
    means = []
    stds = []

    for directory in dirs:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Construct the full path to the image
                img_path = os.path.join(directory, filename)
                
                # Load image
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img).astype(np.float32)

                # Normalize pixel values to the range [0, 1]
                img_array /= 255.0
                
                # Calculate mean and std for each channel
                means.append(img_array.mean(axis=(0, 1)))
                stds.append(img_array.std(axis=(0, 1)))
    
    # Convert lists to numpy arrays
    means = np.array(means)
    stds = np.array(stds)

    # Calculate overall mean and std
    overall_mean = means.mean(axis=0)
    overall_std = stds.mean(axis=0)

    return overall_mean, overall_std

# Execute
# List of directories containing images

# coreset / training set
# path1 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/train_all_frames"
# path1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/training_sets/train_all_frames"
# path1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/train_all_frames"
path1 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/all_frames"
# validationset
# path2 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/val_subset"
# path2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/val_set"
# path2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/val_frames"
path2 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/val"
# testset
# path3 = "/vol/ideadata/ep56inew/ImageNet_VidVRD/subsets/test_subset"
# path3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Dynamic/official_split/test_set"
# path3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-Pediatric/A4C/test_frames"
path3 = "/vol/ideadata/ep56inew/EchoNet/EchoNet-LVH/frame_sets/test"

directories = [path1, path2, path3]
mean, std = calculate_mean_std_from_multiple_dirs(directories)
print('Mean:', mean)
print('Standard Deviation:', std)