import os
import pandas as pd # helps us read the csv file
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class OneClassImagesDataset(Dataset):

    # data loading
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Directory to the csv file with labels (first column: IMAGE name, second colums: labels)
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

        # we probably do not need the following because we access the images through the csv file
        # self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        

    def __len__(self):
        return len(self.annotations)
    

    def __getitem__(self, idx): 
        """
        returns a specific image and the corresponding label to that image
        """
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0]) # self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        
        # assign y_label based on the image[idx] (BUT the labels should be integers!!!)
        y_label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        
        # das ist Standard:
        if self.transform:
            image = self.transform(image)
        
        return image, y_label