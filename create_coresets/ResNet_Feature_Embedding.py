import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class ResNet_Feature_Embedding():
    """
    Extracts 512-dimensional feature embeddings from images using a pre-trained ResNet18 model.

    This class loads the ResNet18 model from torchvision, removes the final classification layer,
    and uses the remaining convolutional layers as a feature extractor. It preprocesses input images
    to match the ResNet input requirements and outputs a 512-dimensional feature vector.

    Methods:
        get_feature_embedding(img_path): Returns the feature embedding of an image as a NumPy array of shape (512,1).
    """

    def __init__(self):

        # Check if GPU is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pre-trained ResNet18 model
        self.resnet18 = models.resnet18(pretrained=True)

        # Remove the classification layer (the last fully connected layer)
        # This will give us the feature extraction part
        self.resnet18 = nn.Sequential(*(list(self.resnet18.children())[:-1]))

        # Move the model to the GPU
        self.resnet18.to(self.device)

        # Set the model to evaluation mode
        self.resnet18.eval()

        # preprocess image 
        self.preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization parameters for ResNet18
        ])

    def get_feature_embedding(self, img_path):
        # Load and preprocess the image
        img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)  # Add a batch dimension and move to GPU

        # Extract features
        with torch.no_grad():  # Disable gradient tracking
            features = self.resnet18(img_tensor)

        # Convert to NumPy and transpose to get shape (512, 1)
        features = features.detach() # detach tensor from the computation graph to reduce memory usage
        features_transposed = features.transpose(0,1)  # Transpose the array
        features_squeezed = features_transposed.squeeze(dim=2).squeeze(dim=2) # Squeeze unnecessary dimensions if any
        features_numpy = features_squeezed.cpu().numpy()

        # Convert to NumPy and transpose to get shape (2048, 1)
        # features_numpy = features.detach().cpu().numpy()  # Convert to NumPy and move back to CPU
        # features_transposed = features_numpy.transpose()  # Transpose the array
        # arr_squeezed = features_transposed.squeeze(axis=(0, 1))

        return features_numpy
    
if __name__ == "__main__":

    instance = ResNet_Feature_Embedding()
    vector = instance.get_feature_embedding(img_path="/vol/ideadata/ep56inew/image_subsets/dog_subsets_rgb/ILSVRC2015_train_00005003_tryrgb_03.png")
    print(vector.shape)