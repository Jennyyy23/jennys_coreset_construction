import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torchvision import models
import math
import logging
from customDataset import OneClassImagesDataset
import pandas as pd
import numpy as np
import wandb
import os
import time 
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from torch.utils.data import Subset
# Basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# measure training time
start_time_preparation = time.time()

"""
This script trains a ResNet18 classification model on a specified image dataset.
Right now, it is implemented for classification tasks on ImageNet-VidVRD, EchoNet-Dynamic, -Pediatric, and -LVH.
In order to do that:
0. Specify dataset (IN or ED or EP or LVH)
1. Choose a run name
2. Specify label csv file path (first column: file name, second column: numeric label)
3. Specify train_labels_csv (if you have several csv_paths)

For a new dataset apart from ImageNet and EchoNet-Dynamic:
4. Specify regression or classification             
5. Set normalization parameters                 
6. Set other configuration of training hyperparameters

Recommendation: start via bash_script_for_training.sh
2do in Bash Script:
1. Specify job Name                               
2. specify time                                     
3. Specify log-file names 2x       
4. Specify data directories (load into $TMPDIR)
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ADAPT THESE PARAMETERS BEFORE STARTING 
dataset = "LVH" # IN or ED or EP or LVH
run = "LVH_all_long"
epochs = 50
#
logging.info(f"run: {run}")
#
# CLASSIFICATION OR REGRESSION
# if classification, set TRUE, if regression, set FALSE
classification = True
#
# early stopping parameters
# Do you want to use early stopping? (set to True or False)
early_stopping = True
patience = 15
# 
# **************************************************
# BEST HYPERPARAMETER CONFIGURATION:
if dataset == "IN":
    my_config = {
        'epochs': epochs ,
        'optimizer': 'sgd',
        'learning_rate': 0.05, 
        'momentum': 0.63,
        'batch_size': 64
        }
    
# misunderstood sweep
elif dataset == "EP":
    my_config = {
        'epochs': epochs ,
        'optimizer': 'sgd',
        'learning_rate': 0.00808037529342434, 
        'momentum': 0.7457167494889976,
        'batch_size': 64
        }
    
elif dataset == "LVH":
    my_config = {
    'epochs': epochs ,
    'optimizer': 'sgd',
    'learning_rate': 0.0240290367740568, 
    'momentum': 0.4989781679753936,
    'batch_size': 64
    }  

# best configuration for EchoNet-Dynamic EF Class Prediction
elif dataset == "ED":
    my_config = {
        'epochs': epochs ,
        'optimizer': 'sgd',
        'learning_rate': 0.025238401009823302, 
        'momentum': 0.17241839844188864,
        'batch_size': 64
        }

else:
    raise ValueError("Dataset not defined. Choose IN or ED.")
# **************************************************

# * * * * * * * DATA SETS * * * * * * * * * *
# Do not forget to set normalization parameters.
#
# access datasets via tmpdir in bash script:
tmpdir = os.getenv('TMPDIR')
val_set_dir = tmpdir + '/validationset'
test_set_dir = tmpdir + '/testset'
train_dir = tmpdir + '/trainingset'

# EchoNet-Dynamic EF Class Prediction (official split)
if dataset == "ED":
    no_of_classes = 3
    val_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/all_random_val_test/ED_cat_val.csv'
    test_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/all_random_val_test/ED_cat_test.csv'
    train_all_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/all_random_val_test/ED_cat_all_frames.csv'
    train_random_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/all_random_val_test/ED_cat_random.csv'
    # ED POD DB1
    train_wasserstein_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/ED_wasserstein.csv'
    train_euclid_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/ED_euclid.csv'
    train_js_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/ED_js.csv'
    train_spherical_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/ED_spherical.csv'
    train_tvd_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/ED_tvd.csv'
    train_distillation_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/distillation_labels.csv'
    # best
    train_reduced_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/labels_FE_DB2/wasserstein_reduced.csv'
    # random
    train_random_distill_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/random_reduced_distill.csv'
    train_random_wasser_csv = '/home/woody/iwai/iwai110h/EchoNet-Dynamic/labels/EF_labels_classification/official_split_classification_labels/random_reduced_10percent.csv'

# EchoNet-Pediatric EF Class Prediction
elif dataset == "EP":
    no_of_classes = 3
    val_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/val_frames_labels.csv'
    test_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/test_frames_labels.csv'
    train_all_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/all_frames_labels.csv'
    train_random_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/train_random_1_labels.csv'
    # FE DB1 tvd
    train_tvd_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/FE_DB1_tvd_labels.csv'
    # FE DB2 wasserstein
    train_wasserstein_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/FE_DB2_wasserstein_labels.csv'
    # distillation
    train_distillation_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/EP_distillation_labels.csv'
    # best
    train_reduced_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/train_FE_DB2_wasserstein_reduced_labels.csv'
    # random 
    train_random_distill_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/train_random_frames_distill.csv'
    train_random_wasser_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/labels/train_random_frames_DB2reduced.csv'

# ImageNet-VidVRD
elif dataset == "IN":
    no_of_classes = 27
    val_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/test_val_all_random/IN_val.csv'
    test_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/test_val_all_random/IN_test.csv'
    train_all_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/test_val_all_random/IN_train_all_frames.csv'
    train_random_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/test_val_all_random/IN_train_random.csv'
    # IN FE_KMedoids
    train_wasserstein_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/FE_KMedoids/IN_wasserstein.csv'
    train_euclid_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/FE_KMedoids/IN_euclid.csv'
    train_js_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/FE_KMedoids/IN_js.csv'
    train_spherical_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/FE_KMedoids/IN_spherical.csv'
    train_tvd_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/FE_KMedoids/IN_tvd.csv'
    train_distillation_csv = '/home/woody/iwai/iwai110h/ImageNetVidVRD/official_split/labels/distillation_labels.csv'

# EchoNet-LVH
elif dataset == "LVH":
    no_of_classes = 2
    val_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_val.csv'
    test_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_test.csv'
    train_all_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_all_frames.csv'
    train_random_1_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_random_jenny_cs.csv'
    train_random_2_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_random_dd.csv'
    # FE DB2 wasserstein
    train_jenny_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_jennys_coreset.csv'
    train_dd_csv = '/home/woody/iwai/iwai110h/EchoNet-LVH/labels/LVH_distillation_labels.csv'

train_labels_csv = train_all_csv

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# EchoNet-Pediatric Gender
# no_of_classes = 2
# val_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_val_num.csv'
# test_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_test_num.csv'
# train_all_frames_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_train_all_frames_num.csv'
# train_jennys_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_train_jenny_num.csv'
# train_random_ss_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_random_num.csv'
# train_euclid_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_train_euclid_num.csv'
# train_js_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_train_js_num.csv'
# train_spherical_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_gender_num/numerical/EP_labels_gender_train_spherical_num.csv'
# 
# EchoNet-Pediatric Age (all ages) (MUSS NOCHMAL GEÄNDERT WERDEN, ANDERE ORDNERSTRUKTUR)
# no_of_classes = 19
# val_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_labels_age_val.csv'
# test_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_test_labels_age.csv'
# train_all_frames_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_all_frames_labels_age.csv'
# train_jennys_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_wasserstein_labels_age.csv'
# train_random_ss_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_random_age.csv'
# train_euclid_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_euclid_labels_age.csv'
# train_js_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_js_labels_age.csv'
# train_spherical_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/frame_labels_age/EP_train_spherical_labels_age.csv'
# 
# EchoNet-Pediatric Age (age categories)
# no_of_classes = 3
# val_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_val_age_cat.csv'
# test_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_test_labels_age_cat.csv'
# train_all_frames_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_train_all_frames_age_cat.csv'
# train_wasserstein_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_train_wasserstein_age_cat.csv'
# train_random_ss_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_random_age_cat.csv'
# train_euclid_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_train_euclid_age_cat.csv'
# train_js_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_train_js_age_cat.csv'
# train_spherical_cs_csv = '/home/woody/iwai/iwai110h/EchoNet-Pediatric/A4C/frame_labels_age/age_3_categories/EP_train_spherical_age_cat.csv'
# # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

os.environ["WANDB_MODE"] = "dryrun"

# save metrics for each run to csv (since wandb does not work) 
# ATTENTION! only implemented for classification so far. 
def save_to_csv(run, hyperparameters, test_accuracy, training_losses, validation_accuracies, validation_losses):

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Training Loss': training_losses,
        'Validation Accuracy': validation_accuracies,
        'Validation Loss': validation_losses
    })

    # Create a DataFrame for hyperparameters
    hyperparams_df = pd.DataFrame([hyperparameters])

    # Create a DataFrame for test accuracy
    test_accuracy_df = pd.DataFrame({'Test Accuracy': [test_accuracy]})
    
    # Write everything to CSV
    with open(f'/home/hpc/iwai/iwai110h/myCode/master/csv_logs/ED_classification_official/run_{run}.csv', 'w', newline='') as file:
        # Write hyperparameters
        hyperparams_df.to_csv(file, index=False, header=True)

        # Write test accuracy
        file.write('\n')# Write an empty row for separation
        test_accuracy_df.to_csv(file, index=False, header=True)
        
        # Write metrics
        file.write('\n')
        metrics_df.to_csv(file, index=False, header=True)

# # # # # OFFICIAL TRAINING LOOP # # # # # 
def train(config, run, hyperparameters):

    logging.info(f"RUN: {run}")

    saved_model = "resnet18_trained_" + run + ".pth"

    # Initialize lists to store metrics
    training_losses = []
    validation_accuracies = []
    validation_losses = []

    logging.info("NEW TRAINING STARTED")
    # PLEASE SET THE FOLLOWING PARAMETERS:
    # How many epochs do you want to train (maximally)? --> set in wandb config now
    # epochs = 200

    # ensure that the random numbers generated by PyTorch are deterministic
    torch.manual_seed(42)

    # **************************************************************************************************************************
    # *** WANDB ***
    # wandb.login()
    # start a new wandb run to track this script
    wandb.init(
        project=run,
        reinit=True,
        mode="offline",
        config=config,
        # sync_tensorboard=False,  # Prevent W&B from trying to sync data
        # settings={"start_method": "fork"}  # Fork mode to ensure no background processes try to sync

    )
    # Access hyperparameters using wandb.config
    config = wandb.config
    # just for debugging wandb
    # exit()

    # # # HYPERPARAMETER SETTING # # #
    # Set hyperparameters using wandb
    batch_size = config.batch_size
    optimizer_name = config.optimizer
    learning_rate = config.learning_rate
    epochs = config.epochs
    momentum = config.momentum
    # # # # # # # # # # # # # # # # # 

    logging.info(f"NEW TRAINING: Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Momentum: {momentum}")

    # **************************************************************************************************************************

    # pre-trained models expect a certain data format (https://pytorch.org/hub/pytorch_vision_resnet/)
    # ImageNet VidVRD dataset
    if dataset == "IN":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # taken from the pre-trained model
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # wie bei Wutikorn 
            transforms.Normalize(mean=[0.458, 0.462, 0.406], std=[0.214, 0.212, 0.208])
        ])

    elif dataset == "ED":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.128, 0.129, 0.130], std=[0.190, 0.191, 0.192])
        ])

    elif dataset == "EP":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.105, 0.098, 0.112], std=[0.168, 0.161, 0.175])
        ])

    elif dataset == "LVH":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.075, 0.075, 0.74], std=[0.160, 0.160, 0.159])
        ])

    else:
        raise ValueError("Dataset not defined. Choose IN or ED or EP.")

    # Load Data (how shall I do that? I first have to write my own Dataset Class (inherit from Dataset))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # examplary data (insert your customDataset Class here!!!)
    # this is the standard way to do
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #
    # my dataset:
    trainset = OneClassImagesDataset(csv_file = train_labels_csv,
                                    root_dir = train_dir,
                                    transform = transform)

    valset = OneClassImagesDataset(csv_file = val_csv,
                                    root_dir = val_set_dir,
                                    transform = transform)

    testset = OneClassImagesDataset(csv_file = test_csv,
                                    root_dir = test_set_dir,
                                    transform = transform)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # * # * # * # * # * # * # * # * # *
    # Since my classes are imbalanced, I want t o use WeightedRandomSampler (29.08.2024)
    # Load the second column (labels) from training labels csv
    labels = pd.read_csv(train_labels_csv, header=None, usecols=[1]).values.flatten()
    if classification:
        # Initialize the weights array with zeros
        class_weights = np.zeros(no_of_classes)
        # Calculate the frequency of each class
        class_counts = np.bincount(labels)
        logging.info(f"Class counts for my classes: {class_counts}")
        # Fill in the weights only for the classes that exist
        # Weights are the inverse of the class frequency (but we want to avoid dividing by zero due to classes with zero samples!)
        for label in np.unique(labels):
            if class_counts[label] > 0:
                class_weights[label] = 1.0 / class_counts[label]
            else:
                class_weights[label] = 0.0
        logging.info(f"final class weights array: {class_weights}")
        # Assign a weight to each sample based on its original label
        sample_weights = [class_weights[label] for label in labels]
        # Create the sampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # * # * # * # * # * # * # * # * # *

    # trainloader and test loader (you always need that, too) (create testloader with the sampler)
    if classification:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # model weights for pre-trained ResNet 18 already downloaded:
    local_weights_path = '/home/hpc/iwai/iwai110h/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'

    # * * * * * * * * * * * * * * * * * * * *
    # Load ResNet-18 Model:
    model = models.resnet18(pretrained=False)
    # Load the state dictionary from the downloaded weights file if you use a pre-trained ResNet18
    # model.load_state_dict(torch.load(local_weights_path))
    # Modify the final layer to fit the number of classes:
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    # Replace the final fully connected layer with a new one (e.g., for 10 classes)
    if classification:
        logging.info(f"I have {no_of_classes} classes")
        # model.fc is the fully connected layer in ResNet
        model.fc = nn.Linear(num_ftrs, no_of_classes)
        # Define Loss Function
        criterion = nn.CrossEntropyLoss()
    if not classification:
        # define last layer according to regression task
        model.fc = nn.Linear(in_features=num_ftrs, out_features=1)
        # Define Loss Function
        # Use Mean Squared Error (MSE) as the loss function for regression
        criterion = nn.MSELoss()
    # * * * * * * * * * * * * * * * * * * * *

    # Choose the optimizer based on the configuration
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # without wandb:
    # optimizer = optim.SGD(model.parameters(), learning_rate=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), learning_rate=learning_rate)

    # Train the Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_samples = len(trainset)
    n_iterations = math.ceil(total_samples / batch_size)

    # # # # # # # # # # # # # #
    # EARLY STOPPING PARAMETERS:
    # How many epochs to wait after last improvement
    # patience = 30
    # Max number of epochs
    # num_epochs = epochs # set in wandb config now
    # criterion: validation loss
    min_delta = 0.001
    best_val_loss = float('inf')
    counter = 0
    # criterion: validation accuracy
    best_accuracy = 0
    epochs_no_improve = 0
    # # # # # # # # # # # # # 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # APPLY CHECKPOINTING # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Check if a checkpoint exists to resume training
    checkpoint_path = "checkpoint.pth"
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        logging.info("Checkpoint found! Resuming training...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Resume from last epoch
        start_epoch = checkpoint['epoch'] + 1  
        best_val_loss = checkpoint['best_val_loss']

        # Restore training history
        training_losses = checkpoint.get('training_losses', [])  # Load or initialize empty list
        validation_losses = checkpoint.get('validation_losses', [])
        validation_accuracies = checkpoint.get('validation_accuracies', [])

        logging.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss}")

    else:
        logging.info("No checkpoint found. Starting fresh training.")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # end_time_preparation = time.time()
    # elapsed_time_preparation = end_time_preparation - start_time_preparation
    # logging.info(f"Elapsed time for preparation process: {elapsed_time_preparation} seconds")

    # *********************************************************************************************************************
    start_time_training = time.time()
    for epoch in range(start_epoch, epochs):

        logging.info(f'Epoch {epoch+1} started')

        # Set the model to training mode
        model.train()
        running_loss = 0.0

        # # # # # # # # # # # # 
        #    Training loop    #
        # # # # # # # # # # # #

        for batch_idx, (inputs, labels) in enumerate(trainloader):

            # Optionally print some information every 10th step
            # if (batch_idx + 1) % 10 == 0:
            #     logging.info(f'epoch {epoch+1}/{num_epochs}')

            inputs, labels = inputs.to(device), labels.to(device)

            if not classification:
                # Convert labels to float, since regression tasks require continuous values
                labels = labels.float()
                # Reshape labels to match the output shape [32] -> [32, 1]
                labels = labels.view(-1, 1)  # Reshape to [batch_size, 1]
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()

        # Compute epoch loss
        current_training_loss = (running_loss/len(trainloader))
        training_losses.append(current_training_loss)  # Append loss to history
        # logging
        logging.info(f"Epoch {epoch+1}, Training Loss: {current_training_loss}")
        # log metrics to wandb
        wandb.log({"training-loss": current_training_loss})
        
        # Append metrics to lists
        training_losses.append(current_training_loss)

        # # # # # # # # # # # # 
        #   Validation loop   #
        # # # # # # # # # # # #

        # Evaluate the model on validation set after each epoch (Check loss and accuracy)

        # Set the model to evaluation mode
        model.eval()

        correct = 0
        total = 0
        total_mape = 0
        val_loss = 0.0
        total_r2 = 0
        total_samples = 0

        with torch.no_grad(): # in evaluation process gradients do not have to be computed (save memory and computational resources)

            for inputs, labels in valloader:

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                if not classification:
                    # Convert labels to float, since regression tasks require continuous values
                    labels = labels.float()
                    # Reshape labels to match the output shape [32] -> [32, 1]
                    labels = labels.view(-1, 1)  # Reshape to [batch_size, 1]

                # calculate validation loss as early stopping criterium
                loss = criterion(outputs, labels)
                # accumulate total validation loss across all batches in the validation set
                val_loss += loss.item()

                if not classification:
                    # For regression, we directly calculate MAPE and R²
                    # Detach outputs and labels to move them to CPU and convert to numpy
                    outputs_np = outputs.cpu().detach().numpy()
                    labels_np = labels.cpu().detach().numpy()

                if classification:
                    # calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                if not classification:
                    # Calculate MAPE
                    mape = torch.mean(torch.abs((labels - outputs) / labels)).item() * 100  # Percentage form
                    total_mape += mape

                    # Calculate R² using sklearn's r2_score
                    r2 = r2_score(labels_np, outputs_np)
                    total_r2 += r2

                    # Keep track of the number of samples processed
                    total_samples += len(labels)
        
        # Compute average validation loss
        val_loss /= len(valloader)
        validation_losses.append(val_loss)
        logging.info(f"Epoch {epoch+1}, Validation loss: {val_loss}")
        wandb.log({"validation-loss": val_loss})

        if classification:
            # Compute accuracy
            val_accuracy = 100 * correct / total
            logging.info(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy} %")
            wandb.log({"validation-accuracy": val_accuracy})
            validation_accuracies.append(val_accuracy)

        if not classification:
            # Compute mean MAPE and R² over all batches
            mean_mape = total_mape / len(valloader)
            mean_r2 = total_r2 / len(valloader)

            # Log metrics and print them
            logging.info(f"Epoch {epoch+1}, Validation MAPE: {mean_mape} %")
            logging.info(f"Epoch {epoch+1}, Validation R²: {mean_r2}")

            wandb.log({"validation-mape": mean_mape, "validation-r2": mean_r2})

        # Early Stopping Check

        # Option 1: Check Validation Loss
        # Check if validation loss has improved
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
            # save the best model
            torch.save(model, saved_model)
        else:
            counter += 1  # Increment counter if no improvement

        # checkpoint update #
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'training_losses': training_losses,  # Save training loss history
        'validation_losses': validation_losses,  # Save validation loss history
        'validation_accuracies': validation_accuracies
        }

        torch.save(checkpoint, checkpoint_path)  # Save checkpoint
        logging.info(f"Checkpoint saved at epoch {epoch+1}")
        # checkpoint update #
        
        # early stopping execution
        if early_stopping and counter >= patience:
            logging.info(f'Early stopping triggered after {epoch+1} epochs')
            break

        # Option 2: Check Accuracy
        # Check if accuracy has improved 
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     epochs_no_improve = 0 # Reset counter
        #     # Save the best model
        #     torch.save(model, saved_model)
        # else:
        #     epochs_no_improve += 1
        #
        # # early stopping execution
        # if early_stopping == True:
        #     if epochs_no_improve >= patience:
        #         logging.info('Early stopping triggered')
        #         break

    end_time_training = time.time()
    elapsed_time_training = end_time_training - start_time_training
    logging.info(f"Elapsed time for training process: {elapsed_time_training} seconds")

    # *****************************************************************************************
 
    # *****************************************************************************************
    # ********************************** TEST PHASE *******************************************
    # *****************************************************************************************

    # Load the best model after early stopping:
    model = torch.load(saved_model)
    model.to(device)

    # if you just saved weights:
    # model = models.resnet18(pretrained=False)
    # Adapt last layer !!!
    # model.load_state_dict(torch.load('best_model.pth'))

    # Set the model to evaluation mode
    model.eval() 

    if classification:
        # Check Accuracy of best model on test set
        # Initialize counters for correct predictions and total samples
        correct = 0
        total = 0

        # Initialize arrays to store true positives, false positives, and false negatives per class
        true_positives = np.zeros(no_of_classes)
        false_positives = np.zeros(no_of_classes)
        false_negatives = np.zeros(no_of_classes)

        with torch.no_grad():

            # start_time_test = time.time()

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Update total and correct counts for accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # For each class, update TP, FP, FN
                for i in range(no_of_classes):
                    true_positives[i] += ((predicted == i) & (labels == i)).sum().item()  # TP
                    false_positives[i] += ((predicted == i) & (labels != i)).sum().item()  # FP
                    false_negatives[i] += ((predicted != i) & (labels == i)).sum().item()  # FN

            # end_time_test = time.time()
            # elapsed_time_test = end_time_test - start_time_test
            # logging.info(f"Elapsed time for inference process: {elapsed_time_test} seconds")

        # Calculate accuracy
        test_accuracy = 100 * correct / total

        # Calculate precision and recall per class
        precision_per_class = true_positives / (true_positives + false_positives + 1e-6)  # Add small epsilon to avoid division by zero
        recall_per_class = true_positives / (true_positives + false_negatives + 1e-6)

        # Calculate F1 score per class
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-6)

        # Calculate macro-averaged F1 score (mean of F1 scores across all classes)
        macro_f1 = np.mean(f1_per_class)

        # Print precision, recall, F1 per class, macro average F1 and and accuracy
        for i in range(no_of_classes):
            logging.info(f"Class {i} - Precision: {precision_per_class[i] * 100:.2f}%, Recall: {recall_per_class[i] * 100:.2f}%, F1-Score: {f1_per_class[i] * 100:.2f}%")

        logging.info(f"Best Test Accuracy: {test_accuracy:.2f}%")
        logging.info(f"Macro-Averaged F1-Score: {macro_f1 * 100:.2f}%")

        wandb.log({
        "test-accuracy": test_accuracy,
        "macro_f1": macro_f1
        })

    # Regression
    else:
        # Initialize counters for total samples and list to store predicted and actual values
        total = 0
        predicted_vals = []
        actual_vals = []

        with torch.no_grad():

            # start_time_test = time.time()

            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Convert labels to float, since regression tasks require continuous values
                labels = labels.float()
                # Reshape labels to match the output shape [32] -> [32, 1]
                labels = labels.view(-1, 1)  # Reshape to [batch_size, 1]

                # Get the model's predicted outputs
                outputs = model(inputs)

                # Accumulate predicted values and actual labels
                predicted_vals.extend(outputs.cpu().numpy())  # Move predictions to CPU and convert to numpy
                actual_vals.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy

                # Update the total number of samples
                total += labels.size(0)

            # end_time_test = time.time()
            # elapsed_time_test = end_time_test - start_time_test
            # logging.info(f"Elapsed time for inference process: {elapsed_time_test} seconds")

        # Convert lists to numpy arrays for MSE and R² calculation
        predicted_vals = np.array(predicted_vals)
        actual_vals = np.array(actual_vals)
        # Calculate Mean Squared Error (MSE)
        # rmse = root_mean_squared_error(actual_vals, predicted_vals)
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = mean_absolute_percentage_error(actual_vals, predicted_vals)

        # Calculate R-squared (R²)
        r2 = r2_score(actual_vals, predicted_vals)

        # Print the regression performance metrics
        # logging.info(f"Root Mean Squared Error (MSE): {rmse:.4f}")
        logging.info(f"Mean Absolute Percentage Error: {mape * 100}%")
        logging.info(f"R-squared (R²): {r2:.4f}")

        # Log the results to wandb
        wandb.log({
            #"rmse": rmse,
            "mape": mape,
            "r2": r2
        })

    # *****************************************************************************************
    # ********************************** TEST PHASE *******************************************
    # *****************************************************************************************

    # *****************************************************************************************
    # ********************************** INFERENCE ********************************************
    # *****************************************************************************************

    # inference on 10 samples
    # Create a subset containing only the first 10 samples
    subset_indices = list(range(10))
    test_subset = Subset(testset, subset_indices)

    inferenceloader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=1, shuffle=False, num_workers=2)

    with torch.no_grad(): # Disable gradient calculation for faster inference

        # Start timer before the loop
        inference_start_time = time.time()

        for inputs, labels in inferenceloader:

            # Move inputs and labels to the appropriate device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass (inference)
            outputs = model(inputs)

        # Stof timer after the loop
        inference_end_time = time.time()
        elapsed_inference_time = inference_end_time - inference_start_time
        logging.info(f"Elapsed time for 10 samples inference process: {elapsed_inference_time} seconds")
    
    # *****************************************************************************************
    # ****************************** SAVING VALUES TO CSV ********************************************
    # *****************************************************************************************

    # if classification:
    #     # logging.info("PARAMETERS FOR CSV:")
    #     # logging.info(f"hyperparameters: {hyperparameters}")
    #     # logging.info(f"test accuracy: {test_accuracy}")
    #     # logging.info(f"training losses: {training_losses}")
    #     # logging.info(f"validation accuracies: {validation_accuracies}")
    #     # logging.info(f"validation losses: {validation_losses}")

    #     save_to_csv(run, hyperparameters, test_accuracy, training_losses, validation_accuracies, validation_losses)

    #     print(f"Parameters successfully saved to csv.")

    # if regression  
    # method has to be adapted!   
    # else:
    #     save_to_csv(run, hyperparameters, mape, r2, training_losses, validation_losses)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # The code inside this block will only be executed if the script is run as the main program.
    # If the script is imported into another module, the code inside this block will not be executed.

    # If optimizer is 'sgd', use momentum, otherwise set it to None
    if my_config['optimizer'] == 'sgd':
        config = {
            'epochs': my_config['epochs'],
            'learning_rate': my_config['learning_rate'],
            'batch_size': my_config['batch_size'],
            'optimizer': my_config['optimizer'],
            'momentum': my_config['momentum']  # Use momentum for SGD
        }
        hyperparameters = {
            'epochs': my_config['epochs'],
            'learning_rate': my_config['learning_rate'],
            'batch_size': my_config['batch_size'],
            'optimizer': my_config['optimizer'],
            'momentum': my_config['momentum']  # Use momentum for SGD
            }

    else:
        # For 'adam' or 'rmsprop', set momentum to None
        config = {
            'epochs': my_config['epochs'],
            'learning_rate': my_config['learning_rate'],
            'batch_size': my_config['batch_size'],
            'optimizer': my_config['optimizer'],
            'momentum': None  # Use momentum for SGD
        }
        hyperparameters = {
            'epochs': my_config['epochs'],
            'learning_rate': my_config['learning_rate'],
            'batch_size': my_config['batch_size'],
            'optimizer': my_config['optimizer'],
            'momentum': None  # Use momentum for SGD
            }
                                
    # Logging the configuration
    logging.info(f"NEW TRAINING: Epochs: {my_config['epochs']}, Learning Rate: {my_config['learning_rate']}, Batch Size: {my_config['batch_size']}, Optimizer: {my_config['optimizer']}, Momentum: {my_config['momentum']}")

    # Call the train function with the final configuration and hyperparameters
    train(config, run, hyperparameters)