import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torchvision import models
import math
import logging
from customDataset import OneClassImagesDataset
import pandas as pd
import numpy as np
import wandb 
import sys
import os
from sklearn.metrics import precision_score, recall_score

# Checkliste was du anpassen musst:
# 2. project name (2x)
# 3. model_pth_name
# 4. transforms
# 5. labels und frame set

def train():

    # set these parameters for each new training set
    project_name = 'ResNet18-VidVRD-random-subset-1' # muss extra unten angepasst werden
    model_pth_name = 'resnet18_IN_random_subset.pth'

    # Initialize lists to store metrics
    training_losses = []
    validation_accuracies = []
    validation_losses = []

    #logging.info("NEW TRAINING STARTED")
    print("NEW TRAINING STARTED")
    # PLEASE SET THE FOLLOWING PARAMETERS:
    # Do you want to use early stopping? (set to True or False)
    early_stopping = False
    # How many epochs do you want to train (maximally)? --> set in wandb config now
    # epochs = 200

    # ensure that the random numbers generated by PyTorch are deterministic
    torch.manual_seed(42)

    # **************************************************************************************************************************
    # *** WANDB ***
    # wandb.login()
    # start a new wandb run to track this script
    wandb.init(
        project='ResNet18-IN-random-subset',
        reinit=True
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

    print(f"NEW TRAINING: Epochs: {epochs}, Learning Rate: {learning_rate}, Batch Size: {batch_size}, Optimizer: {optimizer_name}, Momentum: {momentum}")

    # **************************************************************************************************************************

    # pre-trained models expect a certain data format (https://pytorch.org/hub/pytorch_vision_resnet/)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # taken from the pre-trained model
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # calculated on ImageNet VidVRD dataset
        transforms.Normalize(mean=[0.458, 0.462, 0.406], std=[0.214, 0.212, 0.208])
        # calculated on EchoNet-Dynamic dataset
        # transforms.Normalize(mean=[0.128, 0.129, 0.130], std=[0.190, 0.191, 0.192])
    ])

    # # # # # # # # # # #
    # set batch size (now it is set above using wandb)
    # batch_size = 64
    # # # # # # # # # # #

    # Load Data (how shall I do that? I first have to write my own Dataset Class (inherit from Dataset))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # examplary data (insert your customDataset Class here!!!)
    # this is the standard way to do
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #
    # my dataset
    #
    train_labels_csv = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/val_test_random_all/IN_train_random.csv"
    #
    trainset = OneClassImagesDataset(csv_file = train_labels_csv, \
                                    root_dir = "/vol/ideadata/ep56inew/ImageNet_VidVRD/again_official_split/train_random", \
                                    transform = transform)

    valset = OneClassImagesDataset(csv_file = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/val_test_random_all/IN_val.csv", \
                                    root_dir = "/vol/ideadata/ep56inew/ImageNet_VidVRD/again_official_split/val_subset", \
                                    transform = transform)

    testset = OneClassImagesDataset(csv_file = "/vol/ideadata/ep56inew/ImageNet_VidVRD/labels/official_split/val_test_random_all/IN_test.csv", \
                                    root_dir = "/vol/ideadata/ep56inew/ImageNet_VidVRD/again_official_split/test_subset", \
                                    transform = transform)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # * # * # * # * # * # * # * # * # *
    # Since my classes are imbalanced, I want t o use WeightedRandomSampler (29.08.2024)
    # Load the second column (labels) from training labels csv
    labels = pd.read_csv(train_labels_csv, header=None, usecols=[1]).values.flatten()
    # all classes
    classes = list(set(labels))
    # Find the maximum label value
    max_label = np.max(labels)
    # Initialize the weights array with zeros
    class_weights = np.zeros(max_label + 1)
    # Calculate the frequency of each class
    class_counts = np.bincount(labels)
    # Fill in the weights only for the classes that exist
    # Weights are the inverse of the class frequency (but we want to avoid dividing by zero due to classes with zero samples!)
    for label in np.unique(labels):
        if class_counts[label] > 0:
            class_weights[label] = 1.0 / class_counts[label]
        else:
            class_weights[label] = 0.0
    # Assign a weight to each sample based on its original label
    sample_weights = [class_weights[label] for label in labels]
    # Create the sampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    # * # * # * # * # * # * # * # * # *

    # trainloader and test loader (you always need that, too) (create testloader with the sampler)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=2, sampler=sampler)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # model weights for pre-trained ResNet 18 already downloaded:
    # local_weights_path = '/home/hpc/iwai/iwai110h/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'

    # * * * * * * * * * * * * * * * * * * * *
    # Load ResNet-18 Model:
    model = models.resnet18(pretrained=False)
    # Load the state dictionary from the downloaded weights file if you use a pre-trained ResNet18
    # model.load_state_dict(torch.load(local_weights_path))
    # Modify the final layer to fit the number of classes:
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    # Replace the final fully connected layer with a new one (e.g., for 10 classes)
    num_classes = 29
    model.fc = nn.Linear(num_ftrs, num_classes)
    # * * * * * * * * * * * * * * * * * * * *

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()

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
    patience = 20
    # Max number of epochs
    # num_epochs = epochs # set in wandb config now
    # criterion: validation loss
    min_delta = 0.001
    best_val_loss = float('inf')
    counter = 0
    # criterion: validation accuracy
    best_accuracy = 0
    epochs_no_improve = 0
    # # # # # # # # # # # # #  #

    # *********************************************************************************************************************
    for epoch in range(epochs):

        # logging.info(f'Epoch {epoch+1} started')
        print(f'Epoch {epoch+1} started')

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
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(trainloader)}")
        # log metrics to wandb
        current_training_loss = (running_loss/len(trainloader))
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
        val_loss = 0.0

        with torch.no_grad(): # in evaluation process gradients do not have to be computed (save memory and computational resources)

            for inputs, labels in valloader:

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # calculate validation loss as early stopping criterium
                loss = criterion(outputs, labels)
                # accumulate total validation loss across all batches in the validation set
                val_loss += loss.item()

                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Compute average validation loss
        val_loss /= len(valloader)

        # Compute accuracy
        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy} %")
        print(f"Epoch {epoch+1}, Validation loss: {val_loss}")
        # wandb logging
        # Log metrics
        wandb.log({"validation-accuracy": val_accuracy, "validation-loss": val_loss})

        # Append metrics to lists
        validation_accuracies.append(val_accuracy)
        validation_losses.append(val_loss)

        # Early Stopping Check

        # Option 1: Check Validation Loss
        # Check if validation loss has improved
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0  # Reset counter if validation loss improves
            # save the best model
            torch.save(model, model_pth_name)
        else:
            counter += 1  # Increment counter if no improvement
        
        # early stopping execution
        if early_stopping == True:
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        # Option 2: Check Accuracy
        # Check if accuracy has improved 
        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     epochs_no_improve = 0 # Reset counter
        #     # Save the best model
        #     torch.save(model, model_pth_name)
        # else:
        #     epochs_no_improve += 1
        #
        # # early stopping execution
        # if early_stopping == True:
        #     if epochs_no_improve >= patience:
        #         print('Early stopping triggered')
        #         break

    # *********************************************************************************************************************

    # Load the best model after early stopping:
    model = torch.load(model_pth_name)
    model.to(device)

    # if you just saved weights:
    # model = models.resnet18(pretrained=False)
    # Adapt last layer !!!
    # model.load_state_dict(torch.load('best_model.pth'))

    # Set the model to evaluation mode
    model.eval()  

    # *****************************************************************************************
    # ********************************** TEST PHASE BEGIN *************************************
    # *****************************************************************************************

    # Check Accuracy of best model on test set
    # Initialize counters for correct predictions and total samples
    correct = 0
    total = 0

    # for EchoNet-Dynamic specifically
    num_classes = 27

    # Initialize arrays to store true positives, false positives, and false negatives per class
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct counts for accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # For each class, update TP, FP, FN
            for i in range(num_classes):
                true_positives[i] += ((predicted == i) & (labels == i)).sum().item()  # TP
                false_positives[i] += ((predicted == i) & (labels != i)).sum().item()  # FP
                false_negatives[i] += ((predicted != i) & (labels == i)).sum().item()  # FN

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
    for i in range(num_classes):
        print(f"Class {i} - Precision: {precision_per_class[i] * 100:.2f}%, Recall: {recall_per_class[i] * 100:.2f}%, F1-Score: {f1_per_class[i] * 100:.2f}%")

    print(f"Macro-Averaged F1-Score: {macro_f1 * 100:.2f}%")
    print(f"Best Test Accuracy: {test_accuracy:.2f}%")

    wandb.log({
    "test-accuracy": test_accuracy,
    "macro_f1": macro_f1
    })

    # *****************************************************************************************
    # ********************************** TEST PHASE END ***************************************
    # *****************************************************************************************

    # Ensure the run is marked as finished
    wandb.finish()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# wandb configuration

sweep_config = {
    'method': 'bayes',  # Can be grid, random, or bayes
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'  # We're trying to minimize the loss
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']  # Different optimizers to try
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.9  # Momentum values for SGD
        },
        'batch_size': {
            'values': [32, 64, 128] # common starting points in many deep learning experiments
        },
        'epochs': {
            'values': [15]  # Epochs to run [50, 100, 150]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='ResNet18-VidVRD-random-subset-1') # set the wandb project where this run will be logged
wandb.agent(sweep_id, function=train, count=15)