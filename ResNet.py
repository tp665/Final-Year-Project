import torch.nn as nn
from torchvision import models, transforms, datasets
import torch.optim as optim
import torch
import time
import utils
from torch.utils.data import DataLoader
import os


def get_resnet(device, img_dims=224, number_of_classes=10):
    # Load pretrained Model
    net = models.resnet18(pretrained=True)
    net = net.to(device)

    # Freeze model parameters
    for param in net.parameters():
        param.requires_grad = False

    # Change the final layer of ResNet50 Model for Transfer Learning
    fc_inputs = net.fc.in_features

    net.fc = nn.Sequential(
        nn.Linear(fc_inputs, 720),
        nn.ReLU(),
        nn.Linear(720, number_of_classes)
    )

    # Convert model to be used on device
    net = net.to(device)

    # Define Optimizer and Loss Function
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adamax(net.parameters())

    # train final layer
    # Train the model for 25 epochs
    num_epochs = 25
    net, _, _ = train_and_validate(net, loss_func, optimiser, device, img_dims, num_epochs)

    # Release model parameters
    for param in net.parameters():
        param.requires_grad = True

    # Create hooks for feature statistics catching
    bn_layer_stats = []
    for module in net.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_layer_stats.append(utils.BatchNormHook(module))

    return net, bn_layer_stats

def train_and_validate(model, loss_criterion, optimizer, device, img_dims, epochs=25):
    # Load the Data
    # Set train and valid directory paths
    dataset = 'dataset'
    train_directory = os.path.join(dataset, 'train')
    valid_directory = os.path.join(dataset, 'valid')

    # define transforms
    image_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=img_dims, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=img_dims),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=img_dims),
            transforms.CenterCrop(size=img_dims),
            transforms.ToTensor(),
        ])
    }

    # Load Data from folders
    data = {
        'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    }

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    training_batch_size = 256
    train_data_loader = DataLoader(data['train'], batch_size=training_batch_size, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=training_batch_size, shuffle=True)
    history = []
    best_loss = 100000.0
    best_epoch = None
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):
            epoch_start = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch

        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\tValidation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
            
    return model, history, best_epoch
    
    