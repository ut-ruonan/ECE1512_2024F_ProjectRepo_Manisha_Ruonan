#!/usr/bin/env python
# coding: utf-8
"""
Header for Continual_Learning_mnist_mhist.py
--------------------
This code is the answer for "Part 1 Question 4: Apply your synthetic small datasets to one of the machine learning applications."

The synthetic small dataset is applied to continual leaning. So this case a model is trained and evaluated.
code is constructed by refering from: https://www.kaggle.com/code/dlarionov/continual-learning-on-permuted-mnist

Part II as of the experiment set up for continual learning in paper: 
DATASET CONDENSATION WITH GRADIENT MATCHING: Ref- "B. Zhao, K. R. Mopuri, and H. Bilen, “Dataset condensation with gradient matching,” arXiv preprint:2006.05929, 2021."

"""
import os
import random
import numpy as np
import torch
import nbimporter
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision.transforms as transforms
from collections import Counter
from utils_continual import (
    get_dataset,
    get_loops,
    # get_network,
    get_eval_pool,
    evaluate_synset,
    get_default_convnet_setting,
)

from torch.utils.data import Subset
import torch.nn.functional as F

from Covnet_continual import ConvNet

# Set seeds for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device and hyperparameters
device = torch.device("cpu")
train_bs = 64  # Training batch size
test_bs = 2000  # Testing batch size
lr = 0.1  # Learning rate
gamma = 0.9  # Decay factor
num_tasks = 5  # Number of tasks (adjust as needed for growing test sets)

# Define the paths to your saved datasets
base_path = 'C:\\Users\\mahagam3\\Documents\\ECE course\\Project A\\saved_datasets\\'  # Update this if necessary
synthetic_dataset_mhist_path = os.path.join(base_path, 'synthetic_mhist_gray.pt')
synthetic_dataset_mnist_path = os.path.join(base_path, 'synthetic_mnist.pt')
mhist_test_loader_path = os.path.join(base_path, 'test_mhist.pt')
mnist_test_loader_path = os.path.join(base_path, 'test_mnist.pt')

# Load the datasets
synthetic_dataset_mhist = torch.load(synthetic_dataset_mhist_path)
synthetic_dataset_mnist = torch.load(synthetic_dataset_mnist_path)
mhist_test_loader = torch.load(mhist_test_loader_path)
mnist_test_loader = torch.load(mnist_test_loader_path)

# Function to create test subset
def get_test_subset(dataset, size):
    size = min(size, len(dataset))
    indices = np.random.choice(len(dataset), size, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=test_bs, shuffle=False)

# MHIST configurations
num_images_mhist = 100
num_classes_mhist = 2
# image_size_mhist = 224 * 224
channels_mhist = 1

# MNIST configurations
num_images_mnist = 100
num_classes_mnist = 10
# image_size_mnist = 28 * 28
channels_mnist = 1

# Resize MHIST images 
transform_resize_mhist = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 224x224
    transforms.ToTensor()  # Convert to tensor
])

# Resize MNIST images
transform_resize_mnist = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 28x28
    transforms.ToTensor()  # Convert to tensor
])

# Synthetic data for MHIST
synthetic_images_tensor_mhist = torch.rand(num_images_mhist, channels_mhist, 32, 32)
synthetic_labels_tensor_mhist = torch.randint(0, num_classes_mhist, (num_images_mhist,))

# Synthetic data for MNIST
synthetic_images_tensor_mnist = torch.rand(num_images_mnist, channels_mnist, 32, 32)
synthetic_labels_tensor_mnist = torch.randint(0, num_classes_mnist, (num_images_mnist,))

# Create synthetic datasets
synthetic_dataset_mhist = TensorDataset(synthetic_images_tensor_mhist, synthetic_labels_tensor_mhist)
synthetic_dataset_mnist = TensorDataset(synthetic_images_tensor_mnist, synthetic_labels_tensor_mnist)

# Create tasks for MHIST dataset
tasks_mhist = [
    (DataLoader(TensorDataset(
        synthetic_images_tensor_mhist[:2000 * (i + 1)], 
        synthetic_labels_tensor_mhist[:2000 * (i + 1)]
    ), batch_size=train_bs, shuffle=True), 
    TensorDataset(
        synthetic_images_tensor_mhist[:2000 * (i + 1)], 
        synthetic_labels_tensor_mhist[:2000 * (i + 1)]
    ))
    for i in range(num_tasks)
]

# Create tasks for MNIST dataset
tasks_mnist = [
    (DataLoader(TensorDataset(
        synthetic_images_tensor_mnist[:2000 * (i + 1)], 
        synthetic_labels_tensor_mnist[:2000 * (i + 1)]
    ), batch_size=train_bs, shuffle=True), 
    TensorDataset(
        synthetic_images_tensor_mnist[:2000 * (i + 1)], 
        synthetic_labels_tensor_mnist[:2000 * (i + 1)]
    ))
    for i in range(num_tasks)
]


# Replay Buffer Class
class ReplayBuffer:
    def __init__(self, buffer_size=100):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_to_buffer(self, samples):
        for sample in samples:
            if len(sample) == 2:
                input_tensor, target_tensor = sample
                target_tensor = target_tensor.view(-1)  # Flatten to 1D tensor
                self.buffer.append((input_tensor, target_tensor))

                # Maintain buffer size
                if self.buffer_size and len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)  # Remove oldest sample if buffer exceeds size

    def sample_from_buffer(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        inputs = [sample[0] for sample in samples]
        targets = [sample[1] for sample in samples]

        inputs_tensor = torch.stack(inputs) if inputs else torch.empty(0)
        targets_tensor = torch.stack(targets) if targets else torch.empty(0)

        return list(zip(inputs_tensor, targets_tensor))

# Function to create test subset
def get_test_subset(dataset, size):
    size = min(size, len(dataset))
    indices = np.random.choice(len(dataset), size, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=test_bs, shuffle=False)

# Train Model with Replay Buffer
def train_model_with_replay(model, device, synthetic_loader, replay_buffer, optimizer, criterion, scheduler, replay_ratio=0.3):
    model.train()
    synthetic_data_iter = iter(synthetic_loader)
    total_steps = len(synthetic_loader)
    
    for step in range(total_steps):
        # Get current batch from synthetic data
        try:
            synthetic_batch = next(synthetic_data_iter)
        except StopIteration:
            synthetic_data_iter = iter(synthetic_loader)
            synthetic_batch = next(synthetic_data_iter)
        
        # Sample from replay buffer
        replay_batch = replay_buffer.sample_from_buffer(int(replay_ratio * len(synthetic_batch[0])))

        # Add synthetic data to replay buffer
        synthetic_samples = list(zip(synthetic_images_tensor_mhist.view(-1, 1, 32, 32), synthetic_labels_tensor_mhist))
        replay_buffer.add_to_buffer(synthetic_samples)

        # Combine synthetic data with replay data
        if replay_batch:
            # Ensure replay input tensors have the correct shape
            replay_inputs = []
            for x in replay_batch:
                input_tensor = x[0]
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)  # Convert to (1, H, W, C)
                input_tensor = F.interpolate(input_tensor, size=(32, 32), mode='bilinear', align_corners=False)
                replay_inputs.append(input_tensor)

            # Resize synthetic input
            synthetic_input = synthetic_batch[0]
            if synthetic_input.dim() == 3:
                synthetic_input = synthetic_input.unsqueeze(0)
            synthetic_input = F.interpolate(synthetic_input, size=(32, 32), mode='bilinear', align_corners=False)

            # Concatenate inputs and targets
            combined_inputs = torch.cat([synthetic_input] + replay_inputs, dim=0)
            combined_targets = torch.cat([synthetic_batch[1]] + [x[1] for x in replay_batch], dim=0)
        else:
            combined_inputs, combined_targets = synthetic_batch

        # Move to device
        combined_inputs, combined_targets = combined_inputs.to(device), combined_targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(combined_inputs)
        loss = criterion(output, combined_targets)
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()

# Test function with growing test set
def test_with_growing_test_set(model, device, test_dataset, num_tasks, tasks):
    test_set_sizes = [2000, 4000]
    metrics = []
    replay_buffer = ReplayBuffer(buffer_size=5000)  # Replay buffer with a limit of 5000 samples

    # Evaluate untrained model on test subsets
    for size in test_set_sizes:
        test_loader = get_test_subset(test_dataset, size)
        test_acc = evaluate_model(model, test_loader)
        print(f"Test accuracy on {size} images: {test_acc:.2f}%")
        metrics.append(test_acc)
    
    for i in range(num_tasks):
        print(f'Train on Task {i + 1}')
        synthetic_loader, synthetic_data = tasks[i]  # Unpack the task
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        
        # Train the model on the current task using memory replay
        train_model_with_replay(model, device, synthetic_loader, replay_buffer, optimizer, criterion, scheduler)
        
        # Add the synthetic data to the replay buffer
        replay_buffer.add_to_buffer(synthetic_data.tensors)
        
        # Evaluate the model on different test set sizes
        for size in test_set_sizes:
            test_loader = get_test_subset(test_dataset, size)
            test_acc = evaluate_model(model, test_loader)
            print(f"Test accuracy after training on Task {i + 1} with {size} test images: {test_acc:.2f}%")
            metrics.append(test_acc)
    
    return metrics
    
# Evaluate ConvNet3
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    return accuracy


# Create and test models for both datasets
all_experiments_mhist = []
all_experiments_mnist = []
num_experiments = 5

def get_network(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
    model = ConvNet(channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size)
    return model

convnet7_mhist = get_network(
    channel=channels_mhist, 
    num_classes=num_classes_mhist, 
    net_width=128,  # Set your desired width
    net_depth=7,    # This value was previously indicated
    net_act='relu',  # Use the activation function of your choice
    net_norm='batchnorm',  # Specify normalization (if needed)
    net_pooling='maxpooling',  # Specify pooling method (if needed)
    im_size=(32, 32)  # Ensure this matches your input size
).to(device)

convnet3_mnist = get_network(
    channel=channels_mnist, 
    num_classes=num_classes_mnist, 
    net_width=128,  # Adjust if needed
    net_depth=3,    # This value was previously indicated
    net_act='relu',  # Use the activation function of your choice
    net_norm='batchnorm',  # Specify normalization (if needed)
    net_pooling='maxpooling',  # Specify pooling method (if needed)
    im_size=(32, 32)  # Ensure this matches your input size
).to(device)

for experiment in range(num_experiments):
    print(f"Running Experiment {experiment + 1} for MHIST")
    model_mhist = convnet7_mhist
    criterion = torch.nn.CrossEntropyLoss()
    metrics_mhist = test_with_growing_test_set(model_mhist, device, synthetic_dataset_mhist, num_tasks, tasks_mhist)
    all_experiments_mhist.append(metrics_mhist)

    print(f"Running Experiment {experiment + 1} for MNIST")
    model_mnist = convnet3_mnist
    metrics_mnist = test_with_growing_test_set(model_mnist, device, synthetic_dataset_mnist, num_tasks, tasks_mnist)
    all_experiments_mnist.append(metrics_mnist)


# Convert results to numpy array for easy computation of mean and std
all_experiments_mnist = np.array(all_experiments_mnist)

mean_results_mnist = np.mean(all_experiments_mnist, axis=0)
std_results_mnist = np.std(all_experiments_mnist, axis=0)

# Print mean and std for each task
for i, (mean, std) in enumerate(zip(mean_results_mnist, std_results_mnist)):
    print(f"Task {i + 1} - Mean accuracy MNIST: {mean:.4f}, Standard deviation MNIST: {std:.4f}")


# Convert results to numpy array for easy computation of mean and std
all_experiments_mhist = np.array(all_experiments_mhist)

mean_results_mhist = np.mean(all_experiments_mhist, axis=0)
std_results_mhist = np.std(all_experiments_mhist, axis=0)

# Print mean and std for each task
for i, (mean, std) in enumerate(zip(mean_results_mhist, std_results_mhist)):
    print(f"Task {i + 1} - Mean accuracy MHSIT: {mean:.4f}, Standard deviation MHSIT: {std:.4f}")

