#!/usr/bin/env python
# coding: utf-8

"""
Header for MHIST_Task1_main.py
--------------------
This is the code for the "Task 1: Dataset Distillation with Attention Matching' Question 2. Dataset Distillation Learning - MHIST Dataset"

    ########################################### Code start here for MHIST Dataset ############################################
"""
# #Import Required Libraries

# In[ ]:


get_ipython().system('pip install nbimporter')
get_ipython().system('pip install torch torchvision')
get_ipython().system('pip install fvcore')
get_ipython().system('pip install thop')
get_ipython().system('pip install matplotlib')


# # Import Required Functions

# In[ ]:


import sys
sys.path.append('C:\\Users\\manis\\Documents\\ECE digital image processing_Class\\Project A')

import nbimporter
from networks import ConvNet 
from utils import get_dataset, get_loops, get_eval_pool, evaluate_synset, get_default_convnet_setting, get_network

import os
import time
import zipfile
import random
import thop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from collections import Counter

from torchvision.utils import save_image
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image

device = torch.device('cpu')


# # Load datasets

# In[ ]:


data_path = 'C:/Users/mahagam3/Documents/ECE course/Project A'
zip_file_path = os.path.join(data_path, 'images.zip')  
extract_folder = os.path.join(data_path, 'images')  
os.makedirs(extract_folder, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f'Images extracted to {extract_folder}')


# In[ ]:


data_path = 'C:/Users/mahagam3/Documents/ECE course/Project A/images/images'
annotations_path = os.path.join(data_path, 'annotations.csv') 
annotations = pd.read_csv(annotations_path)

# Resize transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 32x32
    transforms.ToTensor()  # Convert to tensor
])

class MHISTDataset(Dataset):
    def __init__(self, annotations, data_path, partition, transform=None):
        self.annotations = annotations[annotations['Partition'] == partition]  # Filter based on partition
        self.data_path = data_path
        self.transform = transform  # Use the transform passed in

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['Image Name']  # Get image name
        label = self.annotations.iloc[idx]['Majority Vote Label']  # Get label from the updated column
        label = 0 if label == 'HP' else 1 
        
        # Load image
        image = Image.open(os.path.join(self.data_path, img_name)).convert("RGB")
        
        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Create datasets for training and testing
mhist_train = MHISTDataset(annotations, data_path, partition='train', transform=transform)
mhist_test = MHISTDataset(annotations, data_path, partition='test', transform=transform)

# Create DataLoaders
mhist_loader = DataLoader(mhist_train, batch_size=64, shuffle=True)
mhist_test_loader = DataLoader(mhist_test, batch_size=64, shuffle=False)


# In[ ]:


class_counts = annotations['Majority Vote Label'].value_counts()
print("Number of images per class:")
print(class_counts)


# In[ ]:


#################### Edit to MHIST ###################
 ###### MHIST Dataset ######
from collections import Counter

# Initialize a counter for the classes
class_counts = Counter()

# Iterate through the training dataset
for _, label in mhist_train:
    class_counts[label] += 1  # Increment the count for the corresponding class

# Print the number of images per class
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} images")


# In[ ]:


# Define parameters for MHIST ConvNet7
channel_mhist = 3  
num_classes_mhist = 2 
im_size_mhist = (224, 224)

# Resize MHIST images
transform_resize = transforms.Compose([
    transforms.Resize(im_size_mhist),  
    transforms.ToTensor() 
])

# Instantiate ConvNetD7
convnet7 = get_network('ConvNetD7', channel_mhist, num_classes_mhist, im_size_mhist).to('cpu')

optimizer7 = optim.SGD(convnet7.parameters(), lr=0.01) # Define optimizer
criterion = nn.CrossEntropyLoss() # Define loss function
scheduler7 = CosineAnnealingLR(optimizer7, T_max=20) # Cosine Annealing Scheduler

# Create datasets for training and testing
mhist_train = MHISTDataset(annotations, data_path, partition='train', transform=transform)
mhist_test = MHISTDataset(annotations, data_path, partition='test', transform=transform)

# Create DataLoaders
mhist_loader = DataLoader(mhist_train, batch_size=64, shuffle=True)
mhist_test_loader = DataLoader(mhist_test, batch_size=64, shuffle=False)

"""
# # Part 1 Question 2(a):
Train the selected model with the original dataset and report the classification accuracy along  with floating-point operations per second (FLOPs) for the test set. Use SGD as an optimizer with a cosine annealing scheduler with an initial learning rate of 0.01 for 20 epochs. (For more information on experimental setting, look at the implementation details of [51]) These scores give you the upper bound benchmark evaluation.
"""

# In[ ]:


# Training Function
def train_model(model, dataloader, optimizer, criterion, scheduler, num_epochs=20):
    model.train()  # Set the model to training mode
    start_time = time.time()  # Start timing
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()  # Zero the parameter gradients
            images, labels = images.to(device), labels.to(device)  # Move to device
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model
            running_loss += loss.item()
        
        scheduler.step()  # Step the learning rate scheduler
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MHIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time

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


# In[ ]:


# Training ConvNet7 on MHIST
print("Training ConvNet7 on MHIST...")
train_model(convnet7, mhist_loader, optimizer7, criterion, scheduler7)

# Evaluate ConvNet7 on test set
accuracy_mhist = evaluate_model(convnet7, mhist_test_loader)
print(f'ConvNet7 Test Accuracy on MHIST: {accuracy_mhist:.2f}%')


# In[ ]:


# Flop is used to measure the coputational cost of running the model and to evaluate the efficiency of the learning model.
# Calculate FLOPs (Floating Point Operations per Second)
def calculate_flops(model, input_tensor):
    from thop import profile
    
    # Clear any existing 'total_ops' and 'total_params' attributes
    for layer in model.modules():
        if hasattr(layer, 'total_ops'):
            del layer.total_ops  # Remove existing 'total_ops'
        if hasattr(layer, 'total_params'):
            del layer.total_params  # Remove existing 'total_params'

    # Now calculate FLOPs
    flops, params = profile(model, inputs=(input_tensor,))
    return flops

# Input tensor for calculating FLOPs
input_tensor_mhist = torch.randn(len(mhist_train), channel_mhist, 224, 224)
flops_mhist = calculate_flops(convnet7, input_tensor_mhist)
print(f'FLOPs for ConvNet7: {flops_mhist:.2f} FLOPs')

"""
Part 1: Question 2(b): 
Learn the synthetic dataset S using the selected model and Attention Matching algorithm. For initialization of condensed images, randomly select from real training images. The experimental setup can be found in Table 1.
"""

# In[ ]:


# Set hyperparameters for MHIST
K_mhist = 200
T_mhist = 10
eta_S_mhist = 0.1
zeta_S_mhist = 1
eta_theta_mhist = 0.01
zeta_theta_mhist = 50
lambda_mhist = 0.01
num_classes_mhist = 2
images_per_class_mhist = 50
batch_size_mhist = 128


# In[ ]:


# Function to generate a synthetic dataset S by randomly sampling from a real dataset
def generate_synthetic_dataset(real_dataset, num_classes, images_per_class):
    synthetic_images = []
    synthetic_labels = []
    
    for class_id in range(num_classes):
        indices = random.sample(
            [i for i, (_, label) in enumerate(real_dataset) if label == class_id],
            images_per_class
        )
        synthetic_images.extend([real_dataset[i][0] for i in indices])
        synthetic_labels.extend([real_dataset[i][1] for i in indices])
    
    return synthetic_images, synthetic_labels


# In[ ]:


# Generate synthetic samples for MHIST
synthetic_images_mhist, synthetic_labels_mhist = generate_synthetic_dataset(
    mhist_train, num_classes_mhist, images_per_class_mhist
)

# Convert synthetic images to a tensor
synthetic_images_tensor_mhist = torch.stack(synthetic_images_mhist).to(device)
synthetic_labels_tensor_mhist = torch.tensor(synthetic_labels_mhist).to(device)

# Create DataLoader for the synthetic MHIST dataset
synthetic_dataset_mhist = torch.utils.data.TensorDataset(synthetic_images_tensor_mhist, synthetic_labels_tensor_mhist)
synthetic_loader_mhist = DataLoader(synthetic_dataset_mhist, batch_size=batch_size_mhist, shuffle=True)


# In[ ]:


##### Save the synthetic images ########
# Save synthetic images as PNG files
output_dir = "synthetic_images_mhist"
os.makedirs(output_dir, exist_ok=True)

# Save each synthetic image as a PNG file
for idx, synthetic_image in enumerate(synthetic_images_mhist):
    image_path = os.path.join(output_dir, f"synthetic_image_{idx}.png")
    save_image(synthetic_image, image_path)

print(f"Synthetic images saved in {output_dir}")

# Save synthetic images and labels as a .pt file
torch.save({
    'images': synthetic_images_tensor_mhist,
    'labels': synthetic_labels_tensor_mhist
}, "synthetic_mhist.pt")

print("Synthetic images and labels saved as 'synthetic_mhist.pt'")


# In[ ]:


from attention_module import get_attention
# Training Function with Attention Matching
def train_with_attention_matching(model, synthetic_dataloader, optimizer, criterion, 
                                  param=0, exp=4, num_epochs=10, lambda_param=0.01):
    model.train()
    start_time = time.time()  # Start timing
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in synthetic_dataloader:
            optimizer.zero_grad()
            
            # Forward pass to get features
            features = model(images)
            # Get attention maps
            attention_maps = get_attention(features, param=param, exp=exp)
            
            # Compute loss with attention matching
            loss = criterion(features, labels)
            # Incorporate task balance parameter λ
            loss += lambda_param * torch.mean(attention_maps)  
            
            loss.backward()  
            optimizer.step() 
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(synthetic_dataloader):.4f}')
        total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MHIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time

import torch.nn.functional as F

# Optimizer setup for the model (using SGD as per the table)
optimizer_mhist = torch.optim.SGD(convnet7.parameters(), lr=eta_theta_mhist)
convnet7 = get_network('ConvNetD7', channel_mhist, num_classes_mhist, im_size_mhist).to(device)

# Train the model using the synthetic datasets, cross_entropy measure how well the model predicts the classes.
print("Training on MHIST dataset...")
train_with_attention_matching(convnet7, synthetic_loader_mhist, optimizer_mhist, F.cross_entropy, 
                              param=0, exp=4, num_epochs=T_mhist, lambda_param=lambda_mhist)

# # Evaluate ConvNet7 on the test set
# accuracy_attention_mhist = evaluate_model(convnet7, mhist_test_loader)
# print(f'ConvNet7 Test Accuracy on MHIST: {accuracy_attention_mhist:.2f}%')


# # Part 1: Question 2(c):
# Provide the visualization of condensed images per class for both MNIST and MHIST datasets. 
# Do you think these condensed images are recognizable? Support your explanations.

# In[ ]:


def visualize_condensed_images(synthetic_images, num_classes, images_per_class, title, save_folder):
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class * 2, num_classes * 2))
    fig.suptitle(title)

    for class_id in range(num_classes):
        for img_id in range(images_per_class):
            idx = class_id * images_per_class + img_id
            # Permute the tensor to make it compatible with imshow (from CxHxW to HxWxC)
            img = synthetic_images[idx].permute(1, 2, 0).cpu().numpy()
            axes[class_id, img_id].imshow(img)
            axes[class_id, img_id].axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_folder}/{title}.png')
    plt.show()

save_folder = 'Save_images'

# visualize MHIST when data is available
visualize_condensed_images(synthetic_images_tensor_mhist, num_classes_mhist, images_per_class_mhist, 
                           title="Condensed Images for MHIST", save_folder=save_folder)

"""
Part 1 Question 2(d):
Repeat parts 2b and 2c while the condensed images are initialized with Gaussian noise. Discuss in full detail the qualitative and quantitative results you have achieved. Are the results and visualizations are comparable with parts 2b and 2c?
"""
# # Repeat parts 2(b) while the condensed images are initialized with Gaussian noise

# In[ ]:


# Function to generate synthetic dataset S with Gaussian noise
# Reduce the standard deviation to make the noise less pronounced, leads to less spread in the distribution,......
# meaning values stay closer to the mean. This results in less noise.
def generate_synthetic_dataset_with_noise(real_dataset, num_classes, images_per_class, noise_std=0.8):
    synthetic_images = []
    synthetic_labels = []
    
    for class_id in range(num_classes):
        indices = random.sample(
            [i for i, (_, label) in enumerate(real_dataset) if label == class_id],
            images_per_class
        )
        for i in indices:
            # Generate Gaussian noise
            noise = torch.normal(mean=0, std=noise_std, size=real_dataset[i][0].size())
            synthetic_image = real_dataset[i][0] + noise
            
            # Ensure the pixel values are within valid range
            synthetic_image = torch.clamp(synthetic_image, 0, 1)
            
            synthetic_images.append(synthetic_image)
            synthetic_labels.append(real_dataset[i][1])
    
    return synthetic_images, synthetic_labels


# In[ ]:


# Generating synthetic samples for MHIST
synthetic_images_mhist_noise, synthetic_labels_mhist_noise = generate_synthetic_dataset_with_noise(
    mhist_train, num_classes_mhist, images_per_class_mhist
) 
# Convert synthetic images to a tensor
synthetic_images_tensor_mhist_noise = torch.stack(synthetic_images_mhist_noise).to(device)
synthetic_labels_tensor_mhist_noise = torch.tensor(synthetic_labels_mhist_noise).to(device)

# Create DataLoader for the synthetic MHIST dataset with noise
synthetic_dataset_mhist_noise = torch.utils.data.TensorDataset(synthetic_images_tensor_mhist_noise, synthetic_labels_tensor_mhist_noise)
synthetic_loader_mhist_noise = DataLoader(synthetic_dataset_mhist_noise, batch_size=batch_size_mhist, shuffle=True)


# In[ ]:


##### Save the synthetic images with noise ########
# Save synthetic images as PNG files
output_dir = "synthetic_images_mhist_noise"
os.makedirs(output_dir, exist_ok=True)

# Save each synthetic image as a PNG file
for idx, synthetic_image in enumerate(synthetic_images_mhist_noise):
    image_path = os.path.join(output_dir, f"synthetic_image_{idx}.png")
    save_image(synthetic_image, image_path)

print(f"Synthetic images saved in {output_dir}")

# Save synthetic images and labels as a .pt file
torch.save({
    'images': synthetic_images_tensor_mhist_noise,
    'labels': synthetic_labels_tensor_mhist_noise
}, "synthetic_mhist_noise.pt")

print("Synthetic images and labels saved as 'synthetic_mhist_noise.pt'")


# In[ ]:


# Training Function with Attention Matching
def train_with_attention_matching(model, synthetic_dataloader, optimizer, criterion, 
                                  param=0, exp=4, num_epochs=10, lambda_param=0.01):
    model.train()
    start_time = time.time()  # Start timing
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in synthetic_dataloader:
            optimizer.zero_grad()
            
            # Forward pass to get features
            features = model(images)
            # Get attention maps
            attention_maps = get_attention(features, param=param, exp=exp)
            
            # Compute loss with attention matching
            loss = criterion(features, labels)
            # Incorporate task balance parameter lambda
            loss += lambda_param * torch.mean(attention_maps)  # Adjust this based on your needs
            
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(synthetic_dataloader):.4f}')
        total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MHIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time

# Optimizer setup for the model (using SGD as per the table)
optimizer_mhist_noise = torch.optim.SGD(convnet7.parameters(), lr=eta_theta_mhist)

# Train the model using the synthetic datasets with Gaussian noise
print("Training on MHIST dataset with Gaussian noise...")
train_with_attention_matching(convnet7, synthetic_loader_mhist_noise, optimizer_mhist_noise, 
                              F.cross_entropy, param=0, exp=4, num_epochs=T_mhist, lambda_param=lambda_mhist)

# # Evaluate ConvNet7 on the test set
# accuracy_attention_mhist_noise = evaluate_model(convnet7, mhist_test_loader)
# print(f'ConvNet7 Test Accuracy on MHIST: {accuracy_attention_mhist_noise:.2f}%')


# # Repeat parts 2(c) while the condensed images are initialized with Gaussian noise

# In[ ]:


def visualize_condensed_images(synthetic_images_tensor, num_classes, images_per_class, save_dir, title):
    plt.figure(figsize=(images_per_class * 2, num_classes * 2))
    
    for class_id in range(num_classes):
        class_images = synthetic_images_tensor[class_id * images_per_class: (class_id + 1) * images_per_class]
        for i, image in enumerate(class_images):
            plt.subplot(num_classes, images_per_class, class_id * images_per_class + i + 1)
            # Permute the tensor to match (H, W, C) format and convert to numpy
            img = image.permute(1, 2, 0).cpu().detach().numpy()
            plt.imshow(img) 
            plt.axis('off')
            
    plt.suptitle(title)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")  # Define the path where the image will be saved
    plt.savefig(save_path)
    plt.show()
    print(f"Image saved at: {save_path}")
    
# Visualize and save condensed images for MHIST with Gaussian noise
visualize_condensed_images(synthetic_images_tensor_mhist_noise, num_classes_mhist, images_per_class_mhist, 
                           save_dir="Save_images", title="Condensed Images for MHIST with Gaussian Noise")

"""
Part 1 Question 2(e):
Now that you have had a chance to understand, learn, and visualize the condensed dataset, we can train the selected network from scratch on the condensed images. Train the selected network on a learned synthetic dataset (with 100 training images), then evaluate it on the
real testing data. Compare the test accuracy performance and the training time with part 2a. Explain your results. (For a fair comparison, you should use the exact same experimental setting as part 2a)
"""

# In[ ]:


# Function to train the model on the synthetic condensed dataset
def train_model_on_synthetic(model, dataloader, optimizer, criterion, scheduler, num_epochs=20):
    model.train()  
    start_time = time.time()  

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model
            running_loss += loss.item()
        
        scheduler.step()  # Step the learning rate scheduler
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    
    total_training_time = time.time() - start_time
    print(f"Training Time on Condensed Dataset: {total_training_time:.2f} seconds")
    return total_training_time


# In[ ]:


# Training on the condensed synthetic dataset (Part 2e)
print("\nTraining ConvNet7 on Condensed Synthetic Dataset...")
training_time_condensed = train_model_on_synthetic(convnet7, synthetic_loader_mhist, optimizer7, criterion, scheduler7)
print(f"Total Training Time on Condensed Dataset: {training_time_condensed:.2f} seconds")

# Evaluate model on the real test data (MHIST test set). This is the same evaluation model 'ConvNet7' applied to MHIST data in 2(a)
accuracy_synthetic_mhist = evaluate_model(convnet7, mhist_test_loader)
print(f'ConvNet7 Test Accuracy on Real MHIST after training on condensed dataset: {accuracy_synthetic_mhist:.2f}%')


# In[ ]:


# Input tensor for calculating FLOPs
input_tensor_mhist = torch.randn(100, channel_mhist, 224, 224)
flops_mhist = calculate_flops(convnet7, input_tensor_mhist)
print(f'FLOPs for ConvNet7: {flops_mhist:.2f} FLOPs')

"""
Part 1 Question 3: Cross-architecture Generalization
The ResNet18 is used in this section from Resnet18_mhist.py file to evaluate its cross-architecture performance in terms of classification accuracy on the test sets.
"""

# In[ ]:


from Resnet18_mhist import ResNet18
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Define parameters
model_name = 'ResNet18' 
channel = 3              
num_classes = 2          
im_size = (224, 224)        

transform_resize = transforms.Compose([
    transforms.Resize(im_size),  
    transforms.ToTensor()  
])

# Get the network instance using the ResNet18 function
resnet18 = ResNet18(channel=channel, num_classes=num_classes)

# Prepare the condensed MHIST dataset
synthetic_dataset_mhist = TensorDataset(torch.stack(synthetic_images_mhist), torch.tensor(synthetic_labels_mhist))
synthetic_loader_mhist = DataLoader(synthetic_dataset_mhist, batch_size=batch_size_mhist, shuffle=True)

# Define optimizer, criterion, and scheduler
optimizer_mhist = torch.optim.Adam(resnet18.parameters(), lr=0.001)
criterion_mhist = torch.nn.CrossEntropyLoss()
scheduler_mhist = torch.optim.lr_scheduler.StepLR(optimizer_mhist, step_size=10, gamma=0.1)

# Train ResNet18 on the condensed MHIST dataset
print("\nTraining ResNet18 on Condensed MHIST Dataset...")
training_time_mhist = train_model_on_synthetic(resnet18, synthetic_loader_mhist, optimizer_mhist, criterion_mhist, scheduler_mhist)

# Evaluate on the real MHIST test set
mhist_test_loader = DataLoader(mhist_test, batch_size=batch_size_mhist, shuffle=False)
mhist_accuracy = evaluate_model(resnet18, mhist_test_loader)
print(f'ResNet18 Test Accuracy on Real MHIST: {mhist_accuracy:.2f}%')

"""
          ############################# Code ends here for MHIST Dataset  ####################################

"""




