#!/usr/bin/env python
# coding: utf-8

"""
Header for MNIST_Task1_main.py
--------------------
This is the code for the "Task 1: Dataset Distillation with Attention Matching' Question 2. Dataset Distillation Learning - MNIST Dataset"

    ########################################### Code start here for MNIST Dataset ############################################
"""
# # Import Required Libraries

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
from utils import get_dataset, get_loops, get_network, get_eval_pool, evaluate_synset

import os
import time
import random
import thop
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from collections import Counter
import torchvision.transforms as transforms

device = torch.device('cpu')


# # Load datasets

# In[ ]:


data_path = 'C:/Users/mahagam3/Documents/ECE course/Project A'
channel, im_size, num_classes, class_names, mean, std, mnist_train, mnist_test, testloader = get_dataset('MNIST', data_path)

# print out some information about the datasets - MNIST
print(f"Channel: {channel}")
print(f"Image size: {im_size}")
print(f"Number of classes: {num_classes}")
print(f"Training dataset size: {len(mnist_train)}")
print(f"Test dataset size: {len(mnist_test)}")


# In[ ]:


###### MNIST Dataset ######
# Initialize a counter for the classes
class_counts = Counter()

# Iterate through the training dataset
for _, label in mnist_train:
    class_counts[label] += 1  # Increment the count for the corresponding class

# Print the number of images per class
for class_label, count in class_counts.items():
    print(f"Class {class_label}: {count} images")


# In[ ]:


# Set parameters for ConvNetD3
channel_mnist = 1  # Grayscale images (MNIST)
num_classes_mnist = 10  # MNIST has 10 classes
im_size_mnist = (32, 32)  # Actual MNIST image size is 28x28

# Resize MNIST images from 28x28 to 32x32
transform_resize = transforms.Compose([
    transforms.Resize(im_size_mnist),  # Resize to 32x32
    transforms.ToTensor()  # Convert to tensor
])

# Instantiate ConvNetD3
convnet3 = get_network('ConvNetD3', channel_mnist, num_classes_mnist, im_size_mnist).to('cpu')

optimizer3 = optim.SGD(convnet3.parameters(), lr=0.01) # Define optimizer
criterion = nn.CrossEntropyLoss() # Define loss function
mnist_loader = DataLoader(mnist_train, batch_size=64, shuffle=True) # Create DataLoaders
scheduler3 = CosineAnnealingLR(optimizer3, T_max=20)# Cosine Annealing Scheduler

"""
Part 1 Question 2(a):
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
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model
            running_loss += loss.item()
        
        scheduler.step()  # Step the learning rate scheduler
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

    total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MNIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time


# In[ ]:


# Training ConvNet3 on MNIST
print("Training ConvNet3 on MNIST...")
train_model(convnet3, mnist_loader, optimizer3, criterion, scheduler3)


# In[ ]:


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

# Evaluate ConvNet3 on test set
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
accuracy_mnist = evaluate_model(convnet3, mnist_test_loader)
print(f'ConvNet3 Test Accuracy on MNIST: {accuracy_mnist:.2f}%')


# In[ ]:


# Flop is used to measure the coputational cost of running the model and to evaluate the efficiency of the learning model.
# Calculate FLOPs (Floating Point Operations per Second)
def calculate_flops(model, input_tensor):
    from thop import profile
    flops, params = profile(model, inputs=(input_tensor,))
    return flops

# Input tensor for calculating FLOPs
input_tensor_mnist = torch.randn(60000, channel_mnist, 32, 32)
flops_mnist = calculate_flops(convnet3, input_tensor_mnist)
print(f'FLOPs for ConvNet3: {flops_mnist:.2f} FLOPs')

"""
Part 1: Question 2(b): 
Learn the synthetic dataset S using the selected model and Attention Matching algorithm. For initialization of condensed images, randomly select from real training images. The experimental setup can be found in Table 1.
"""

# In[ ]:


# Set hyperparameters for MNIST
K_mnist = 100
T_mnist = 10
eta_S_mnist = 0.1
zeta_S_mnist = 1
eta_theta_mnist = 0.01
zeta_theta_mnist = 50
lambda_mnist = 0.01
num_classes_mnist = 10
images_per_class_mnist = 10
batch_size_mnist = 256


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


# Example of generating synthetic samples for MNIST
synthetic_images_mnist, synthetic_labels_mnist = generate_synthetic_dataset(
    mnist_train, num_classes_mnist, images_per_class_mnist
)
# Convert synthetic images to a tensor
synthetic_images_tensor_mnist = torch.stack(synthetic_images_mnist).to(device)
synthetic_labels_tensor_mnist = torch.tensor(synthetic_labels_mnist).to(device)

# Create DataLoader for the synthetic MNIST dataset
synthetic_dataset_mnist = torch.utils.data.TensorDataset(synthetic_images_tensor_mnist, synthetic_labels_tensor_mnist)
synthetic_loader_mnist = DataLoader(synthetic_dataset_mnist, batch_size=batch_size_mnist, shuffle=True)


# In[ ]:


##### Save the synthetic images ########
# Save synthetic images as PNG files
output_dir = "synthetic_images_mnist"
os.makedirs(output_dir, exist_ok=True)

# Save each synthetic image as a PNG file
for idx, synthetic_image in enumerate(synthetic_images_mnist):
    image_path = os.path.join(output_dir, f"synthetic_image_{idx}.png")
    save_image(synthetic_image, image_path)

print(f"Synthetic images saved in {output_dir}")

# Save synthetic images and labels as a .pt file
torch.save({
    'images': synthetic_images_tensor_mnist,
    'labels': synthetic_labels_tensor_mnist
}, "synthetic_mnist.pt")

print("Synthetic images and labels saved as 'synthetic_mnist.pt'")


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
            
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the model
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(synthetic_dataloader):.4f}')
        total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MNIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time


import torch.nn.functional as F 
# Optimizer setup for the model (using SGD as per the table)
optimizer_mnist = torch.optim.SGD(convnet3.parameters(), lr=eta_theta_mnist)

# Train the model using the synthetic datasets, cross_entropy measure how well the model predicts the classes.
print("Training on MNIST dataset...")
train_with_attention_matching(convnet3, synthetic_loader_mnist, optimizer_mnist, F.cross_entropy, 
                              param=0, exp=4, num_epochs=T_mnist, lambda_param=lambda_mnist)

# accuracy_attention_mnist = evaluate_model(convnet3, mnist_test_loader)
# print(f'ConvNet7 Test Accuracy on MHIST: {accuracy_attention_mnist:.2f}%')


# # Part 1: Question 2(c):
# Provide the visualization of condensed images per class for both MNIST and MHIST datasets. 
# Do you think these condensed images are recognizable? Support your explanations.

# In[ ]:


def visualize_condensed_images(synthetic_images, num_classes, images_per_class, title, save_folder=None):
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)

    for class_id in range(num_classes):
        for img_id in range(images_per_class):
            idx = class_id * images_per_class + img_id
            axes[class_id, img_id].imshow(synthetic_images[idx].squeeze(), cmap='gray')
            axes[class_id, img_id].axis('off')

    plt.tight_layout()

    # Save the image if save_folder is provided
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")
        
    plt.show()

save_folder = 'Save_images'

# Visualize condensed images for MNIST
visualize_condensed_images(synthetic_images_tensor_mnist, num_classes_mnist, images_per_class_mnist, 
                           title="Condensed Images for MNIST", save_folder=save_folder)

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


# Generating synthetic samples for MNIST
synthetic_images_mnist_noise, synthetic_labels_mnist_noise = generate_synthetic_dataset_with_noise(
    mnist_train, num_classes_mnist, images_per_class_mnist
) 
# Convert synthetic images to a tensor
synthetic_images_tensor_mnist_noise = torch.stack(synthetic_images_mnist_noise).to(device)
synthetic_labels_tensor_mnist_noise = torch.tensor(synthetic_labels_mnist_noise).to(device)

# Create DataLoader for the synthetic MNIST dataset with noise
synthetic_dataset_mnist_noise = torch.utils.data.TensorDataset(synthetic_images_tensor_mnist_noise, synthetic_labels_tensor_mnist_noise)
synthetic_loader_mnist_noise = DataLoader(synthetic_dataset_mnist_noise, batch_size=batch_size_mnist, shuffle=True)


# In[ ]:


##### Save the synthetic images with noise ########
# Save synthetic images as PNG files
output_dir = "synthetic_images_mnist_noise"
os.makedirs(output_dir, exist_ok=True)

# Save each synthetic image as a PNG file
for idx, synthetic_image in enumerate(synthetic_images_mnist_noise):
    image_path = os.path.join(output_dir, f"synthetic_image_{idx}.png")
    save_image(synthetic_image, image_path)

print(f"Synthetic images saved in {output_dir}")

# Save synthetic images and labels as a .pt file
torch.save({
    'images': synthetic_images_tensor_mnist_noise,
    'labels': synthetic_labels_tensor_mnist_noise
}, "synthetic_mnist_noise.pt")

print("Synthetic images and labels saved as 'synthetic_mnist_noise.pt'")


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
            # Incorporate task balance parameter λ
            loss += lambda_param * torch.mean(attention_maps) 
            
            loss.backward()  
            optimizer.step()  
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(synthetic_dataloader):.4f}')
        total_training_time = time.time() - start_time  # Calculate total time
    print(f"Training Time on Real MNIST Dataset: {total_training_time:.2f} seconds")
    return total_training_time

# Optimizer setup for the model (using SGD as per the table)
optimizer_mnist_noise = torch.optim.SGD(convnet3.parameters(), lr=eta_theta_mnist)

# Train the model using the synthetic datasets with Gaussian noise
print("Training on MNIST dataset with Gaussian noise...")
train_with_attention_matching(convnet3, synthetic_loader_mnist_noise, optimizer_mnist_noise, 
                              F.cross_entropy, param=0, exp=4, num_epochs=T_mnist, lambda_param=lambda_mnist)

# accuracy_attention_mnist_noise = evaluate_model(convnet3, mnist_test_loader)
# print(f'ConvNet7 Test Accuracy on MHIST: {accuracy_attention_mnist_noise:.2f}%')


#### Repeat parts 2(c) while the condensed images are initialized with Gaussian noise

# In[ ]:


def visualize_condensed_images(synthetic_images_tensor, num_classes, images_per_class, save_dir="Save_images", title="Condensed Images"):
    plt.figure(figsize=(10, 10))
    
    for class_id in range(num_classes):
        class_images = synthetic_images_tensor[class_id * images_per_class:(class_id + 1) * images_per_class]
        for i, image in enumerate(class_images):
            plt.subplot(num_classes, images_per_class, class_id * images_per_class + i + 1)
            plt.imshow(image.cpu().detach().numpy().squeeze(), cmap='gray')
            plt.axis('off')
    
    plt.suptitle(title)
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png") # Define the path where the image will be saved
    plt.savefig(save_path)
    plt.show()
    print(f"Image saved at: {save_path}")
    
# Visualize and save condensed images for MNIST with Gaussian noise
visualize_condensed_images(synthetic_images_tensor_mnist_noise, num_classes_mnist, images_per_class_mnist, 
                           save_dir="Save_images", title="Condensed Images for MNIST with Gaussian Noise")

"""
Part 1 Question 2(e):
Now that you have had a chance to understand, learn, and visualize the condensed dataset, we can train the selected network from scratch on the condensed images. Train the selected network on a learned synthetic dataset (with 100 training images), then evaluate it on the real testing data. Compare the test accuracy performance and the training time with part 2a. Explain your results. (For a fair comparison, you should use the exact same experimental setting as part 2a)
"""

# In[ ]:


# Function to train the model on the synthetic condensed dataset
def train_model_on_synthetic(model, dataloader, optimizer, criterion, scheduler, num_epochs=20):
    model.train() 
    start_time = time.time()  

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad() 
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  
            running_loss += loss.item()
        
        scheduler.step()  # Step the learning rate scheduler
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
    
    total_training_time = time.time() - start_time
    print(f"Training Time on Condensed Dataset: {total_training_time:.2f} seconds")
    return total_training_time


# In[ ]:


# Training on the condensed synthetic dataset (Part 2e)
print("\nTraining ConvNet3 on Condensed Synthetic Dataset...")
training_time_condensed = train_model_on_synthetic(convnet3, synthetic_loader_mnist, optimizer3, criterion, scheduler3)
print(f"Total Training Time on Condensed Dataset: {training_time_condensed:.2f} seconds")

# Evaluate model on the real test data (MNIST test set). This is the same evaluation model 'ConvNet3' applied to MNIST data in 2(a)
accuracy_synthetic_mnist = evaluate_model(convnet3, mnist_test_loader)
print(f'ConvNet3 Test Accuracy on Real MNIST after training on condensed dataset: {accuracy_synthetic_mnist:.2f}%')


# In[ ]:


# Input tensor for calculating FLOPs
input_tensor_mnist = torch.randn(100, channel_mnist, 32, 32)
flops_mnist = calculate_flops(convnet3, input_tensor_mnist)
print(f'FLOPs for ConvNet3: {flops_mnist:.2f} FLOPs')

"""
Part 1 Question 3: Cross-architecture Generalization
The ResNet18 is used in this section from networks.py file to evaluate its cross-architecture performance in terms of classification accuracy on the test sets.
"""

# In[ ]:


##### MNIST dataset #####
# Define parameters
model_name = 'ResNet18'  
channel = 1              
num_classes = 10          
im_size = (32, 32)        

transform_resize = transforms.Compose([
    transforms.Resize(im_size), 
    transforms.ToTensor()  
])

# Get the network instance
resnet18 = get_network(model=model_name, channel=channel, num_classes=num_classes, im_size=im_size)

# Modify the first layer if necessary (for MNIST grayscale images)
if hasattr(resnet18, 'model'):
    resnet18.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Prepare the condensed MNIST dataset
# Synthetic_images and synthetic_labels are defined from part 2b
synthetic_dataset_mnist = TensorDataset(torch.stack(synthetic_images_mnist), torch.tensor(synthetic_labels_mnist))
synthetic_loader_mnist = DataLoader(synthetic_dataset_mnist, batch_size=batch_size_mnist, shuffle=True)

# Define optimizer, criterion, and scheduler
optimizer_mnist = torch.optim.Adam(resnet18.parameters(), lr=0.001)
criterion_mnist = torch.nn.CrossEntropyLoss()
scheduler_mnist = torch.optim.lr_scheduler.StepLR(optimizer_mnist, step_size=10, gamma=0.1)

# Train ResNet18 on the condensed MNIST dataset
print("\nTraining ResNet18 on Condensed MNIST Dataset...")
training_time_mnist = train_model_on_synthetic(resnet18, synthetic_loader_mnist, optimizer_mnist, criterion_mnist, scheduler_mnist)

# Evaluate on the real MNIST test set
mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)
mnist_accuracy = evaluate_model(resnet18, mnist_test_loader)
print(f'ResNet18 Test Accuracy on Real MNIST: {mnist_accuracy:.2f}%')

"""
          ############################# Code ends here for MNIST Dataset  ####################################
"""




