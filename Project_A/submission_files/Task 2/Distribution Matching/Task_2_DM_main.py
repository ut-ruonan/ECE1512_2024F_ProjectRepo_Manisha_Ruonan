#!/usr/bin/env python
# coding: utf-8
"""
Header for Task2_DM.py
--------------------
"Task 2: Apply Distribution/feature matching method to selected architecture in part 2 on the MNIST dataset."

This code provided, the distribution matching is implemented within the training loop, specifically in the section where the synthetic data is trained against the real data.

The paper refered to this code is: *"B. Zhao and H. Bilen, “Dataset condensation with distribution matching,” in 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023, pp. 6503–6512."*
Reference code: https://github.com/VICO-UoE/DatasetCondensation/blob/master/main_DM.py
"""

# In[ ]:


import sys
sys.path.append('C:\\Users\\manis\\Documents\\ECE course\\Project A')
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils_Task2_DM import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
from networks import ConvNet


device = torch.device('cpu')

"""

      ################# The code from here just use ConvnetD3 for MNIST as in the Project A Task 2(a) ###########################

"""
"""
 With ConvNetD3
"""
# In[ ]:


def main():
    if 'ipykernel' in sys.modules:
        print("Running in a Jupyter environment. Overriding sys.argv.")
        sys.argv = [''] 
        
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
        parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')  # for MNIST it's 10. For MHIST, it's 50
        parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode')  # S: same as training model
        parser.add_argument('--num_exp', type=int, default=5, help='number of experiments')
        parser.add_argument('--num_eval', type=int, default=20, help='number of evaluating models')
        parser.add_argument('--epoch_eval_train', type=int, default=20, help='epochs for model training with synthetic data')#1000
        parser.add_argument('--Iteration', type=int, default=10, help='training iterations') #20000, update to 10 steps: suggested in Assignment
        parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for synthetic images') # 1.0
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='real', help='initialize synthetic images from noise or real')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

        args = parser.parse_args()
        args.method = 'DM'
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

        # Creating necessary directories if not present
        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        # Set evaluation iterations
        eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
        print('eval_it_pool: ', eval_it_pool)

        ''' Load the MNIST dataset '''
        transform = transforms.Compose([transforms.ToTensor()])
        channel = 1
        im_size = (28, 28) # im_size = (32, 32) 
        num_classes = 10
        mean = (0.1307,)
        std = (0.3081,)

        dst_train = MNIST(args.data_path, train=True, download=True, transform=transform)
        dst_test = MNIST(args.data_path, train=False, download=True, transform=transform)
        testloader = DataLoader(dst_test, batch_size=args.batch_real, shuffle=False)

        ''' Convnet3 for MNIST dataset '''
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
    mnist_loader = DataLoader(dst_train, batch_size=256, shuffle=True) # Create DataLoaders
    scheduler3 = CosineAnnealingLR(optimizer3, T_max=20)# Cosine Annealing Scheduler

    # Create a pool of models for evaluation
    model_eval_pool = get_eval_pool(args.eval_mode, 'ConvNetD3', 'ConvNetD3')
    accs_all_exps = dict()  # Record performance of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' Organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        # Class-wise image indices
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        
        # Display class distribution
        for c in range(num_classes):
            print(f'class {c} = {len(indices_class[c])} real images')
        
        # Helper function to fetch random images from a class
        def get_images(c, n):  # Get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        ''' Initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)
        
        # Initialize synthetic images with real or random noise
        if args.init == 'real':
            print('Initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('Initialize synthetic data from random noise')

        ''' Training '''
        optimizer_img = torch.optim.SGD(convnet3.parameters(), lr=0.01)# optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        print('%s training begins' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        for it in range(args.Iteration + 1):
                ''' Evaluate synthetic data '''
                if it in eval_it_pool:
                    for model_eval in model_eval_pool:
                        print(f'-------------------------\nEvaluation\nmodel_train = {'ConvNetD3'}, model_eval = {model_eval}, iteration = {it}')
                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            accs.append(acc_test)
                        print(f'Evaluate {len(accs)} random {model_eval}, mean = {np.mean(accs):.4f} std = {np.std(accs):.4f}\n-------------------------')

                        if it == args.Iteration:
                            accs_all_exps[model_eval] += accs

                    save_name = os.path.join(args.save_path, f'vis_{args.method}_{args.dataset}_{'ConvNetD3'}_{args.ipc}ipc_exp{exp}_iter{it}.png') 
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                    # image_syn_vis[image_syn_vis < 0] = 0.0
                    # image_syn_vis[image_syn_vis > 1] = 1.0
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    image_syn_vis = (image_syn_vis - image_syn_vis.min()) / (image_syn_vis.max() - image_syn_vis.min())
                    save_image(image_syn_vis, save_name, nrow=args.ipc)

                ''' Train synthetic data '''
                net = get_network('ConvNetD3', channel, num_classes, im_size).to(args.device)
                net.train()
                for param in net.parameters():
                    param.requires_grad = False

                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

                loss_avg = 0
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    # Calculate the distribution matching loss
                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if it % 10 == 0:
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} iter = {it:05d}, loss = {loss_avg:.4f}')

                if it == args.Iteration:
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])

        print('%s training ends' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    accs = []
    for key in model_eval_pool:
        accs.append(np.mean(accs_all_exps[key]))
    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'
          % (args.num_exp, 'ConvNetD3', np.mean(accs), np.std(accs))) # % (args.num_exp, args.model, np.mean(accs), np.std(accs)))
    
    current_dir = os.getcwd()# Get the current working directory
    # Save the results in the same directory as the script
    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps}, 
                os.path.join(current_dir, f'res_{args.method}_{args.dataset}_convnet3_{args.ipc}ipc.pt'))
    
    print(f'Data saved to {os.path.join(current_dir, f"res_{args.method}_{args.dataset}_convnet3_{args.ipc}ipc.pt")}')

    return 

if __name__ == '__main__':
     main()


"""
With Restnet18
"""
# In[ ]:


def main():
    # Check if running inside a Jupyter Notebook environment
    if 'ipykernel' in sys.modules:
        print("Running in a Jupyter environment. Overriding sys.argv.")
        sys.argv = ['']  # Clear sys.argv to avoid passing unwanted arguments

        parser = argparse.ArgumentParser(description='Parameter Processing')
        parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
        parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')  # for MNIST it's 10. For MHIST, it's 50
        parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode')  # S: same as training model
        parser.add_argument('--num_exp', type=int, default=5, help='number of experiments')
        parser.add_argument('--num_eval', type=int, default=20, help='number of evaluating models')
        parser.add_argument('--epoch_eval_train', type=int, default=20, help='epochs for model training with synthetic data')#1000
        parser.add_argument('--Iteration', type=int, default=10, help='training iterations') #20000, update to 10 steps: suggested in Assignment
        parser.add_argument('--lr_img', type=float, default=0.01, help='learning rate for synthetic images') # 1.0
        parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for network parameters')
        parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
        parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
        parser.add_argument('--init', type=str, default='real', help='initialize synthetic images from noise or real')
        parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
        parser.add_argument('--data_path', type=str, default='data', help='dataset path')
        parser.add_argument('--save_path', type=str, default='result', help='path to save results')
        parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

        args = parser.parse_args()
        args.method = 'DM'
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.dsa_param = ParamDiffAug()
        args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

        # Creating necessary directories if not present
        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        # Set evaluation iterations
        eval_it_pool = np.arange(0, args.Iteration + 1, 500).tolist() if args.eval_mode in ['S', 'SS'] else [args.Iteration]
        print('eval_it_pool: ', eval_it_pool)

        ''' Load the MNIST dataset '''
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor()
        ])
        
        channel = 1
        # im_size = (28, 28) # im_size = (32, 32) 
        num_classes = 10
        mean = (0.1307,)
        std = (0.3081,)

        dst_train = MNIST(args.data_path, train=True, download=True, transform=transform)
        dst_test = MNIST(args.data_path, train=False, download=True, transform=transform)
        testloader = DataLoader(dst_test, batch_size=args.batch_real, shuffle=False)


        ''' ResNet18 for MNIST dataset '''
        # Set parameters for ResNet18
        model_name = 'ResNet18'  
        channel = 1  # Grayscale images (MNIST)
        num_classes = 10  # MNIST has 10 classes
        im_size = (32, 32)  # Actual MNIST image size is 28x28
        
        # Resize MNIST images from 28x28 to 32x32
        transform_resize = transforms.Compose([
            transforms.Resize(im_size),  # Resize to 32x32
            transforms.ToTensor()  # Convert to tensor
        ])
    
        # Get the network instance
        resnet18 = get_network(model=model_name, channel=channel, num_classes=num_classes, im_size=im_size)
        
        # Modify the first layer if necessary (for MNIST grayscale images)
        if hasattr(resnet18, 'model'):
            resnet18.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    optimizer3 = optim.SGD(resnet18.parameters(), lr=0.01) # Define optimizer
    criterion = nn.CrossEntropyLoss() # Define loss function
    mnist_loader = DataLoader(dst_train, batch_size=256, shuffle=True) # Create DataLoaders
    scheduler3 = CosineAnnealingLR(optimizer3, T_max=20)# Cosine Annealing Scheduler

    # Create a pool of models for evaluation
    model_eval_pool = get_eval_pool(args.eval_mode, 'ResNet18', 'ResNet18')
    accs_all_exps = dict()  # Record performance of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print(f'\n================== Exp {exp} ==================\n')
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' Organize the real dataset '''
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        
        # Class-wise image indices
        indices_class = [[] for _ in range(num_classes)]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        
        # Display class distribution
        for c in range(num_classes):
            print(f'class {c} = {len(indices_class[c])} real images')
        
        # Helper function to fetch random images from a class
        def get_images(c, n):  # Get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        ''' Initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

        # Initialize synthetic images with real or random noise
        if args.init == 'real':
            print('Initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
        else:
            print('Initialize synthetic data from random noise')

        ''' Training '''
        optimizer_img = torch.optim.SGD(resnet18.parameters(), lr=0.01)# optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_img.zero_grad()
        print('%s training begins' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        for it in range(args.Iteration + 1):
                ''' Evaluate synthetic data '''
                if it in eval_it_pool:
                    for model_eval in model_eval_pool:
                        print(f'-------------------------\nEvaluation\nmodel_train = {'ResNet18'}, model_eval = {model_eval}, iteration = {it}')
                        accs = []
                        for it_eval in range(args.num_eval):
                            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
                            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
                            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                            accs.append(acc_test)
                        print(f'Evaluate {len(accs)} random {model_eval}, mean = {np.mean(accs):.4f} std = {np.std(accs):.4f}\n-------------------------')

                        if it == args.Iteration:
                            accs_all_exps[model_eval] += accs

                    save_name = os.path.join(args.save_path, f'vis_{args.method}_{args.dataset}_{'ResNet18'}_{args.ipc}ipc_exp{exp}_iter{it}.png') 
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    for ch in range(channel):
                        image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                    # image_syn_vis[image_syn_vis < 0] = 0.0
                    # image_syn_vis[image_syn_vis > 1] = 1.0
                    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                    image_syn_vis = (image_syn_vis - image_syn_vis.min()) / (image_syn_vis.max() - image_syn_vis.min())
                    save_image(image_syn_vis, save_name, nrow=args.ipc)

                ''' Train synthetic data '''
                net = get_network('ResNet18', channel, num_classes, im_size).to(args.device)
                net.train()
                for param in net.parameters():
                    param.requires_grad = False

                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed

                loss_avg = 0
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                    output_real = embed(img_real).detach()
                    output_syn = embed(img_syn)

                    # Calculate the distribution matching loss
                    loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if it % 10 == 0:
                    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} iter = {it:05d}, loss = {loss_avg:.4f}')

                if it == args.Iteration:
                    data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])

        print('%s training ends' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    accs = []
    for key in model_eval_pool:
        accs.append(np.mean(accs_all_exps[key]))
    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'
          % (args.num_exp, 'ResNet18', np.mean(accs), np.std(accs))) # % (args.num_exp, args.model, np.mean(accs), np.std(accs)))
    
    current_dir = os.getcwd()# Get the current working directory
    # Save the results in the same directory as the script
    torch.save({'data': data_save, 'accs_all_exps': accs_all_exps}, 
                os.path.join(current_dir, f'res_{args.method}_{args.dataset}_resnet18_{args.ipc}ipc.pt'))
    
    print(f'Data saved to {os.path.join(current_dir, f"res_{args.method}_{args.dataset}_resnet18_{args.ipc}ipc.pt")}')

    return 

if __name__ == '__main__':
     main()

"""
      ####################################### End of the code for Project A Task 2(a) ########################################
"""

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------
This code provides:
1. Learning the Condensed Images Using Distribution/Feature Matching:

Feature Matching Surrogate Objective: The code includes a section where real and synthetic images' feature embeddings are compared for each class. The embeddings from real and synthetic data are obtained via the embed method of the network model (denoted by output_real for real data and output_syn for synthetic data).
 
Distribution Matching Loss: The loss function, defined as the squared difference between the mean embeddings of real and synthetic images, represents a form of distribution matching. It aims to make the synthetic data resemble real data by minimizing the feature distribution distance between real and synthetic images: ---> 
 {"loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)"}
 
Optimization: The SGD optimizer (optimizer_img) updates the synthetic images to reduce the distribution matching loss, refining these images iteratively to better match real data features.
---------------------------------------------------------------------------------------------------------------------------------------------------------------
2. Training the Network from Scratch on Condensed Images and Evaluating on Real Test Data:
 
Training from Scratch: In each evaluation iteration (for-loop with it in eval_it_pool), a new network (net_eval) is initialized and trained from scratch using the condensed images (image_syn). This network is specifically trained only on synthetic images:

{"net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)"}
{"image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()))"}
{"_, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args))"}

Ealuation on Real Test Data: After training, the network is evaluated on real test data (testloader). The evaluate_synset function calculates and logs the test accuracy, which measures how well the network generalizes from condensed (synthetic) training data to real test data.

---------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
