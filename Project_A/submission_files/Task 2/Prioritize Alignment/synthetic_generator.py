import nbformat
from IPython.core.interactiveshell import InteractiveShell


def load_ipynb_as_module(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(cell.source, globals())


load_ipynb_as_module('../../utils.ipynb')

import os
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import get_network, get_time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""##### b) Learn the synthetic dataset with prioritize alignment

"""
num_classes = 2
num_images_per_class = 50
img_size = (3, 224, 224)


def get_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    real_dataset = datasets.ImageFolder(root='mhist_dataset/train', transform=transform)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=64, shuffle=True)

    trajectory_path = 'task2_results/teacher_trajectory.pth'
    teacher_trajectory = torch.load(trajectory_path, map_location=device)

    model_path = 'models/teacher_net.pth'
    temp_net = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
    temp_net.load_state_dict(torch.load(model_path, map_location=device))

    return real_dataset, real_loader, teacher_trajectory, temp_net


def generate_synthetic_dataset(real_dataset, num_classes=2, images_per_class=50, K=200, eta_s=0.1,
                               zeta_s=1, eta_theta=0.01, zeta_theta=50, minibatch_size=128):
    synthetic_images = []
    synthetic_labels = []

    for class_id in range(num_classes):

        class_indices = [i for i, (_, label) in enumerate(real_dataset) if label == class_id]
        sampled_indices = random.sample(class_indices, images_per_class)

        for i in sampled_indices:
            img_real = real_dataset[i][0].to(device)

            synthetic_versions = []
            for _ in range(K):
                synthetic_image = img_real.clone().detach().requires_grad_(True)
                synthetic_versions.append(synthetic_image)

            for synthetic_image in synthetic_versions:
                optimizer_condensed = optim.SGD([synthetic_image], lr=eta_s)

                for _ in range(zeta_s):
                    optimizer_condensed.zero_grad()
                    loss = torch.nn.functional.mse_loss(synthetic_image, img_real)
                    loss.backward()
                    optimizer_condensed.step()

            final_synthetic_image = torch.mean(torch.stack(synthetic_versions), dim=0).detach()

            synthetic_images.append(final_synthetic_image)
            synthetic_labels.append(class_id)

    img_syn = torch.stack(synthetic_images)
    labels_syn = torch.tensor(synthetic_labels, device=device)

    return img_syn, labels_syn


def match_with_deep_layers_only(agent_model, target_params, synthetic_data, match_layers_threshold=0.5):
    model_params = list(agent_model.parameters())

    num_layers = len(model_params)
    num_shallow_layers = int(match_layers_threshold * num_layers)

    deep_layers_params = model_params[num_shallow_layers:]

    target_params_list = [param.detach().to(deep_layers_params[0].device)
                          for _, param in target_params.items()][num_shallow_layers:]

    loss = sum(F.mse_loss(param, target_param)
               for param, target_param in zip(deep_layers_params, target_params_list))

    return loss


def train_synthetic_images_with_trajectory(
        agent_model,
        teacher_trajectory,
        synthetic_data,
        labels_syn,
        num_iterations=10,
        eta_s=0.1,
        match_layers_threshold=0.5,
):
    synthetic_data.requires_grad_(True)
    optimizer_img = optim.SGD([synthetic_data], lr=eta_s)
    model_params = list(agent_model.parameters())

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        start_idx = torch.randint(0, len(teacher_trajectory) - 1, (1,)).item()
        target_params = teacher_trajectory[start_idx + 1]

        agent_model.train()
        for _ in range(5):
            output = agent_model(synthetic_data)
            loss = F.cross_entropy(output, labels_syn)
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()

        match_loss = match_with_deep_layers_only(agent_model, target_params, synthetic_data, match_layers_threshold)

        optimizer_img.zero_grad()
        match_loss.backward()
        optimizer_img.step()

        print(f"Iteration {iteration + 1}, Match Loss: {match_loss.item():.4f}")

    print("Synthetic image generation with deep-layer alignment completed.")
    return synthetic_data, labels_syn


def save_results(img_syn, labels_syn, noise_type=''):
    save_path = f"mhist_result/PAD_{noise_type}_synthetic_dataset.pt"

    torch.save({'images': img_syn, 'labels': labels_syn}, save_path)

    print(f"Synthetic dataset saved to {save_path}")


def display_image_grid(img_syn, labels_syn, noise_type):
    # Create directory to save grid images
    save_dir = f"mhist_result/PAD_{noise_type}_grid"
    os.makedirs(save_dir, exist_ok=True)

    num_samples_per_class = 50

    class_dict = {}

    for img, label in zip(img_syn, labels_syn):
        label = label.item()
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(img)

    for class_label, images in class_dict.items():
        selected_images = images[:num_samples_per_class]

        fig, axs = plt.subplots(5, 10, figsize=(15, 7))
        fig.suptitle(f"Class {class_label} - {noise_type}", fontsize=16)
        fig.patch.set_facecolor('white')

        for idx, img in enumerate(selected_images):
            img_np = img.permute(1, 2, 0).detach().cpu().numpy()
            img_np = img_np - img_np.min()  # Normalize to 0
            img_np = img_np / img_np.max()  # Normalize to 1
            axs[idx // 10, idx % 10].imshow(img_np)
            axs[idx // 10, idx % 10].axis('off')
            axs[idx // 10, idx % 10].set_facecolor('white')

        plt.show()

        grid_save_path = os.path.join(save_dir, f"class_{class_label}_grid.png")
        fig.savefig(grid_save_path, facecolor='white')
        plt.close()

    print(f"Image grids saved to {save_dir}")


real_dataset, real_loader, teacher_trajectory, temp_net = get_data()
img_syn, labels_syn = generate_synthetic_dataset(real_dataset)

synthetic_data, labels_syn = train_synthetic_images_with_trajectory(temp_net,
                                                                    teacher_trajectory,
                                                                    img_syn,
                                                                    labels_syn)

save_results(synthetic_data, labels_syn)
display_image_grid(synthetic_data, labels_syn, noise_type='original')
