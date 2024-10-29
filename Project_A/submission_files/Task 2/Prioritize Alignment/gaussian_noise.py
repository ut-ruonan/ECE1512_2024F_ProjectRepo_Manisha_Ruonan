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

import random
import torch
import torch.optim as optim
from utils import get_network
from synthetic_generator import get_data, train_synthetic_images_with_trajectory, save_results

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""##### b) Learn the synthetic dataset with prioritize alignment

"""
num_classes = 2
num_images_per_class = 50
img_size = (3, 224, 224)


def generate_synthetic_dataset_with_noise(real_dataset, num_classes, images_per_class=50, K=200, eta_s=0.01,
                                          zeta_s=1, eta_theta=0.01, zeta_theta=50, minibatch_size=128, noise_std=0.8):
    synthetic_images = []
    synthetic_labels = []

    for class_id in range(num_classes):
        # Randomly sample 50 images from each class
        class_indices = [i for i, (_, label) in enumerate(real_dataset) if label == class_id]
        sampled_indices = random.sample(class_indices, images_per_class)

        for i in sampled_indices:
            img_real = real_dataset[i][0].to(device)

            noisy_versions = []
            for k_init in range(K):
                noise = torch.normal(mean=0, std=noise_std, size=img_real.shape, requires_grad=True).to(device)
                noisy_image = torch.clamp(img_real + noise, 0, 1).clone().detach().requires_grad_(True)
                noisy_versions.append(noisy_image)

            for k_init in range(K):
                noisy_image = noisy_versions[k_init]
                optimizer_condensed = optim.SGD([noisy_image], lr=eta_s)

                for step in range(zeta_s):
                    optimizer_condensed.zero_grad()
                    loss = torch.nn.functional.mse_loss(noisy_image, img_real)
                    loss.backward()
                    optimizer_condensed.step()

            selected_synthetic_image = torch.mean(torch.stack(noisy_versions), dim=0).detach()

            synthetic_images.append(selected_synthetic_image)
            synthetic_labels.append(real_dataset[i][1])

    img_syn = torch.stack(synthetic_images)
    labels_syn = torch.tensor(synthetic_labels, device=device)

    return img_syn, labels_syn


real_dataset, real_loader, teacher_trajectory, temp_net = get_data()

gaussian_img_syn, gaussian_labels_syn = generate_synthetic_dataset_with_noise(real_dataset, num_classes)

gaussian_synthetic_data, gaussian_labels_syn = train_synthetic_images_with_trajectory(temp_net,
                                                                                      teacher_trajectory,
                                                                                      gaussian_img_syn,
                                                                                      gaussian_labels_syn)
save_results(gaussian_synthetic_data, gaussian_labels_syn, 'Gaussian')

