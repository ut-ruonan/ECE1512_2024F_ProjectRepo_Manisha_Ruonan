# Import the function to load notebooks
import nbformat
from IPython.core.interactiveshell import InteractiveShell


def load_ipynb_as_module(notebook_path):
    """Load and execute a Jupyter Notebook as a module."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            exec(cell.source, globals())  # Execute the notebook's code in the global namespace


# Load the notebook
load_ipynb_as_module('utils.ipynb')

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 2b
# set up the synthetic dataset
num_classes = 2
num_images_per_class = 50
img_size = (3, 224, 224)

# step 4: real dataset loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
real_dataset = datasets.ImageFolder(root='mhist_dataset/train', transform=transform)
real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=128, shuffle=True)


def generate_synthetic_dataset_with_noise(real_dataset, num_classes, images_per_class=50, noise_std=0.8):
    synthetic_images = []
    synthetic_labels = []

    for class_id in range(num_classes):
        # Randomly sample 50 images from each class
        indices = random.sample(
            [i for i, (_, label) in enumerate(real_dataset) if label == class_id],
            images_per_class
        )

        for i in indices:
            img_real = real_dataset[i][0]
            noise = torch.normal(mean=0, std=noise_std, size=img_real.shape)

            noise = noise.to(img_real.device)

            synthetic_image = img_real + noise

            synthetic_image = torch.clamp(synthetic_image, 0, 1)

            synthetic_images.append(synthetic_image)
            synthetic_labels.append(real_dataset[i][1])

    # Stack images to return a tensor for training
    img_syn = torch.stack(synthetic_images)
    labels_syn = torch.tensor(synthetic_labels)

    return img_syn, labels_syn


img_syn, _ = generate_synthetic_dataset_with_noise(real_dataset, num_classes)
# step 2: optimizer
img_syn = torch.nn.Parameter(img_syn)
optimizer_img = optim.SGD([img_syn], lr=0.1)  # lr is eta_s
# step 3: set up model - ConvNET - 7 in this case
# freeze the model's weights so that only the synthetic dataset is updated

model_path = 'models/mhist_original.pth'
net = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
net.load_state_dict(torch.load(model_path))
net.train()

for param in list(net.parameters()):
    param.requires_grad = False


# step 5: hook
# Using hook to extract the activations from the layers (attention maps) to compare the attention maps from real to synthetic
activations = {}


def get_activation(name):
    def hook_func(m, inp, op):
        activations[name] = op.detach()

    return hook_func


''' Defining the Refresh Function to store Activations and reset Collection '''


def refresh_activations(activations):
    model_set_activations = []  # Jagged Tensor Creation
    for i in activations.keys():
        model_set_activations.append(activations[i])
    activations = {}
    return activations, model_set_activations


def delete_hooks(hooks):
    for i in hooks:
        i.remove()
    return


def attach_hooks(net):
    hooks = []
    for name, module in net.named_modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(name)))
    return hooks


# step 6: Attention Matching Map
def get_attention(feature_set, param=0, exp=4, norm='l2'):
    if param == 0:
        attention_map = torch.sum(torch.abs(feature_set), dim=1)

    elif param == 1:
        attention_map = torch.sum(torch.abs(feature_set) ** exp, dim=1)

    elif param == 2:
        attention_map = torch.max(torch.abs(feature_set) ** exp, dim=1)

    if norm == 'l2':
        # Dimension: [B x (H*W)] -- Vectorized
        vectorized_attention_map = attention_map.view(feature_set.size(0), -1)
        normalized_attention_maps = F.normalize(vectorized_attention_map, p=2, dim=1)

    return normalized_attention_maps


# step 7: error function
def error(real, syn, err_type="MSE"):
    if err_type == "MSE":
        err = torch.sum((torch.mean(real, dim=0) - torch.mean(syn, dim=0)) ** 2)
    elif err_type == "MAE":
        err = torch.sum(torch.abs(torch.mean(real, dim=0) - torch.mean(syn, dim=0)))
    elif err_type == "MSE_B":
        err = torch.sum(
            (torch.mean(real.reshape(2, -1).cpu(), dim=1) - torch.mean(syn.reshape(2, -1).cpu(), dim=1)) ** 2)
    else:
        raise ValueError("Invalid error type")
    return err


# step 8: training loop
def train_dataset(img_syn, activations={}):
    num_iterations = 200
    learning_rate_model = 0.01

    losses = []
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")

        images_syn_all = []
        images_real_all = []
        for c in range(num_classes):
            img_real, _ = next(iter(real_loader))
            img_syn_per_class = img_syn[c * num_images_per_class:(c + 1) * num_images_per_class]

            images_real_all.append(img_real)
            images_syn_all.append(img_syn_per_class)

        images_real_all = torch.cat(images_real_all, dim=0)
        images_syn_all = torch.cat(images_syn_all, dim=0)

        net.train()
        hooks = attach_hooks(net)

        output_real = net(images_real_all)[0]
        activations, original_model_set_activations = refresh_activations(activations)

        output_syn = net(images_syn_all)[0]
        activations, syn_model_set_activations = refresh_activations(activations)
        delete_hooks(hooks)

        length_of_network = len(original_model_set_activations)

        loss = torch.tensor(0.0)
        mid_loss = 0
        out_loss = 0
        loss_avg = 0

        for layer in range(length_of_network - 1):
            real_attention = get_attention(original_model_set_activations[layer], param=1, exp=1, norm='l2')
            syn_attention = get_attention(syn_model_set_activations[layer], param=1, exp=1, norm='l2')

            tl = 100 * error(real_attention, syn_attention, err_type="MSE_B")
            loss += tl
            mid_loss += tl.item()

        output_loss = 100 * 0.01 * error(output_real, output_syn, err_type="MSE_B")
        loss += output_loss
        out_loss += output_loss.item()

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()
        torch.cuda.empty_cache()

        loss_avg /= (num_classes)
        out_loss /= (num_classes)
        mid_loss /= (num_classes)
        losses.append((loss_avg, out_loss, mid_loss))
        if iteration % 10 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), iteration, loss_avg))

    print("training completed.")
    return img_syn, losses


def save_results(img_syn, losses, noise_type):
    save_path = f"mhist_result/{noise_type}_synthetic_dataset.pt"

    torch.save(img_syn, save_path)

    print(f"Synthetic dataset saved to {save_path}")

    loss_log_path = f"mhist_result/{noise_type}_training_losses.txt"
    with open(loss_log_path, "w") as f:
        for epoch, loss in enumerate(losses):
            f.write(f"Iteration {epoch}: Loss = {loss}\n")

    print(f"Training losses saved to {loss_log_path}")


img_syn, losses = train_dataset(img_syn)
save_results(img_syn, losses, 'Gaussian')
