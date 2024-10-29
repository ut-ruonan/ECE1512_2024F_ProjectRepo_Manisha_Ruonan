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

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import import_ipynb
from utils import get_network, get_time
from evaluate import evaluate_model, training_model
from resnet import ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""  ##### 2e) Train the model with condensed dataset"""


def get_test_data():
    test_folder = 'mhist_dataset/test'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return test_loader


def evaluate_synthetic_image(model, img_syn, labels_syn, test_loader):
    model.to(device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    criterion = torch.nn.CrossEntropyLoss()

    synthetic_dataset = TensorDataset(img_syn, labels_syn)
    train_loader = DataLoader(synthetic_dataset, batch_size=64, shuffle=True)

    training_model(model, optimizer, scheduler, criterion, train_loader)
    evaluate_model(model, test_loader)


test_loader = get_test_data()

synthetic_model = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
img_syn_loaded = torch.load('task2_results/PAD__synthetic_dataset.pt')
img_syn = img_syn_loaded['images']
labels_syn = img_syn_loaded['labels']

evaluate_synthetic_image(synthetic_model, img_syn, labels_syn)

random_model = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
random_img_syn_loaded = torch.load('mhist_result/PAD_Random_synthetic_dataset.pt')
random_img_syn = img_syn_loaded['images']
random_labels_syn = img_syn_loaded['labels']

evaluate_synthetic_image(random_model, random_img_syn, random_labels_syn, test_loader)

gaussian_model = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
gaussian_img_syn_loaded = torch.load('mhist_result/PAD_Gaussian_synthetic_dataset.pt')
gaussian_img_syn = img_syn_loaded['images']
gaussian_labels_syn = img_syn_loaded['labels']

evaluate_synthetic_image(gaussian_model, gaussian_img_syn, gaussian_labels_syn, test_loader)

"""#### 3 Cross-Architecture Generalization"""
synthetic_resnet = ResNet18(channel=3, num_classes=2)
evaluate_synthetic_image(synthetic_resnet, img_syn, labels_syn, test_loader)

random_resnet = ResNet18(channel=3, num_classes=2)
evaluate_synthetic_image(random_resnet, random_img_syn, random_labels_syn, test_loader)

gaussian_resnet = ResNet18(channel=3, num_classes=2)
evaluate_synthetic_image(gaussian_resnet, gaussian_img_syn, gaussian_labels_syn, test_loader)

