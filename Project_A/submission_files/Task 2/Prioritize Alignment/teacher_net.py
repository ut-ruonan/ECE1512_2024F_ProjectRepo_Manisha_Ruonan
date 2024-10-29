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

import copy
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from utils import get_network, get_time
from evaluate import evaluate_model
from ptflops import get_model_complexity_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_epochs = 20
add_end_epoch = 7
rm_epoch_first = 12
rm_epoch_second = 17

initial_ratio = 0.75
rm_easy_ratio_first = 0.5
rm_easy_ratio_second = 0.3

batch_size = 64


def get_train_and_test():
    train_folder = 'mhist_dataset/train'
    test_folder = 'mhist_dataset/test'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Expected dataset size: {len(train_dataset)}")

    return train_dataset, test_dataset, train_loader, test_loader


def calculate_el2n_scores(model, dataloader, device):
    model.eval()
    el2n_scores = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            el2n = (probs - F.one_hot(labels, num_classes=probs.size(1))).pow(2).sum(dim=1)
            batch_start = batch_idx * dataloader.batch_size
            for idx, score in enumerate(el2n):
                dataset_idx = batch_start + idx
                el2n_scores.append((dataset_idx, score.item()))

    return el2n_scores


"""###### train teacher model

"""


def linear_scheduler(t, a, b):
    return min(1, 1.0 * a + (1 - a) * 1.0 * t / b)


def get_training_indices(sorted_indices, epoch, add_end_epoch, rm_epoch_first, rm_epoch_second,
                         initial_ratio=0.75, rm_easy_ratio_first=0.5, rm_easy_ratio_second=0.3):
    if epoch <= add_end_epoch:
        current_ratio = linear_scheduler(epoch, initial_ratio, add_end_epoch)
        num_samples = int(current_ratio * len(sorted_indices))
        selected_indices = sorted_indices[:num_samples]
    elif add_end_epoch < epoch <= rm_epoch_first:
        current_ratio = 1.0
        num_samples = int(current_ratio * len(sorted_indices))
        selected_indices = sorted_indices[:num_samples]
    elif rm_epoch_first < epoch <= rm_epoch_second:
        current_ratio = rm_easy_ratio_first
        num_samples = int(current_ratio * len(sorted_indices))
        selected_indices = sorted_indices[-num_samples:]
    else:
        current_ratio = rm_easy_ratio_second
        num_samples = int(current_ratio * len(sorted_indices))
        selected_indices = sorted_indices[-num_samples:]

    return selected_indices


def training_teacher_model(model, optimizer, criterion):
    torch.cuda.empty_cache()

    teacher_trajectory = []
    teacher_trajectory.append(copy.deepcopy(model.cpu().state_dict()))
    model.to(device)
    for epoch in range(train_epochs):
        selected_indices = get_training_indices(sorted_indices, epoch,
                                                add_end_epoch, rm_epoch_first,
                                                rm_epoch_second)
        subset_dataset = Subset(train_dataset, selected_indices)
        sub_train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in sub_train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            torch.cuda.empty_cache()
        teacher_trajectory.append(copy.deepcopy(model.cpu().state_dict()))
        model.to(device)

        train_accuracy = 100. * correct / total
        print(
            f"Epoch [{epoch + 1}/{train_epochs}], Loss: {running_loss / len(train_loader.dataset)}, Accuracy: {train_accuracy:.2f}%")

    return teacher_trajectory


def get_flop(teacher_net, test_loader):
    total_flops = 0

    for inputs, labels in test_loader:
        total_flops += get_model_complexity_info(teacher_net, (3, 224, 224), as_strings=False)[0]

    print(f"Total FLOPs for the test dataset: {total_flops}")


train_dataset, test_dataset, train_loader, test_loader = get_train_and_test()

model_path = 'models/mhist_original.pth'
model = get_network(model='ConvNetD7', channel=3, num_classes=2, im_size=(224, 224))
model.load_state_dict(torch.load(model_path))
evaluate_model(model, test_loader)

el2n_scores = calculate_el2n_scores(model, train_loader, device)
el2n_scores_sorted = sorted(el2n_scores, key=lambda x: x[1])
sorted_indices = [idx for idx, _ in el2n_scores_sorted]

teacher_net = get_network(model='ConvNetD7', channel=3,
                          num_classes=2, im_size=(224, 224))
optimizer = SGD(teacher_net.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

teacher_trajectory = training_teacher_model(teacher_net, optimizer, criterion)

model_path = 'models/teacher_net.pth'
torch.save(teacher_net.cpu().state_dict(), model_path)

trajectory_path = 'task2_results/teacher_trajectory.pth'
torch.save(teacher_trajectory, trajectory_path)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
evaluate_model(teacher_net, test_loader)
get_flop(teacher_net, test_loader)
