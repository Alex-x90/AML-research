import os
import torch
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilons = [.05, .1, .15, .2, .25, .3]
alpha=1e-2
num_iter=40
modelName = "lenet_mnist_model.pth"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def pgd_linf(model, x, y, eps, alpha, num_iter, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    y = y.to(device)
    for i in range(num_iter):
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(_x_adv), y)
        loss.backward()

        with torch.no_grad():
            x_adv += _x_adv.grad.sign() * alpha

        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(0,1)

    return x_adv.detach()

def attackTest(dataloader, model, loss_fn, epsilon):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    correct = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        x_adv = pgd_linf(model, X, y, epsilon, alpha, num_iter, loss_fn)
        with torch.no_grad():
            originalPred = model(X)
            pred = model(x_adv)
            correct += (pred.argmax(1) == originalPred.argmax(1)).type(torch.float).sum().item()

    final_acc = correct/float(len(dataloader.dataset))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(dataloader.dataset), final_acc))
    return final_acc

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

loss_fn = nn.CrossEntropyLoss()

test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load(modelName))
model.eval()

accuracies = [1]
examples = []

for epsilon in epsilons:
    acc = attackTest(test_loader, model, loss_fn, epsilon)
    accuracies.append(acc)

epsilons.insert(0, 0)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
