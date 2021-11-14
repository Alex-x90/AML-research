import os
import torch
import numpy as np
import torch.onnx as onnx
import torchvision.models as models
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilon = .3
alpha=1e-2
num_iter=40

modelName = "robust_model_weights.pth"

learning_rate = 1e-3
epochs = 100

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

#TODO: finish converting train/test w/ pgd
#defines training loop for NN
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x_adv = pgd_linf(model, X, y, epsilon, alpha, num_iter, loss_fn)
        pred = model(x_adv)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Download/create training/test data from MNIST dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

train_dataloader = DataLoader(training_data, batch_size=1000, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load(modelName))

# Initialize the loss/optimizer function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

torch.save(model.state_dict(), modelName)

print("Done!")
