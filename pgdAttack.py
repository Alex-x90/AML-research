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

import time

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilons = [.05, .1, .15, .2, .25, .3] #[0, .05, .1, .15, .2, .25, .3]
alpha=1e-2
beta=0.5
num_iter=40
modelName = "robust_model_weights.pth"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 4))
        x = x.view(-1, 256)
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

def pgd_linf_wip(model, x, y, eps, alpha, beta, loss_fn, max_iter):
    x_adv = x.clone().detach().requires_grad_(True).to(device)
    y = y.to(device)
    iters=0

    while iters<max_iter: #difference between new iteration and previous iteration is too small:
        iters+=1

        # increase alpha by a step backwards to help go over curves. this could be a complete reset, but that might increase necessary calculation
        alpha *= 1.0/beta

        x_adv_new = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(x_adv_new), y)
        loss.backward()

        with torch.no_grad():
            grad = x_adv_new.grad      		    # should this not have .sign()?
        x_adv_new = x_adv + grad * alpha		# should this be -= grad * alpha instead?
        x_adv_new = torch.max(torch.min(x_adv_new, x + eps), x - eps).clamp(0,1)		# projection

        old_loss = -loss_fn(model(x_adv), y).item()
        while(-loss_fn(model(x_adv_new), y).item() > old_loss - .5*torch.sum(grad*torch.sub(x_adv_new, x_adv)).item()): # single dot product
            alpha *= beta
            x_adv_new = x_adv + grad * alpha
            x_adv_new = torch.max(torch.min(x_adv_new, x + eps), x - eps).clamp(0,1)

        # print(iters, loss.item(), alpha)

        x_adv = x_adv_new

    # input("Press Enter to continue...")
    return x_adv.detach() #, iters

def attackTest(dataloader, model, loss_fn, epsilon):
    size = len(dataloader.dataset)
    missclass, correct = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        if(epsilon!=0):
            # x_adv = pgd_linf(model, X, y, epsilon, alpha, num_iter, loss_fn)
            x_adv = pgd_linf_wip(model, X, y, epsilon, alpha, beta, loss_fn, 50)
        with torch.no_grad():
            originalPred = model(X)
            if(epsilon!=0):
                pred = model(x_adv)
                missclass += (pred.argmax(1) != originalPred.argmax(1)).type(torch.float).sum().item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            else:
                correct += (originalPred.argmax(1) == y).type(torch.float).sum().item()
        print("Missclass:", missclass, "Correct:",correct)

    final_acc = correct/size
    missclass_rate = missclass/size
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(dataloader.dataset), final_acc))
    return final_acc, missclass_rate

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

loss_fn = nn.CrossEntropyLoss()

test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load(modelName))
model.eval()

missclass = []
accuracy = []

for epsilon in epsilons:
    start_time = time.time()
    acc, miss = attackTest(test_loader, model, loss_fn, epsilon)
    print("--- %s seconds ---" % (time.time() - start_time))

    accuracy.append(acc)
    missclass.append(miss)

plt.figure(figsize=(5,5))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlim(-0.01,0.305)
ax2.set_xlim(-0.01,0.305)
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
fig.suptitle("PGD attack")
fig.tight_layout()
ax1.set_title("Accuracy vs Epsilon")
ax2.set_title("Missclassification rate vs Epsilon")
ax1.plot(epsilons, accuracy, "*-")
ax2.plot(epsilons, missclass, "*-")
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Accuracy")
ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Missclassification rate")
plt.show()
