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

epsilons = [0, .05, .1, .15, .2, .25, .3]
modelName = "robust_model_weights.pth"

# prebuilt neural net class def
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

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def attackTest( model, device, test_loader, epsilon ):
    # Accuracy counter
    correct, missclass = 0, 0

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        init_pred = model(data)

        if(epsilon!=0):
            # Calculate the loss
            loss = F.nll_loss(init_pred, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Check for success
        if(epsilon!=0):
            pred = model(perturbed_data)
            missclass += (pred.argmax(1) != init_pred.argmax(1)).type(torch.float).sum().item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
        else:
            correct += (init_pred.argmax(1) == target).type(torch.float).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    missclass_rate = missclass/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, missclass_rate

# Download/create test data from MNIST dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#loads training and testing data into dataloader objects
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = Net().to(device)
model.load_state_dict(torch.load(modelName))
model.eval()

accuracy = []
missclass = []

# Run test for each epsilon
for epsilon in epsilons:
    acc, miss = attackTest(model, device, test_loader, epsilon)
    accuracy.append(acc)
    missclass.append(miss)

plt.figure(figsize=(5,5))
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_xlim(-0.01,0.305)
ax2.set_xlim(-0.01,0.305)
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
fig.suptitle("FGSM attack")
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
