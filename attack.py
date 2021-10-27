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
import torch.nn.functional as F

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model = "lenet_mnist_model.pth"
self_trained_model = "model_weights.pth"

# Basic neural net class def
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

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
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# Download/create test data from QMNIST dataset
test_data = datasets.QMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    # target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
#loads training and testing data into dataloader objects
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(self_trained_model))
model.eval()

preModel = Net().to(device)
preModel.load_state_dict(torch.load(pretrained_model))
preModel.eval()

preAccuracies = []
preExamples = []
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = attackTest(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

    acc, ex = attackTest(preModel, device, test_loader, eps)
    preAccuracies.append(acc)
    preExamples.append(ex)

plt.figure(figsize=(5,5))
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
fig.suptitle("Accuracy vs Epsilon")
fig.tight_layout()
ax1.set_title("pre-trained model")
ax2.set_title("self-made model")
ax1.plot(epsilons, preAccuracies, "*-")
ax2.plot(epsilons, accuracies, "*-")
ax1.set_xlabel("Epsilon")
ax1.set_ylabel("Accuracy")
ax2.set_xlabel("Epsilon")
ax2.set_ylabel("Accuracy")
plt.show()

# # Plot several examples of adversarial samples at each epsilon
# cnt = 0
# plt.figure(figsize=(8,10))
# plt.title("Self trained examples")
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons),len(examples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig,adv,ex = examples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.show()
#
# # Plot several examples of adversarial samples at each epsilon
# cnt = 0
# plt.figure(figsize=(8,10))
# plt.title("Pre-trained examples")
# for i in range(len(epsilons)):
#     for j in range(len(preExamples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons),len(preExamples[0]),cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig,adv,ex = preExamples[i][j]
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(ex, cmap="gray")
# plt.tight_layout()
# plt.show()
