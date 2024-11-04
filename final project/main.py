import torch
import torchvision
from Compose import *
from train import Learn
from loss_functions import SSE_loss
loss = SSE_loss()
from model import Model
from evaluat import *
print("final project")

# Load Data: Load the MNIST dataset and preprocess it.
data = np.load("./data/classify_digits_data/mnist.npy")

# preparing of the data
compose = torchvision.transforms.Compose([ToClassifyDigitsData(),Split([6.0/7,1.0/7],False)])

trine_data , text_data = compose(data)

# Define the Model: Create a neural network model suitable for image classification.
model = Model([784,24,10],activation_function=torch.sigmoid , loss_function = loss)

# Train the Model: Run the training loop to adjust model weights based on the training data.
learn = Learn(model,9,8,3.0)
learn(trine_data)


# Evaluate the Model: Use the evaluate, accuracy_score, recall_score, and f1_score functions to assess performance.
print(f"evaluation | accuracy: {accuracy_score(model,text_data)} , F1: {f1_score(model,text_data)} , recall: {recall_score(model,text_data)}")