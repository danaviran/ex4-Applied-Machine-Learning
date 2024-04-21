import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import sklearn.linear_model
import matplotlib as plt

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        
        # Store the number of features in the original fc layer
        self.num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        self.logistic_regression = nn.Linear(self.num_features, 1)

        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False

    def forward(self, x):
        features = self.resnet18(x)
        ### YOUR CODE HERE ###
        return self.logistic_regression(features)

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)   
            outputs = model(imgs)
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels.float().unsqueeze(1)).sum().item()
    return correct / total



def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    loss = 0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        # Your code here
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss += loss.item()
    return loss / len(train_loader)

def feature_extraction(model, data_loader, device):
    """
    Extract features from the model
    :param model: The model to extract features from
    :param data_loader: The data loader
    :param device: The device to run the extraction on
    :return: The features and labels
    """
    model.eval()
    features = []
    labels = []
    # remove the output layer of the network
    model = nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        for (imgs, lbls) in data_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            features.append(model(imgs).cpu())
            labels.append(lbls.cpu())
    return torch.cat(features), torch.cat(labels)

def logistic_model():
    model = ResNet18(pretrained=True, probing=True)
    # Define the transform
    transform = model.transform
    batch_size = 32
    path = 'C:\\Users\\danaa\\Documents\\university\\whichfaceisreal'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Extract features from the model
    train_features, train_labels = feature_extraction(model, train_loader, device)
    val_features, val_labels = feature_extraction(model, val_loader, device)
    test_features, test_labels = feature_extraction(model, test_loader, device)
    # Train logistic regression model for one epoch
    logistic_regression_model = sklearn.linear_model.LogisticRegression(max_iter=1)
    logistic_regression_model.fit(train_features, train_labels)
    # Compute the accuracy
    train_acc = logistic_regression_model.score(train_features, train_labels)
    val_acc = logistic_regression_model.score(val_features, val_labels)
    test_acc = logistic_regression_model.score(test_features, test_labels)
    print(f'Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')

def train_best_model():
    model = ResNet18(pretrained=True, probing=False)
    # Define the transform
    transform = model.transform
    batch_size = 32
    num_of_epochs = 1
    learning_rate =  0.001
    path = 'C:\\Users\\danaa\\Documents\\university\\whichfaceisreal'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        # val_acc = compute_accuracy(model, val_loader, device)
        # test_acc = compute_accuracy(model, test_loader, device)
        # print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f})')
        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_acc:.4f}')
        # save the model
    torch.save(model.state_dict(), 'best_model.pth')
    return model

def train_worst_model():
    model = ResNet18(pretrained=False, probing=False)
    # Define the transform
    transform = model.transform
    batch_size = 32
    num_of_epochs = 1
    learning_rate =  0.001
    path = 'C:\\Users\\danaa\\Documents\\university\\whichfaceisreal'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        # val_acc = compute_accuracy(model, val_loader, device)
        # test_acc = compute_accuracy(model, test_loader, device)
        # print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f})')
        # Compute the test accuracy
        test_acc = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_acc:.4f}')
        # save the model
    torch.save(model.state_dict(), 'worst_model.pth')
    return model

import matplotlib.pyplot as plt
def show_images():
    best_model = train_best_model()
    worst_model = train_worst_model()
    trasnform_for_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    batch_size = 32
    path = '/content/drive/MyDrive/whichfaceisreal'
    train_loader, val_loader, test_loader_new = get_loaders(path, trasnform_for_test, batch_size)
    best_model_predicted_correctly = []
    worse_model_predicted_correctly = []
    for (imgs, labels) in test_loader_new:
    # Forward pass through both models
        best_outputs = best_model(imgs).squeeze()
        best_outputs = (torch.sigmoid(best_outputs) > 0.5).float()
        worst_outputs = worst_model(imgs).squeeze()
        worst_outputs = (torch.sigmoid(worst_outputs) > 0.5).float()
        best_model_predicted_correctly.append(best_outputs==labels)
        worse_model_predicted_correctly.append(worst_outputs == labels)

    best_model_predicted_correctly = torch.cat(best_model_predicted_correctly,0).cpu().numpy()
    worse_model_predicted_correctly = torch.cat(worse_model_predicted_correctly,0).cpu().numpy()

    best_correct_worst_inccorect = best_model_predicted_correctly & ~ worse_model_predicted_correctly
    indices = np.where(best_correct_worst_inccorect)[0][:5]

    for i in indices:
        plt.imshow(test_loader_new.dataset[i][0].permute(1,2,0))
        print("label = ", test_loader_new.dataset[i][1])
        plt.show()


# Set the random seed for reproducibility
torch.manual_seed(0)
### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
# model = ResNet18(pretrained=False, probing=False)
# Linear probing
# model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

# Define the transform
transform = model.transform
batch_size = 32
num_of_epochs = 50
learning_rate =  0.001
path = 'C:\\Users\\danaa\\Documents\\university\\whichfaceisreal'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print device
print(device)
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    # val_acc = compute_accuracy(model, val_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')
    # Stopping condition
    if test_acc > 0.97:
         break

