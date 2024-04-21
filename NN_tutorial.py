import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from helpers import *
import torch.nn.init as init
import torch.cuda

def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=10, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float().to(device), torch.tensor(train_data['country'].values).long().to(device))
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float().to(device), torch.tensor(val_data['country'].values).long().to(device))
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float().to(device), torch.tensor(test_data['country'].values).long().to(device))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    layer_indices = [0, 30, 60, 90, 95, 99]
    # List to store average gradient magnitudes for each layer
    gradient_magnitudes = {layer: [] for layer in layer_indices}
    
    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        epoch_grads = {layer: [] for layer in layer_indices}
        # Iterate over all batches in the training set
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # perform a training iteration
            # move the inputs and labels to the device
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model.forward(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()

            # Store the magnitude of the gradients for each layer and then average them at the end of the epoch
            # The magnitude of of the gradients is defined by: grad magnitude = ||grad||22.
            layer_index = 0
            # Iterate over all modules (layers) within the model.
            for _, m in model.named_modules():
                # Check if the current module is a linear layer
                if isinstance(m, nn.Linear):
                    # Check if the current layer index is in the list of layer indices
                    if layer_index in layer_indices:
                        # Calculate the magnitude of the gradients for the weights and biases
                        gradient_magnitude = torch.norm(m.weight.grad) ** 2 + torch.norm(m.bias.grad) ** 2
                        # Append the gradient magnitude to the list for the current layer
                        epoch_grads[layer_index].append(gradient_magnitude.item())
                    # Increment the layer index for every linear layer
                    layer_index += 1

            # update the weights
            optimizer.step()
            # Store the loss values for plotting
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()
        # Average the gradient magnitudes for each layer
        for layer in layer_indices:
            if len(epoch_grads[layer]) > 0:
                # Store the average gradient magnitude for each layer
                gradient_magnitudes[layer].append(np.mean(epoch_grads[layer]))

        # Calculate train accuracy and loss values for entire epoch
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    # perform an evaluation iteration
                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # forward pass
                    outputs = model.forward(inputs)
                    # calculate the loss
                    loss = criterion(outputs, labels)
                    # sum up batch loss
                    ep_loss += loss.item() 
                    # get the index of the max log-probability
                    _, predicted = torch.max(outputs.data, 1) 
                    # update the total count
                    total += labels.size(0) 
                     # update the correct count
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total) # store the accuracy
                losses.append(ep_loss / len(loader)) # store the loss

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))
        print('Epoch {:}, Train Loss: {:.3f}, Val Loss: {:.3f}, Test Loss: {:.3f}'.format(ep, train_losses[-1], val_losses[-1], test_losses[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradient_magnitudes

def train_implicit_model(train_data, val_data, test_data, model, lr=0.001, epochs=10, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['sin1', 'sin2', 'sin3', 'sin4', 
        'sin5', 'sin6', 'sin7', 'sin8', 'sin9', 'sin10', 'sin11', 'sin12', 'sin13', 
        'sin14', 'sin15', 'sin16', 'sin17', 'sin18', 'sin19', 'sin20']].values).float().
        to(device), torch.tensor(train_data['country'].values).long().to(device))
    
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['sin1', 'sin2', 'sin3', 'sin4', 
        'sin5', 'sin6', 'sin7', 'sin8', 'sin9', 'sin10', 'sin11', 'sin12', 'sin13', 
        'sin14', 'sin15', 'sin16', 'sin17', 'sin18', 'sin19', 'sin20']].values).float().
        to(device), torch.tensor(val_data['country'].values).long().to(device))
    
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['sin1', 'sin2', 'sin3', 'sin4', 
        'sin5', 'sin6', 'sin7', 'sin8', 'sin9', 'sin10', 'sin11', 'sin12', 'sin13', 
        'sin14', 'sin15', 'sin16', 'sin17', 'sin18', 'sin19', 'sin20']].values).float().
        to(device), torch.tensor(test_data['country'].values).long().to(device))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    # List to store average gradient magnitudes for each layer
    gradient_magnitudes = [[] for _ in range(len(model))]
    
    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # perform a training iteration
            # move the inputs and labels to the device
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            outputs = model.forward(inputs)
            # calculate the loss
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
            # Store the loss values for plotting
            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()
            # Store the magnitude of the gradients for each layer
            # The magnitude of of the gradients is defined by: grad magnitude = ||grad||22.
            # This is a common way to measure the magnitude of the gradients.
            # average the magnitudes of each layer in every epoch
            # Compute and store gradient magnitudes for each layer
            # Store the magnitude of the gradients for each layer
            for layer_index, layer in enumerate(model):
                if hasattr(layer, 'weight'):
                    gradient_magnitudes[layer_index].append(layer.weight.grad.norm().item())
                if hasattr(layer, 'bias') and layer.bias is not None:
                    gradient_magnitudes[layer_index].append(layer.bias.grad.norm().item())
            
        # Calculate train accuracy and loss values for entire epoch
        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    # perform an evaluation iteration
                    # move the inputs and labels to the device
                    inputs, labels = inputs.to(device), labels.to(device)
                    # forward pass
                    outputs = model.forward(inputs)
                    # calculate the loss
                    loss = criterion(outputs, labels)
                    # sum up batch loss
                    ep_loss += loss.item() 
                    # get the index of the max log-probability
                    _, predicted = torch.max(outputs.data, 1) 
                    # update the total count
                    total += labels.size(0) 
                     # update the correct count
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total) # store the accuracy
                losses.append(ep_loss / len(loader)) # store the loss

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))
        print('Epoch {:}, Train Loss: {:.3f}, Val Loss: {:.3f}, Test Loss: {:.3f}'.format(ep, train_losses[-1], val_losses[-1], test_losses[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradient_magnitudes

def scenario1():
    output_dim = len(train_data['country'].unique())
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    
    all_val_losses = []  # List to store all validation losses for each learning rate
    all_val_accs = []  # List to store all validation losses for each learning rate
    max_epochs = 0  # Maximum number of epochs among all learning rates
    

    # Train the network with learning rates of: 1., 0.01, 0.001, 0.00001
    for learning_rate in [1.0, 0.01, 0.001, 0.00001]:
      # Define the model architecture
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                nn.Linear(16, output_dim)]  # output layer
        model = nn.Sequential(*model)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_model(train_data, val_data, test_data, model, lr=learning_rate, epochs=50, batch_size=256)

        # Plot validation loss for current learning rate
        plt.figure()
        plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'LR={learning_rate:.5f}', linestyle='', marker='o')
        plt.title(f'Validation Losses - Learning Rate {learning_rate:.5f}')
        padding = 0.1 * len(val_losses)  # Adjust the padding factor as needed
        plt.xlim(1 - padding, len(val_losses) + padding)  # Add padding to x-axis limits
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Append validation losses to the list
        all_val_losses.append((learning_rate, val_losses))
        all_val_accs.append((learning_rate, val_accs))
        max_epochs = max(max_epochs, len(val_losses))

    # Plot loss values through epochs of all learning rates in the same graph
    plt.figure()
    for learning_rate, val_losses in all_val_losses:
        x_values = range(1, len(val_losses) + 1)
        y_values = val_losses + [None] * (max_epochs - len(val_losses))  # Pad shorter lists with None
        plt.plot(x_values, y_values, label=f'LR={learning_rate:.5f}', linestyle='', marker='o')
    # Set x-ticks as integer points divisible by 5
    plt.xticks([tick for tick in range(1, max_epochs + 1) if tick % 5 == 0])
    plt.title('Validation Losses - All Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot accuracy values through epochs of all learning rates in the same graph
    plt.figure()
    for learning_rate, val_accs in all_val_accs:
        x_values = range(1, len(val_accs) + 1)
        y_values = val_accs + [None] * (max_epochs - len(val_accs))  # Pad shorter lists with None
        plt.plot(x_values, y_values, label=f'LR={learning_rate:.5f}', linestyle='', marker='o')
    # Set x-ticks as integer points divisible by 5
    plt.xticks([tick for tick in range(1, max_epochs + 1) if tick % 5 == 0])
    plt.title('Validation Accuracies - All Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def scenario2():
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())

    max_epochs = 0  # Maximum number of epochs among all learning rates

    # Define the model architecture
    model = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 1
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 2
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
            nn.Linear(16, output_dim)]  # output layer
    model = nn.Sequential(*model)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradients = \
    train_model(train_data, val_data, test_data, model, lr=0.001, epochs=100, batch_size=256)

    # Plot validation loss for current learning rate
    plt.figure()
    plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'LR={0.001:.5f}', linestyle='', marker='o')
    plt.title(f'Validation Losses - Learning Rate {0.001:.5f}')
    padding = 0.1 * len(val_losses)  # Adjust the padding factor as needed
    plt.xlim(1 - padding, len(val_losses) + padding)  # Add padding to x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    # Plot validation loss for current learning rate for just 1,5,...
    validation_losses = {
    1: val_losses[0],
    5: val_losses[4],
    10: val_losses[9],
    20: val_losses[19],
    50: val_losses[49],
    100: val_losses[99]}

    # Extract epochs and validation losses
    epochs = list(validation_losses.keys())
    losses = list(validation_losses.values())

    # Plot validation losses
    plt.plot(epochs, losses, marker='o', linestyle='')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Losses - Learning Rate {0.001}')
    plt.legend()
    plt.show()



def scenario3():
    import time

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    
    output_dim = len(train_data['country'].unique())

    for batch_size, epoch_num in [(1,1), (16,10), (128,50), (1024,50)]:
      # Define the model architecture
        model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
                nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
                nn.Linear(16, output_dim)]  # output layer
        model = nn.Sequential(*model)
        start_time = time.time()  # Record the start time
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epoch_num, batch_size=batch_size)

        end_time = time.time()  # Record the end time
        total_time = end_time - start_time  # Calculate the total time
        print(f"Total training time: {total_time:.2f} seconds")

        # Plot validation loss for current learning rate
        plt.figure()
        plt.title(f'Losses - batch size of {batch_size}, {epoch_num} epochs')
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linestyle='', marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='', marker='o')
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', linestyle='', marker='o')
        padding = 0.1 * len(val_losses)  # Adjust the padding factor as needed
        plt.xlim(1 - padding, len(val_losses) + padding)  # Add padding to x-axis limits
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot validation accuracy for current learning rate
        plt.figure()
        plt.title(f'Accuracies - batch size of {batch_size}, {epoch_num} epochs')
        plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy', linestyle='', marker='o')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', linestyle='', marker='o')
        plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy', linestyle='', marker='o')
        padding = 0.1 * len(val_accs)  # Adjust the padding factor as needed
        plt.xlim(1 - padding, len(val_accs) + padding)  # Add padding to x-axis limits
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def scenario4(model, depth, width, epochs):
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradients = \
    train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epochs, batch_size=256)
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)

    # Plot validation loss for current learning rate
    plt.figure()
    plt.title(f'Losses - NN of depth {depth} and width {width} over {epochs} epochs')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linestyle='', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='', marker='o')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', linestyle='', marker='o')
    padding = 0.1 * len(val_losses)  # Adjust the padding factor as needed
    plt.xlim(1 - padding, len(val_losses) + padding)  # Add padding to x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Plot validation accuracy for current learning rate
    plt.figure()
    plt.title(f'Accuracies - NN of depth {depth} and width {width} over {epochs} epochs')
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy', linestyle='', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', linestyle='', marker='o')
    plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy', linestyle='', marker='o')
    padding = 0.1 * len(val_accs)  # Adjust the padding factor as needed
    plt.xlim(1 - padding, len(val_accs) + padding)  # Add padding to x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def scenario4_models():
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())

    # one hidden layer, 16 neurons each
    model1 = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),
              nn.Linear(16, output_dim)]
    model = nn.Sequential(*model1)
    # scenario4(model, 1, 16, 30)

    # two hidden layers. 16 neurons each
    model2 = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), 
              nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), 
              nn.Linear(16, output_dim)]
    model2 = nn.Sequential(*model2)
    scenario4(model2, 2, 16, 40)

    # six hidden layers. 16 neurons each
    model3 = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 1
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 2
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
            nn.Linear(16, output_dim)]  # output layer
    model3 = nn.Sequential(*model3)
    scenario4(model3, 6, 16, 50)

    # ten hidden layers. 16 neurons each
    model4 = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 1
            nn.Linear(16, 16),nn.BatchNorm1d(16), nn.ReLU(),   # hidden layer 2
            nn.Linear(16, 16), nn.BatchNorm1d(16),  nn.ReLU(), # hidden layer 3
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 7
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 8
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 9
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 10
            nn.Linear(16, output_dim)]  # output layer
    model4 = nn.Sequential(*model4)
    scenario4(model4, 10, 16, 50)

    # six hidden layers. 8 neurons each
    model5 = [nn.Linear(2, 8), nn.BatchNorm1d(8), nn.ReLU(),   # hidden layer 1
            nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),   # hidden layer 2
            nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),  # hidden layer 3
            nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),  # hidden layer 4
            nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),  # hidden layer 5
            nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(),  # hidden layer 6
            nn.Linear(8, output_dim)]  # output layer
    model5 = nn.Sequential(*model5)
    scenario4(model5, 6, 8, 50)

    # six hidden layers. 32 neurons each
    model6 = [nn.Linear(2, 32), nn.BatchNorm1d(32), nn.ReLU(),   # hidden layer 1
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),   # hidden layer 2
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),  # hidden layer 3
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),  # hidden layer 4
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),  # hidden layer 5
            nn.Linear(32, 32), nn.BatchNorm1d(32), nn.ReLU(),  # hidden layer 6
            nn.Linear(32, output_dim)]  # output layer
    model6 = nn.Sequential(*model6)
    scenario4(model6, 6, 32, 70)

    # six hidden layers. 64 neurons each
    model7 = [nn.Linear(2, 64), nn.BatchNorm1d(64), nn.ReLU(),  # hidden layer 1
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),   # hidden layer 2
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),  # hidden layer 3
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),  # hidden layer 4
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),  # hidden layer 5
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),  # hidden layer 6
            nn.Linear(64, output_dim)]  # output layer
    model7 = nn.Sequential(*model7)
    scenario4(model7, 6, 64, 70)

def scenario5():
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())
    
    # Define a NN of depth 100 and width 4 with batch normalization
    input_layer = nn.Linear(2, 4)
    hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), 
    nn.BatchNorm1d(4),nn.ReLU() ) for _ in range(100)])
    output_layer = nn.Linear(4, output_dim)
    model = nn.Sequential(input_layer, *hidden_layers, output_layer)

    # Initialize weights using He initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Adjust the standard deviation for He initialization
            # You can adjust the scale_factor based on your requirement
            scale_factor = 2.0  # Adjust this value to increase or decrease initialization strength
            std = scale_factor * (2.0 / m.in_features) ** 0.5
            init.normal_(m.weight, mean=0, std=std)

    # Train the model
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradients = \
    train_model(train_data, val_data, test_data, model, lr=0.001, epochs=2, batch_size=256)

    layers_indices = [0, 30, 60, 90, 95, 99]
    # Plot the gradient magnitudes for each layer
    for layer_index in layers_indices:
        plt.plot(gradients[layer_index], label=f'Layer {layer_index}')
    # Set x-ticks as 1,2,3,...,10
    plt.xticks(range(1,11))
    plt.xlabel('epochs')
    plt.ylabel('gradeint magnitudes')
    plt.title('Gradient Magnitudes for each layer over 10 epochs')
    plt.legend()
    plt.show()

def scenario6():
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())

    # Implement an implicit representation pre-processing to the data, passing the input through 10 sine function
    # sin(α·x): where α ∈ {0.1, 0.2, ..., 1.}.
    # The input data is 2-dimensional, so the pre-processed data will be 20-dimensional.
    # The pre-processed data will be passed through a NN with depth 6 and width 16.

    # Apply the implicit representation pre-processing to the longitude and latitude columns
    # of the train, validation, and test data
    # The input data is 2-dimensional, so the pre-processed data will be 20-dimensional.
    new_train_data = train_data.copy()
    # Iterate over the sinus function parameters for longitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        new_train_data[f'sin{i}'] = np.sin(param * train_data['long'].values)
    # Iterate over the sinus function parameters for latitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=10+1):
        new_train_data[f'sin{i}'] = np.sin(param * train_data['lat'].values)
    new_train_data.drop(columns=['long'], inplace=True)
    new_train_data.drop(columns=['lat'], inplace=True)
    new_train_data.drop(columns=['country'], inplace=True)
    new_train_data['country'] = train_data['country']
    print(new_train_data.head())

    new_val_data = val_data.copy()
    # Iterate over the sinus function parameters for longitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        new_val_data[f'sin{i}'] = np.sin(param * val_data['long'].values)
    # Iterate over the sinus function parameters for latitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=10+1):
        new_val_data[f'sin{i}'] = np.sin(param * val_data['lat'].values)
    new_val_data.drop(columns=['long'], inplace=True)
    new_val_data.drop(columns=['lat'], inplace=True)
    new_val_data.drop(columns=['country'], inplace=True)
    new_val_data['country'] = val_data['country']
    print(new_val_data.head())

    new_test_data = test_data.copy()
    # Iterate over the sinus function parameters for longitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=1):
        new_test_data[f'sin{i}'] = np.sin(param * test_data['long'].values)
    # Iterate over the sinus function parameters for latitude
    for i, param in enumerate(np.arange(0.1, 1.1, 0.1), start=10+1):
        new_test_data[f'sin{i}'] = np.sin(param * test_data['lat'].values)
    new_test_data.drop(columns=['long'], inplace=True)
    new_test_data.drop(columns=['lat'], inplace=True)
    new_test_data.drop(columns=['country'], inplace=True)
    new_test_data['country'] = test_data['country']
    print(new_test_data.head())
    
    # Train a NN with depth 6 and width 16 on top of these representations.
    model = [nn.Linear(20, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 2
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), # hidden layer 3
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), # hidden layer 4
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), # hidden layer 5
            nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(), # hidden layer 6
            nn.Linear(16, output_dim)]  # output layer
    model = nn.Sequential(*model)
    

    # Train the model
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, gradients = \
    train_implicit_model(new_train_data, new_val_data, new_test_data, 
                model, lr=0.001, epochs=60, batch_size=256)
    
    # Plot validation loss for current learning rate
    plt.figure()
    plt.title(f'Losses - NN of depth 6 and width 16 over 100 epochs')
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', linestyle='', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linestyle='', marker='o')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', linestyle='', marker='o')
    padding = 0.1 * len(val_losses)  # Adjust the padding factor as needed
    plt.xlim(1 - padding, len(val_losses) + padding)  # Add padding to x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # Plot validation accuracy for current learning rate
    plt.figure()
    plt.title(f'Accuracies - NN of depth 6 and width 16 over 100 epochs')
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy', linestyle='', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', linestyle='', marker='o')
    plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy', linestyle='', marker='o')
    padding = 0.1 * len(val_accs)  # Adjust the padding factor as needed
    plt.xlim(1 - padding, len(val_accs) + padding)  # Add padding to x-axis limits
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=True)
    

def plot_sixteen_width():
    # Define your x-axis values
    x_values = [1, 2, 6, 10]
    # Define arrays of train, validation, and test accuracies for each x-value
    train_accuracies = [0.847, 0.854, 0.898, 0.910]
    val_accuracies = [0.925, 0.941, 0.956, 0.953]
    test_accuracies = [0.916, 0.938, 0.956, 0.951]
    # Plotting the scatter plot for train accuracies
    plt.scatter(x_values, train_accuracies, color='r', label='Train Accuracy')
    # Plotting the scatter plot for validation accuracies
    plt.scatter(x_values, val_accuracies, color='g', label='Validation Accuracy')
    # Plotting the scatter plot for test accuracies
    plt.scatter(x_values, test_accuracies, color='b', label='Test Accuracy')
    # Add x-axis labels and set ticks
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Hidden Layers for 16 Neurons')
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

def plot_6_layers():
    # Define your x-axis values
    x_values = [8, 16, 32, 64]
    # Define arrays of train, validation, and test accuracies for each x-value
    train_accuracies = [0.857, 0.898, 0.927, 0.939]
    val_accuracies = [0.935, 0.956, 0.960, 0.963]
    test_accuracies = [0.929, 0.956, 0.959, 0.959]
    # Plotting the scatter plot for train accuracies
    plt.scatter(x_values, train_accuracies, color='r', label='Train Accuracy')
    # Plotting the scatter plot for validation accuracies
    plt.scatter(x_values, val_accuracies, color='g', label='Validation Accuracy')
    # Plotting the scatter plot for test accuracies
    plt.scatter(x_values, test_accuracies, color='b', label='Test Accuracy')
    # Add x-axis labels and set ticks
    plt.xlabel('Number of Neurons')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Neurons for 6 Hidden Layers')
    # Set x-axis ticks to be the same as x_values
    plt.xticks(x_values)
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

def skip_connection_network():
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')
    output_dim = len(train_data['country'].unique())

    # Define a NN of depth 100 and width 4 with batch normalization
    input_layer = nn.Linear(2, 4)
    hidden_layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4),nn.ReLU()) for _ in range(100)])
    output_layer = nn.Linear(4, output_dim)
    model = nn.Sequential(input_layer, *hidden_layers, output_layer)

if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # scenario1()
    scenario2()
    # scenario3()
    # scenario4(10, 10, 10)
    # scenario4_models()
    # plot_6_layers()
    # scenario5()
    # scenario6()

    """
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs.')
    plt.legend()
    plt.show()
    

   
    """