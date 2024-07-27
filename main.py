import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np  # Ensure NumPy is imported
import multiprocessing

def main():
    # Load and preprocess CIFAR-10 data
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),  # ResNet18 requires 224x224 input size
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Load the ResNet18 model pre-trained on ImageNet
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 10)  # Adjust the final layer to match the number of classes in CIFAR-10

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    train_losses = []
    test_losses = []
    epochs = 5  # Reduced number of epochs for faster training

    for epoch in range(epochs):
        
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                =
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

        test_losses.append(test_loss / len(testloader))
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
