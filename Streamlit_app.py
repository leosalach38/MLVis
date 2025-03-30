import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
@st.cache_data
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader, train_dataset

train_loader, test_loader, train_dataset = load_data()

# Display sample images
st.title("CNN Learning Visualization (PyTorch)")
st.write("Below are some sample images from the dataset:")
fig, ax = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    ax[i].imshow(train_dataset[i][0].squeeze(), cmap='gray')
    ax[i].axis('off')
st.pyplot(fig)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Function to visualize CNN filters
def plot_filters(layer, num_filters=8):
    filters = layer.weight.data.cpu().numpy()
    fig, axes = plt.subplots(1, num_filters, figsize=(10, 3))
    for i in range(num_filters):
        f = filters[i, 0]  # Get 2D filter
        axes[i].imshow(f, cmap='gray')
        axes[i].axis('off')
    return fig

# Show initial filters (before training)
st.write("Initial (Untrained) Filters:")
st.pyplot(plot_filters(model.conv1))

# Train the model if user clicks the button
if st.button("Train CNN Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 5 epochs
    for epoch in range(5):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    st.write("Training Completed!")

    # Show trained filters
    st.write("Trained Filters:")
    st.pyplot(plot_filters(model.conv1))

    # Visualize feature maps
    def visualize_feature_maps(model, image):
        """Extract and visualize feature maps from the convolutional layers."""
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            x = torch.relu(model.conv1(image))
            x = model.pool(x)

        feature_maps = x.squeeze().cpu().numpy()
        fig, axes = plt.subplots(1, min(8, feature_maps.shape[0]), figsize=(10, 3))
        for i in range(min(8, feature_maps.shape[0])):
            axes[i].imshow(feature_maps[i], cmap='viridis')
            axes[i].axis('off')
        return fig

    st.write("Feature Maps for a Test Image:")
    st.pyplot(visualize_feature_maps(model, train_dataset[0][0]))
