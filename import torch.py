import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Load and prepare data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    features = data[['x_acc', 'y_acc']].values
    targets = data[['x_pos', 'y_pos']].values
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, targets_tensor)
    return DataLoader(dataset, batch_size=10, shuffle=True)

# Define the machine learning model
class BallPredictor(nn.Module):
    def __init__(self):
        super(BallPredictor, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the model
def train_model(model, data_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}')

# Predict function
def predict(model, input_features):
    model.eval()
    with torch.no_grad():
        predictions = model(input_features)
    return predictions

# Example usage:
data_loader = load_data('path_to_your_data.csv')  # Replace with your CSV file path
model = BallPredictor()
train_model(model, data_loader, epochs=20)

# To predict, provide some test features
test_features = torch.tensor([[0.5, 0.6]], dtype=torch.float32)  # Example test features
predicted_positions = predict(model, test_features)
print(predicted_positions)
