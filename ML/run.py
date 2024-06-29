import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random
from pathlib import Path

from makeRep import Game_data


class QuoridorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor3D, p1_walls_left, p2_walls_left = self.data[idx]
        return torch.Tensor(tensor3D), torch.Tensor([p1_walls_left]), torch.Tensor([p2_walls_left])

def generate_synthetic_data(num_samples, rounds, file, batch_size=1):
    X = []
    y = []

    g = Game_data(file)

    for _ in range(num_samples):
        r = random.randint(0, rounds)
        random_num = random.randint(1, len(g.data[r])-1)
        tensor, p1_walls, p2_walls, p1_won, next_move = g.getRoundLine(r, random_num)


        tensor3D_tensor = torch.tensor(tensor, dtype=torch.float32)
        int1_tensor = torch.full_like(tensor3D_tensor[..., :1], p1_walls)  # Create int1_tensor with the same shape as tensor3D_tensor, filled with int1
        int2_tensor = torch.full_like(tensor3D_tensor[..., :1], p2_walls)  # Create int2_tensor with the same shape as tensor3D_tensor, filled with int2

        # Stack them along the last dimension
        tensor4D_tensor = torch.cat((tensor3D_tensor, int1_tensor, int2_tensor), dim=-1)

        X.append(tensor4D_tensor)
        y.append(int(p1_won))

    # Convert to PyTorch Dataset and DataLoader
    X = torch.stack(X, dim=0)  # Stack all tensors in X into a single tensor along batch dimension
    y = torch.tensor(y, dtype=torch.float32)  # Assuming y is a list of labels


    # Convert to PyTorch Dataset and DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the model
model = CNNModel()


def train_model(data_loader, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for x,y in data_loader:
            inputs, p1_walls, p2_walls = x
            labels = y  # Assuming p1_walls or p1_won as labels
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Example usage
TRAINFILE = "saved_games/2024-06-28_22:19_(randomBot | impatientBot)_rounds_1000.pkl"
TRAIN_NUM_DATA_SAMPLES = 128
TRAIN_BATCH_SIZE = 8
EPOCHS = 10

data_loader = generate_synthetic_data(TRAIN_NUM_DATA_SAMPLES, 100, TRAINFILE, TRAIN_BATCH_SIZE)
import code; code.interact(local = locals())
train_model(data_loader, EPOCHS)
