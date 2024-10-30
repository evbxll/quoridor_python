import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt

from makeRep import Game_data
from helper import getNumRounds, COLOR

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def generate_synthetic_data(num_samples, file, batch_size=1):
    X = []
    y = []

    g = Game_data(file)
    rounds = getNumRounds(file)

    for _ in range(num_samples):
        rand_round = random.randint(0, rounds-1)
        rand_line = random.randint(0, len(g.data[rand_round])-1)
        hor_wall_tensor, ver_wall_tensor, player_pos_tensor, p1_walls, p2_walls, p1_won, next_move = g.getRoundLine(rand_round, rand_line)


        # hor_wall_tensor
        # hor_wall_tensor
        # hor_wall_tensor
        # X.append(tensor4D_tensor)
        y.append(int(p1_won), next_move)

    # Convert to PyTorch Dataset and DataLoader
    X = torch.stack(X, dim=0)  # Stack all tensors in X into a single tensor along batch dimension
    y = torch.tensor(y, dtype=torch.float32)  # Assuming y is a list of labels

    # print(X)

    # Convert to PyTorch Dataset and DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.convfc1 = nn.Linear(144, 126)



        self.pool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        board_inps = x[:, :3, :, :]
        ints_matrix = x[:, 3:, :, :]
        int_inps = ints_matrix[:, :, 0, 0]

        cnn_ouput = self.pool(F.relu(self.conv2(F.relu(self.conv1(board_inps)))))
        cnn_ouput = cnn_ouput.view(cnn_ouput.size(0), -1)
        # print(cnn_ouput.shape)
        cnn_ouput = self.convfc1(cnn_ouput)

        # print(cnn_ouput.shape)

        #turn back into batch_size x others matrix
        x = torch.cat((cnn_ouput, int_inps), dim=1)

        # print(x.shape)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Instantiate the model and move it to GPU
model = CNNModel().to(device)


def train_model(data_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # get intial loss, for sanity
    for x,y in data_loader:
        # x is a bx5x9x9
        inputs = x.to(device)
        labels = y.view(-1, 1).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    print(f"Intial Loss: {loss.item():.4f}")

    for epoch in range(epochs):
        for x,y in data_loader:
            # x is a bx5x9x9
            inputs = x.to(device)
            labels = y.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Example usage
TRAIN_FILE = "../saved_games/2024-08-15_08:03_(path-search | path-search)_rounds_2.pkl"
TRAIN_NUM_DATA_SAMPLES = 100
TRAIN_BATCH_SIZE = 1024
EPOCHS = 100

data_loader = generate_synthetic_data(TRAIN_NUM_DATA_SAMPLES, TRAIN_FILE, TRAIN_BATCH_SIZE)
# import code; code.interact(local = locals())
train_model(data_loader, EPOCHS)





TEST_FILE = "../saved_games/2024-08-18_20:40_(path-search | path-search)_rounds_20.pkl"#"saved_games/2024-07-01_21:25_(path-search | path-search)_rounds_1000.pkl"
TEST_NUM_DATA_SAMPLES = 10

def runtest(test_rounds, file):
    total_loss = 0.0
    pos_truth_loss = 0.0
    neg_truth_loss = 0.0
    actual = 0
    squared_error = 0.0
    plt.figure(figsize=(10, 6))


    data_loader = generate_synthetic_data(test_rounds, file, 1)

    for x, y in data_loader:
        inputs = x.to(device)
        labels = y.to(device)

        TRUTH = labels.item()

        prediction = model(inputs)[0]
        pred = prediction.item()

        loss = F.mse_loss(prediction, labels.float())
        total_loss += loss.item()

        print("\nPrediction:", f"{pred:.6f}")
        print("Actual:", f"{TRUTH:.6f}")

        color = COLOR.RED if loss > 0.1 else COLOR.GREEN
        print(color, "Loss:", f"{loss.item():.6f}", COLOR.RESET)

        c = 'green' if loss < 0.1 else 'red'
        plt.scatter(pred, TRUTH, color=c)

        if TRUTH == 1.0:
            actual += 1
            pos_truth_loss += loss
        else:
            neg_truth_loss += loss

        squared_error += abs(pred - TRUTH)

    mean_squared_error = squared_error / test_rounds

    # Print statements with accuracy
    print("\n\n")
    print(f"Total     {test_rounds}")
    print(f"P1 wins   {actual}\n")

    # Print average loss and mean squared error
    print(f"MSE:     {total_loss / test_rounds:.6f}")
    print(f"Avg err: {mean_squared_error:.6f}\n")

    print(f"Loss for pos truth:   {pos_truth_loss / (actual):.6f}")
    print(f"Loss for neg truth:   {neg_truth_loss / (test_rounds - actual):.6f}\n")


    # Add labels and title
    # plt.xlabel('Index')
    # plt.ylabel('Predicted Value')
    # plt.title('Predictions with Error Color Coding')
    plt.savefig('plot.png')
    # Optionally, explicitly close the plot to free up resources
    plt.close()

runtest(TEST_NUM_DATA_SAMPLES, TEST_FILE)

'''
CNN architecture, prune available moves
'''