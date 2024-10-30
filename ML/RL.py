from console.game import Game
from console.game_state import GameState, WallColor

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


class RL_CNNModel(nn.Module):
    def __init__(self):
        super(RL_CNNModel, self).__init__()

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
model = RL_CNNModel().to(device)


def tree_expl():
    game_state = GameState()
    game_state.get_available_wall_placements()


'''
RL policy (no expert)
Value policy

action space (12 + 2*(N-1)^2)
    moves {left, right, up, down, top-left, top-right, bottom_left, bottom_right, skip_left, skip_right, skip_up, skip_down}
    walls {(i,j, h) for [i,j in N-1],[h in 0,1]}

value space:
    [-1.0, ... , 1.0]

We don't have expert, either use estimation expert, or find other way of exploration

transformer?
(embedding, pass through multi-layer transformer w/ self-attention)
- issue is that all need to be embeded into same dim, they sort of all need to be embedded together
embedding:
    create token for each things (vert wall present, vert wall not present, horiz wall present, no hor wall, empty_cell, player1, player2, player1_walls, player2_walls, is_player1_turn)
    positional embedding
transformer:
    outputs either
    - softmax on action space (then can choose top k)
    - single value sigmoided to (-1 ... 1)


CNN?
Make a grid of the game board, perform conv on that, flatten, then add numerical inputs (wall nums, player pos, etc)



'''
