import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from makeRep import Game_data
from helper import getNumRounds, COLOR
from tokenizer import Tokenizer
from console.game import Game
from console.game_state import GameState


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


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
tokenizer = Tokenizer()


def generate_synthetic_data(num_samples: int, file: str, batch_size: int = 1, end:float = 1.0) -> DataLoader:
    X, y = [], []
    g = Game_data(file)
    rounds = getNumRounds(file)

    for _ in range(num_samples):
        # Select random round and line within the round
        rand_round = random.randint(0, int(rounds*end))
        rand_line = random.randint(0, len(g.data[rand_round]) - 1)

        # Extract features for the selected round and line
        hor_wall_tensor, ver_wall_tensor, player_pos_tensor, p1_walls, p2_walls, p1_won, next_move = g.getRoundLine(rand_round, rand_line)
        player1_turn = rand_line % 2 == 0  # Alternate turns based on line number

        # Encode features for the model and add to the dataset
        encoded_features = tokenizer.encode(hor_wall_tensor, ver_wall_tensor, player_pos_tensor, p1_walls, p2_walls, player1_turn)
        X.append(encoded_features)
        y.append(int(p1_won))  # Append outcome as target label

    # Stack features and labels into tensors for DataLoader
    X = torch.stack(X)  # Stack encoded features along batch dimension
    y = torch.tensor(y, dtype=torch.float32)  # Convert labels to tensor

    # Wrap into TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def train_model(data_loader, model, optimizer, criterion, scheduler, epochs, save_path="model.pth"):
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        for X, y in tqdm(data_loader, desc="Batches"):
            inputs = X.to(device)
            labels = y.view(-1, 1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def evaluate_model(data_loader: DataLoader, model: torch.nn.Module, criterion, device: torch.device) -> float:
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for X, y in tqdm(data_loader, desc="Evaluating"):
            inputs = X.to(device)
            labels = y.view(-1, 1).to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)  # Accumulate loss

            # Calculate accuracy
            predictions = (outputs > 0.5).float()  # Threshold for binary classification
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Compute average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy




# Example usage
TRAIN_FILE = "saved_games/2024-10-31_16:56_(path-search | path-search)_rounds_100.pkl"
split = 0.8
TRAIN_NUM_DATA_SAMPLES = 1000
TRAIN_BATCH_SIZE = 8
EPOCHS = 30
data_loader = generate_synthetic_data(TRAIN_NUM_DATA_SAMPLES, TRAIN_FILE, TRAIN_BATCH_SIZE, split)

from architectures.transformer import SimpleTransformer
# Instantiate the model and move it to GPU
d_model = 32
num_heads = 2
num_layers = 2
dim_feedforward = 64
for inp, out in data_loader:
    max_len = inp.shape[1]
    break

model = SimpleTransformer(tokenizer, d_model, num_heads, num_layers, dim_feedforward, max_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # replace model with your model instance
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.00001)

# import code; code.interact(local = locals())
try:
    model.load_state_dict(torch.load("trained_model.pth"))
except:
    train_model(data_loader, model, optimizer, criterion, scheduler, EPOCHS, "trained_model.pth")




def test_vis(model, criterion, file):
    '''
    Prints the game board and the model prediction
    '''
    g = Game_data(file)
    rounds = getNumRounds(file)

    for _ in range(3):
        # Select random round and line within the round
        rand_round = random.randint(int(split*rounds + 1), rounds - 1)
        rand_line = random.randint(0, len(g.data[rand_round]) - 1)

        # Extract features for the selected round and line
        hor_wall_tensor, ver_wall_tensor, player_pos_tensor, p1_walls, p2_walls, p1_won, next_move = g.getRoundLine(rand_round, rand_line)
        player1_turn = rand_line % 2 == 0  # Alternate turns based on line number

        # Encode features for the model and add to the dataset
        encoded_features = tokenizer.encode(hor_wall_tensor, ver_wall_tensor, player_pos_tensor, p1_walls, p2_walls, player1_turn)

        game = Game()
        gamestate = GameState(player_pos_tensor.shape[0], 0)
        gamestate.player1_walls_num = p1_walls
        gamestate.player2_walls_num = p2_walls

        player1_pos = torch.nonzero(player_pos_tensor == 1, as_tuple=True)
        player2_pos = torch.nonzero(player_pos_tensor == -1, as_tuple=True)
        gamestate.player1_pos = (player1_pos[0].item(), player1_pos[1].item())
        gamestate.player2_pos = (player2_pos[0].item(), player2_pos[1].item())

        gamestate.horwalls = hor_wall_tensor
        gamestate.verwalls = ver_wall_tensor

        gamestate.player1 = player1_turn
        game.game_state = gamestate
        game.print_game_stats()
        game.print_board()
        print(f"Player: {'p1' if player1_turn else 'p2'}")
        print(f"Turn: {rand_line}")
        # print(1*hor_wall_tensor)
        # print(1*ver_wall_tensor)
        # print(player_pos_tensor)


        with torch.no_grad():
            X = encoded_features.to(device).unsqueeze(0)
            y = torch.tensor([[float(p1_won)]]).to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            print(f"Actual:   \t {y.item()}")
            print(f"Predicted:\t {outputs.item()}")
            print(f"Loss:     \t {loss.item()}")

test_loader = test_vis(model, criterion, TRAIN_FILE)