import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim

from helper import getNumRounds, COLOR
from tokenizer import Tokenizer
from console.game_state import GameState
from architectures.transformer import SimpleTransformer

class NN_Bot:
    def __init__(self, is_p1, SIZE = 9):

        self.is_p1 = is_p1
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        tokenizer = Tokenizer()
        self.d_model = 32
        self.num_heads = 4
        self.num_layers = 2
        self.dim_feedforward = 32
        self.max_len = (SIZE**2) + 2*(SIZE * (SIZE-1)) + 3

        model = SimpleTransformer(tokenizer, 
                                  self.d_model, 
                                  self.num_heads, 
                                  self.num_layers, 
                                  self.dim_feedforward, 
                                  self.max_len
                                  ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        self.device = device
        self.model = model
        self.load_model()

        self.tokenizer = tokenizer
        self.criterion = criterion
        self.optimizer = optimizer

        self.decision_states = []  # Stores the states of decisions made during the game

        

    def gamestate_to_tensor(self, g: GameState):
        # Encode features for the model and add to the dataset
        size = g.horwalls.shape[0]+1
        h_walls = g.horwalls.clone()
        v_walls = g.verwalls.clone()
        
        player_pos_tensor = torch.zeros((size, size))
        x,y = g.player1_pos
        player_pos_tensor[x,y] = 1
        x,y = g.player2_pos
        player_pos_tensor[x,y] = -1

        p1_walls = g.player1_walls_num
        p2_walls = g.player2_walls_num

        p1_turn = g.player1

        t = self.tokenizer.encode(h_walls, v_walls, player_pos_tensor, p1_walls, p2_walls, p1_turn)
        return t


    def best_move(self, game_state : GameState):
        next_states = []
        next_moves = []

        moves = game_state.get_available_moves()
        for move in moves:
            temp_pos = game_state.get_current_player_pos()

            # Make and evaluate the move
            game_state.move_piece(move, False)
            game_state.player1 = not game_state.player1
            
            state = self.gamestate_to_tensor(game_state)
            next_states.append(state)
            next_moves.append(move)

            # Undo the move
            game_state.player1 = not game_state.player1
            game_state.move_piece(temp_pos, False)

        # Evaluate wall placements
        wall_placements = game_state.get_available_wall_placements()
        for wall in wall_placements:
            x, y, is_horizontal = wall
            pos = (x, y)

            # Temporarily place the wall
            game_state.temp_set_wall(pos, is_horizontal)
            game_state.player1 = not game_state.player1
            
            state = self.gamestate_to_tensor(game_state)
            next_states.append(state)
            next_moves.append(wall)

            # Undo the wall placement
            game_state.player1 = not game_state.player1
            game_state.unset_wall(pos)
            game_state.saved_wall_placements = wall_placements  # Reinstate available wall placements

        X = torch.stack(next_states).to(self.device)
        outputs = self.model(X)
        if self.is_p1:
            best_index = torch.argmax(outputs).item()
        else:
            best_index = torch.argmin(outputs).item()

        self.decision_states.append(next_states[best_index])
        self.decision_states.append(self.gamestate_to_tensor(game_state))
        return next_moves[best_index]


    def update_model(self, p1_won : bool):
        '''
        Perform updates to model weights based on game outcome
        '''
        if not self.decision_states:
            return

        X = torch.stack(self.decision_states).to(self.device)

        y = torch.full((X.size(0),), float(p1_won), dtype=torch.float32).to(self.device)

        # Forward pass: compute predictions
        predictions = self.model(X).squeeze()  # Shape: (n,)

        # Compute the loss
        loss = self.criterion(predictions, y)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear decision states after updating
        self.decision_states.clear()

    def model_to_filename(self):
        return f"model_(p1-{self.is_p1})_(device-{self.device})_(dmodel-{self.d_model})_(heads-{self.num_heads})_(layers-{self.num_layers})_(ff-{self.dim_feedforward})_(maxlen-{self.max_len}).pt"

    def save_model(self):
        self.filename = self.model_to_filename()
        torch.save(self.model.state_dict(), self.filename)
        print(f"Model saved to {self.filename}")
    
    def load_model(self):
        try:
            model_file_name = self.model_to_filename()
            self.model.load_state_dict(torch.load(model_file_name))
            print(f'Model loaded: "{model_file_name}"')
        except:
            print('No model loaded')
