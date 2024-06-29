import pickle
from typing import Tuple
import numpy as np

class Game_data:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = []


        self.initialize_game()

    def initialize_game(self):
        with open(self.file_path, 'rb') as file:
            game_data = pickle.load(file)

        rows = len(game_data)  # Number of rows (outer list length)
        columns = [len(row) for row in game_data]
        print(f"Number of rows (outer list length): {rows}")
        print(f"Number of columns (inner list lengths): {columns}")
        print(f"Size {game_data[0][0][0]}\nWalls {game_data[0][0][1]}")

        self.data = game_data

    def getRoundLine(self, round, stopline):

        game = self.data[round][1:]

        board_size, walls = self.data[round][0]
        horizontal_placement = np.zeros((board_size, board_size), dtype=bool)
        vertical_placement = np.zeros((board_size, board_size), dtype=bool)
        player_pos = np.zeros((board_size, board_size), dtype=int)
        p1_walls_left = walls
        p2_walls_left = walls


        for i, action in enumerate(game):
          if (i >= stopline):
              break
          if len(action) == 2:
            if i % 2 == 0: # player 1
                # Unset the other value of 1 in the matrix
                player_pos[player_pos == 1] = 0
                player_pos[action[0], action[1]] = 1
            else:
                # Unset the other value of -1 in the matrix
                player_pos[player_pos == -1] = 0
                player_pos[action[0], action[1]] = -1
          elif len(action) == 3:
              x, y, placement_type = action
              if placement_type == 0:
                  horizontal_placement[x, y] = True
              elif placement_type == 1:
                  vertical_placement[x, y] = True

              # decrement walls left
              if i % 2 == 0:
                  p1_walls_left -= 1
              else:
                  p2_walls_left -= 1

        horizontal_int = horizontal_placement.astype(int)
        vertical_int = vertical_placement.astype(int)
        board_tensor = np.stack((horizontal_int, vertical_int, player_pos), axis=-1)

        return (board_tensor, p1_walls_left, p2_walls_left)

d = Game_data('/home/linux_root/GithubProjs/quoridor_python/saved_games/2024-06-28_21:09_(path-search | path-search)_rounds_1000.pkl')
r=d.getRoundLine(100,14)

# import code; code.interact(local = locals())
