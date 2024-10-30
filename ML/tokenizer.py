
import torch

class Tokenizer:
  tokens = [
    "h_wall",
    "blank_h_wall",

    "v_wall",
    "blank_v_wall",

    "empty_cell",
    "p1_cell",
    "p2_cell",

    "p1_walls",#remember to multiply these embeddings by the number of walls
    "p2_walls",

    "p1_turn",
    "p2_turn",
  ]

  def __init__(self):
    pass

  def encode(self, h_walls : torch.Tensor, v_walls: torch.Tensor, player_pos: torch.Tensor):
    assert h_walls.shape[0] == h_walls.shape[1]
    assert h_walls.shape[0] == v_walls.shape[1]
    assert v_walls.shape[0] == v_walls.shape[1]
    h_walls.flatten()