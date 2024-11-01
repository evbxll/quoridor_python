
import torch

from dataclasses import dataclass

tokens = [

      "h_wall",
      "no_h_wall",

      "v_wall",
      "no_v_wall",

      "empty_cell",
      "p1_cell",
      "p2_cell",

      "p1_turn",
      "p2_turn",
    ] + list(range(20)) # this is for wall size purposes

class Tokenizer:
  def __init__(self):
    self.token_to_ind = dict([(e,i) for i,e in enumerate(tokens)])
    self.ind_to_token = dict([e for e in enumerate(tokens)])

  def __len__(self):
    return len(self.token_to_ind)

  def encode(self, h_walls : torch.Tensor, v_walls: torch.Tensor, player_pos: torch.Tensor, p1_walls : int, p2_walls: int, p1_turn: bool):
    assert h_walls.shape[0] == h_walls.shape[1]
    assert h_walls.shape[0] == v_walls.shape[1]
    assert v_walls.shape[0] == v_walls.shape[1]

    d = h_walls.shape[0]
    false_column = torch.zeros((d, 1), dtype=torch.bool)
    new_h = torch.cat((false_column, h_walls), dim=1) | torch.cat((h_walls, false_column), dim=1)

    false_row = torch.zeros((1,d), dtype=torch.bool)
    new_v = torch.cat((false_row, v_walls), dim=0) | torch.cat((v_walls, false_row), dim=0)

    flat_h = torch.where(new_h.flatten(), self.token_to_ind['h_wall'], self.token_to_ind['no_h_wall'])
    flat_v = torch.where(new_v.flatten(), self.token_to_ind['v_wall'], self.token_to_ind['no_v_wall'])

    flat_p = torch.full_like(player_pos, fill_value=self.token_to_ind['empty_cell'])
    flat_p[player_pos == 1] = self.token_to_ind['p1_cell']
    flat_p[player_pos == -1] = self.token_to_ind['p2_cell']
    flat_p = flat_p.flatten()

    p1_wal = torch.tensor([self.token_to_ind[p1_walls]])
    p2_wal = torch.tensor([self.token_to_ind[p2_walls]])
    player_turn = torch.tensor([self.token_to_ind['p1_turn' if p1_turn else 'p2_turn']])


    return torch.cat((flat_h, flat_v, flat_p, p1_wal, p2_wal, player_turn)).int()


# h = torch.tensor([[True, False, False], [True, False, True], [False, False, False]])
# p = torch.tensor([[0, 0, 1], [0, 0, 0], [0, -1, 0]])
# t = Tokenizer()
# print(t.encode(h,h,p, 10, 3, True))