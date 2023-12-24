from console.states.game_state import GameState
from collections import defaultdict
import random
import numpy as np


def randombot_action(game_state: GameState):

    move_prob = 0.5

    wall_placements = game_state.get_available_wall_placements(False)
    available_moves = game_state.get_available_moves(False)
    # print(random.sample(wall_placements + available_moves, k=10))

    if not wall_placements:
        move_prob = 1.0

    # Randomly select from list1 or list2 based on the probabilities
    chosen_list = random.choices([available_moves, wall_placements], weights=[move_prob, 1-move_prob], k=1)[0]

    # Randomly select an item from the chosen list
    chosen_action = random.choice(chosen_list)

    return chosen_action
