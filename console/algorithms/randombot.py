from console.game_state import GameState
import random

def randombot_action(game_state: GameState, move_prob : float):
    '''
    Makes a random move, prefers to move with prob move_prob
    '''

    wall_placements = game_state.get_available_wall_placements()
    available_moves = game_state.get_available_moves()
    available_moves = sorted(available_moves, key=lambda x: x[0], reverse = not game_state.player1)

    if not wall_placements:
        move_prob = 1.0

    # Randomly select from list1 or list2 based on the probabilities
    chosen_list = random.choices([available_moves, wall_placements], weights=[move_prob, 1-move_prob], k=1)[0]

    # print(chosen_list)
    if chosen_list == available_moves:
        first_move_prob = 0.5
        probs = [first_move_prob] + [(1-first_move_prob)/ (max(len(chosen_list) - 1, 1))] * (len(chosen_list) - 1)

        if game_state.player1:
            if available_moves[0][0] < game_state.player1_pos[0]:
                chosen_action = random.choices(chosen_list, weights=probs, k=1)[0]
            else:
                chosen_action = random.choice(chosen_list)
        else:
            if available_moves[0][0] > game_state.player2_pos[0]:
                chosen_action = random.choices(chosen_list, weights=probs, k=1)[0]
            else:
                chosen_action = random.choice(chosen_list)
    else:
        chosen_action = random.choice(chosen_list)

    return chosen_action
