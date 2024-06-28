from console.game_state import GameState
from console.search.astar import astar_search

import numpy as np

import math

def z_minimax_helper(g : GameState, func, is_max : bool):
    p1, p2 = astar_search(g)
    if p1 == float('inf') or p2 == float('inf'):
        if is_max:
            return np.inf
        else:
            return -np.inf
    reward = func(p1,p2)
    return reward

def minimax_search(game_state : GameState, depth : int, alpha : float, beta : float, is_max : bool):

    if depth <= 0 or game_state.is_end_state():
        reward_func = lambda p1, p2: (p2-p1)
        return z_minimax_helper(game_state, reward_func, is_max)

    # max player
    if is_max:
        max_eval = -math.inf

        # moves
        moves = game_state.get_available_moves()
        for move in moves:

            # alpha beta prune
            if (beta <= alpha):
                break

            temp_pos = game_state.get_current_player_pos()

            # step into make move
            game_state.move_piece(move, False)
            game_state.player1 = not game_state.player1

            ev = minimax_search(game_state, depth - 1, alpha, beta, False)

            # undo move
            game_state.player1 = not game_state.player1
            game_state.move_piece(temp_pos, False)

            max_eval = max(max_eval, ev)
            alpha = max(alpha, max_eval)

        # wall placements
        wall_placements = game_state.get_available_wall_placements()
        for wall in wall_placements:

            # alpha beta prune
            if (beta <= alpha):
                break

            x, y, is_horizontal = wall
            pos = (x,y)

            game_state.temp_set_wall(pos, is_horizontal)
            game_state.player1 = not game_state.player1

            ev = minimax_search(game_state, depth - 1, alpha, beta, False)

            game_state.player1 = not game_state.player1
            game_state.unset_wall(pos)

            game_state.saved_wall_placements = wall_placements  # Reinstate free walls

            max_eval = max(max_eval, ev)
            alpha = max(alpha, max_eval)

        return max_eval if max_eval != -math.inf else math.inf

    # minimize player
    else:
        min_eval = math.inf

        # moves
        moves = game_state.get_available_moves()
        for move in moves:

            # alpha beta prune
            if (beta <= alpha):
                break

            temp_pos = game_state.get_current_player_pos()

            # step into make move
            game_state.move_piece(move, False)
            game_state.player1 = not game_state.player1

            ev = minimax_search(game_state, depth - 1, alpha, beta, True)

            # undo move
            game_state.player1 = not game_state.player1
            game_state.move_piece(temp_pos, False)

            min_eval = min(min_eval, ev)
            beta = min(beta, min_eval)

        # wall placements
        wall_placements = game_state.get_available_wall_placements()
        for wall in wall_placements:

            # alpha beta prune
            if (beta <= alpha):
                break

            x, y, is_horizontal = wall
            pos = (x,y)

            game_state.temp_set_wall(pos, is_horizontal)
            game_state.player1 = not game_state.player1

            ev = minimax_search(game_state, depth - 1, alpha, beta, True)

            game_state.player1 = not game_state.player1
            game_state.unset_wall(pos)

            game_state.saved_wall_placements = wall_placements  # Reinstate free walls

            min_eval = min(min_eval, ev)
            beta = min(beta, min_eval)

        return min_eval if min_eval != math.inf else -math.inf
