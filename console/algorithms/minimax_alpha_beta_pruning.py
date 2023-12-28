from console.game_state import GameState, WallPieceStatus
from console.heuristics.simple_path_finding_heuristic import simple_path_finding_heuristic

import math


def minimax_alpha_beta_pruning(game_state: GameState, depth, alpha, beta, is_max):

    if depth == 0 or game_state.is_end_state():
        reward_func = lambda p1, p2: (p2-p1)
        if_failed_return = float('inf') if is_max else float('-inf')
        return simple_path_finding_heuristic(game_state, reward_func, if_failed_return)

        
    if is_max:
        max_eval = -math.inf
        for child in game_state.get_all_child_states(False, depth>1):
            ev = minimax_alpha_beta_pruning(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        
        return max_eval if max_eval != -math.inf else math.inf
    else:
        min_eval = math.inf
        for child in game_state.get_all_child_states(False, depth>1):
            ev = minimax_alpha_beta_pruning(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval if min_eval != math.inf else -math.inf