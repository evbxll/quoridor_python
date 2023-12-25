from console.states.game_state import GameState
import math
from console.heuristics.state_evaluation_heuristic import state_evaluation_heuristic
from console.heuristics.simple_path_finding_heuristic import simple_path_finding_heuristic


def minimax_alpha_beta_pruning(game_state: GameState, depth, alpha, beta, is_max):
    print(depth)
    if depth == 0:
        return simple_path_finding_heuristic(game_state)
        
    if is_max:
        max_eval = -math.inf
        for child in game_state.get_all_child_states():
            ev = minimax_alpha_beta_pruning(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, ev)
            alpha = max(alpha, ev)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for child in game_state.get_all_child_states():
            ev = minimax_alpha_beta_pruning(child, depth-1, alpha, beta, True)
            min_eval = min(min_eval, ev)
            beta = min(beta, ev)
            if beta <= alpha:
                break
        return min_eval

