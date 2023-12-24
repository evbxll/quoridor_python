def simple_path_finding_heuristic(game_state):
    if game_state.player1:
        return 100 * abs(game_state.player1_pos[0])
    else:
        return 100 * (abs(game_state.player2_pos[0] - 16))
