def simple_state_eval(game_state, is_player_1_minimax):
    if is_player_1_minimax:
        return  -game_state.player1_pos[0]
    else:
        return -(game_state.rows - game_state.player2_pos[0])

