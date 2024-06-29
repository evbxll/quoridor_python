def output_space(board_size: int) -> int:
    player_moves = board_size**2
    placements = 2 * (board_size - 1)**2
    return player_moves + placements

def move_to_int(board_size: int, move: list) -> int:
    if len(move) == 2:
        return move[0] * board_size + move[1]
    elif len(move) == 3:
        player_move_base = board_size**2
        is_horizontal_base = move[2] * (board_size - 1)**2
        index_base = move[0] * (board_size - 1) + move[1]
        return player_move_base + is_horizontal_base + index_base
    else:
        raise ValueError("move is not valid")

def int_to_move(board_size: int, move_int: int) -> list:
    if move_int >= output_space(board_size) or move_int < 0:
        raise ValueError("move_int is out of valid range")

    # Player move
    if move_int < board_size**2:
        row = move_int // board_size
        col = move_int % board_size
        return [row, col]
    else:  # Placement move
        placement_int = move_int - board_size**2
        is_horizontal = 0 if placement_int < (board_size - 1)**2 else 1
        placement_int = placement_int % ((board_size - 1)**2)
        row = placement_int // (board_size - 1)
        col = placement_int % (board_size - 1)
        return [row, col, is_horizontal]



# import code; code.interact(local = locals())