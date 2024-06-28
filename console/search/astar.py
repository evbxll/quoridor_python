import numpy as np
import heapq



class Node:
    def __init__(self, x, y, f_score):
        self.x = x
        self.y = y
        self.f_score = f_score

    def __lt__(self, other):
        return self.f_score < other.f_score

def z_asearch(game_state, start_pos : tuple, end_row : int, other_player : tuple):

    base_val = -1
    start_val = 0

    if start_pos[0] == end_row:
        return base_val

    size = game_state.size

    flip_dirs = 1 if end_row == 0 else -1
    directions = flip_dirs * np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # Up, Down, Left, Right

    visited = np.zeros((size, size), dtype=bool)
    p_queue = []
    heapq.heappush(p_queue, Node(start_pos[0], start_pos[1], 0))

    g_score = np.full((size, size), np.inf)
    g_score[start_pos[0], start_pos[1]] = start_val

    while p_queue:
        current = heapq.heappop(p_queue)
        x, y = current.x, current.y

        if x == end_row:
            return current.f_score

        if not visited[x, y]:
            visited[x, y] = True

            for dx, dy in directions:
                new_x = x + dx
                new_y = y + dy

                if 0 <= new_x < size and 0 <= new_y < size and game_state.is_valid_move((x, y), (new_x, new_y)):
                    tentative_g_score = g_score[x, y] + 1


                    ## check if have parity to jump, and its our move
                    if (tentative_g_score - start_val) < 3 and (new_x, new_y) == other_player and new_x != end_row and (tentative_g_score - start_val) % 2 == 1:
                        if (game_state.player1 and end_row == 0): # player1 to move, we are player 1
                            tentative_g_score = g_score[x, y]
                        elif (not game_state.player1 and end_row != 0): #player2 to move, we are player 2
                            tentative_g_score = g_score[x, y]

                    if tentative_g_score < g_score[new_x, new_y]:
                        g_score[new_x, new_y] = tentative_g_score
                        f_score = tentative_g_score + abs(new_x - end_row)
                        heapq.heappush(p_queue, Node(new_x, new_y, f_score))

    return np.inf

def astar_search(game_state):
    '''
    Calculates path distance from start_pos to end_row, accounting for other_player and whether a jump can happen using skip_parity

    game_state is a GameState type
    '''
    p1 = z_asearch(game_state, game_state.player1_pos, 0, game_state.player2_pos)
    p2 = z_asearch(game_state, game_state.player2_pos, game_state.size - 1, game_state.player1_pos)
    return p1, p2