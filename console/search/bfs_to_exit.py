from collections import deque


dirs = [(1,0), (0,1), (0,-1), (-1,0)]

def bfs_shortest_to_exit(game_state, path_none):
    p1 = tuple(game_state.player1_pos)
    p2 = tuple(game_state.player2_pos)

    
    
    def bfs_dist(game_state, start, end_row, other_player_pos, skip_other_player_parity = 0):

        if start[0] == end_row:
            return path_none
        visited = {start,}
        queue = deque([(start, 0)])  # (node, path_length)

        possible_skip_other_len = float('inf')

        while queue:
            pos, path_length = queue.popleft()
            for x,y in [ (pos[0]+i, pos[1] + j) for i,j in dirs]:
                # if in bounds and NO wall between player and new_pos
                if 0 <= x < game_state.size and 0 <= y < game_state.size and not game_state.is_wall_blocking_path(pos, (x,y)):
                    if x == end_row:
                        return min(path_length + 1, possible_skip_other_len)
                    elif (x,y) not in visited:
                        visited.add((x,y))
                        if other_player_pos == (x,y) and path_length%2 == skip_other_player_parity:
                            possible_skip_other_len = path_length + bfs_dist(game_state, other_player_pos, end_row, (-1,-1))
                        queue.append(((x,y), path_length + 1))
        
        return float('inf')
    
    dist1 = bfs_dist(game_state, p1, 0, p2, 1*(not game_state.player1)) - 1*(game_state.player1)
    dist2 = bfs_dist(game_state, p2, game_state.size - 1, p1, 1*(game_state.player1)) - 1*(not game_state.player1)
    return (dist1, dist2)




