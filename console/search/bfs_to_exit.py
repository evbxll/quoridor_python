from copy import copy
import numpy as np
from console.util.priority_queue_item import PriorityQueueItem
from collections import deque


dirs = [(1,0), (0,1), (0,-1), (-1,0)]

def bfs_shortest_to_exit(game_state):
    p1 = tuple(game_state.player1_pos)
    p2 = tuple(game_state.player2_pos)

    dist1 = bfs_dist(game_state, p1, 0, p2)
    dist2 = bfs_dist(game_state, p2, game_state.size - 1, p1)
    return (dist1, dist2)
    
def bfs_dist(game_state, start, end_row, other_player_pos):

    if start[0] == end_row:
        return float('-inf')
    visited = {start,}
    queue = deque([(start, 0)])  # (node, path_length)

    while queue:
        pos, path_length = queue.popleft()
        for x,y in [ (pos[0]+i, pos[1] + j) for i,j in dirs]:
            # if in bounds and NO wall between player and new_pos
            if 0 <= x < game_state.size and 0 <= y < game_state.size and not game_state.is_wall_blocking_path(pos, (x,y)):
                if x == end_row:
                    return path_length + 1
                elif (x,y) not in visited:
                    visited.add((x,y))
                    queue.append(((x,y), path_length + 1))
    
    return float('inf')




