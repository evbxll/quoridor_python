from copy import copy
from console.util.priority_queue_item import PriorityQueueItem
from queue import PriorityQueue


dirs = [(1,0), (0,1), (0,-1), (-1,0)]

def dfs_check_if_exit_paths_exist(game_state):
    p1 = dfs_wrapper(game_state, tuple(game_state.player1_pos), -1, 0)
    p2 = dfs_wrapper(game_state, tuple(game_state.player2_pos), 1, game_state.size - 1)

    return p1 and p2
    
def dfs_wrapper(game_state, pos, direc, end_row):

    def dfs(pos, visited=set()):
        visited.add(pos)
        if pos[0] == end_row:
            return True
        for x,y in [ (pos[0]+(direc*i), pos[1] + j) for i,j in dirs ]:
            if (x,y) not in visited:
                # if in bounds and NO wall between player and new_pos
                if 0 <= x < game_state.size and 0 <= y < game_state.size and not game_state.is_wall_blocking_path(pos, (x,y)):
                    if dfs((x,y), visited):
                        return True
        return False

    # temp = set()
    # r = dfs(pos, temp)
    # if len(temp) > (game_state.size**2)//4:
    #     print(len(temp))
    return dfs(pos)




