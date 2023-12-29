from console.search.bfs_to_exit import bfs_shortest_to_exit

def simple_path_finding_heuristic(game_state, func=None, if_failed_what_to_return = float('-inf'), path_none = -1):
    """calculates path lengths for p1 and p2, and uses to calculate REWARD. 
    Defaults to p2 - p1, so reward is better when p1 is winning

    Args:
        game_state (GameState): the game state instance
        func (function, optional): the function which takes in (p1,p2) and return reward. Defaults to None.

    Returns:
        float or int: the reward
    """
    p1, p2 = bfs_shortest_to_exit(game_state, path_none)
    if p1 == float('inf') or p2 == float('inf'):
        return if_failed_what_to_return
    if func:
        reward = func(p1,p2)
    else:
        reward = p2 - p1
    return reward
