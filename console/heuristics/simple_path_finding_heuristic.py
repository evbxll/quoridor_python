from console.search.bfs_to_exit import bfs_shortest_to_exit

def simple_path_finding_heuristic(game_state, func=None):
    """calculates path lengths for p1 and p2, and uses to calculate REWARD. 
    Defaults to p2 - p1, so reward is better when p1 is winning

    Args:
        game_state (GameState): the game state instance
        func (function, optional): the function which takes in (p1,p2) and return reward. Defaults to None.

    Returns:
        float or int: the reward
    """
    p1, p2 = bfs_shortest_to_exit(game_state)
    if func:
        reward = func(p1,p2)
    else:
        reward = p2 - p1
    return reward
