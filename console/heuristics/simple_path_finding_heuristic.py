from console.search.bfs_to_exit import bfs_shortest_to_exit

def simple_path_finding_heuristic(game_state, func=None):
    p1, p2 = bfs_shortest_to_exit(game_state)
    if func:
        ev = func(p1,p2)
    else:
        ev = p1 - p2
    return ev
