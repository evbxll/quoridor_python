from console.search.bfs_to_exit import bfs_shortest_to_exit

def simple_path_finding_heuristic(game_state):
    p1, p2 = bfs_shortest_to_exit(game_state)
    ev = p1 - p2
    return ev
