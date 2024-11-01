from console.game import Game
import os
import pickle
from datetime import datetime


def log_game(p1, p2, data, rounds):
    size = data[0][0][0]
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_path = f"{current_date}_({p1} - {p2})_rounds-{rounds}_size-{size}.pkl"
    prev_folder = "saved_games/"

    # Create directory/folder
    file_path = os.path.join(prev_folder, file_path)
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully created'{file_path}'")
        return 0
    except IOError as e:
        print(f"Error creating file: {file_path}")
        print(e)
        return 1


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script with command-line arguments.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--r', type=int, required=True, help='Number of rounds')
    parser.add_argument('--s', type=float, default=0.0, help='Move delay time (default: 0.1)')
    parser.add_argument('--b', type=int, default=9, help='Board size, n x n')
    parser.add_argument('-n', "--nosave", action='store_true', help='Prevent game file saving')



    args = parser.parse_args()

    VERBOSE = 1 if args.verbose else 0
    GAMES = args.r
    DELAY = args.s
    SIZE = args.b
    NOSAVE = 1 if args.nosave else 0

    g = Game(VERBOSE, GAMES, DELAY, SIZE)
    g.initialize_sim()
    g.play()
    print(g.player_simulation_algorithms)
    print(g.wins)

    t1 = sum([sum([i for i in r[0]]) for r in g.execution_times])
    l1 = sum([len(r[0]) for r in g.execution_times])

    t2 = sum([sum([i for i in r[1]]) for r in g.execution_times])
    l2 = sum([len(r[1]) for r in g.execution_times])

    print("Average ex: ", t1*1.0/l1, t2*1.0/l2)
    if not NOSAVE:
        log_game(g.player_simulation_algorithms[0], g.player_simulation_algorithms[1], g.hist_per_round, len(g.hist_per_round))
