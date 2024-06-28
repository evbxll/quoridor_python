from console.game import Game
import os
import pickle
from datetime import datetime


def log_game(p1, p2, data, rounds):
    current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
    folder_name = f"{current_date}_({p1} | {p2})"
    prev_folder = "saved_games/"

    # Create directory/folder
    folder_path = os.path.join(prev_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Create and write to a file inside the directory
    file_path = os.path.join(folder_path, f"rounds_{rounds}.pkl")
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully created'{file_path}'")
        return 0
    except IOError as e:
        print(f"Error creating file: {file_path}")
        print(e)
        return 1


if __name__ == '__main__':

    VERBOSE = 0
    DELAY = 0#0.5

    g = Game(VERBOSE, 100, DELAY)
    g.play()
    print(g.player_simulation_algorithms)
    print(g.wins)

    t1 = sum([sum([i for i in r[0]]) for r in g.execution_times])
    l1 = sum([len(r[0]) for r in g.execution_times])

    t2 = sum([sum([i for i in r[1]]) for r in g.execution_times])
    l2 = sum([len(r[1]) for r in g.execution_times])

    print("Average ex: ", t1*1.0/l1, t2*1.0/l2)
    log_game(g.player_simulation_algorithms[0], g.player_simulation_algorithms[1], g.hist_per_round, len(g.hist_per_round))
