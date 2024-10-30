import numpy as np
import random
import time

from console.game_state import GameState, WallColor
from console.util.color import Color
from console.algorithms.minimax_alpha_beta_pruning import minimax_search
from console.search.astar import astar_search
from console.algorithms.randombot import randombot_action



DEFAULT_WALL_COLOR = Color.PINK
PLAYER1COLOR = Color.LIGHT_GREEN
PLAYER2COLOR = Color.LIGHT_RED

SIZE = 5
# y=a_{0}+a_{1}x+a_{2}x^{2}+a_{3}x^{3}, a_{0}=-6.4276, a_{1}=3.4745, a_{2}=-0.4058, a_{3}=0.0255
WALLS = 10 if SIZE == 9 else round(-6.4276 + 3.4745*SIZE + -0.4058*(SIZE**2) + 0.0255 * (SIZE**3))

MoveKeyValues = "".join([str(i) for i in range(SIZE)])
WallKeyValues = "".join([chr(ord('a') + i).upper() for i in range(SIZE-1)])

class Game:
    def __init__(self, verbose = True, rounds = 1, sim_delay = 0.5):

        self.player_simulation_algorithms = ["randomBot", "randomBot"]
        self.game_state = GameState(SIZE, WALLS)
        self.verbose = verbose
        self.algorithms = ["randomBot", "impatientBot", "minimax-alpha-beta-pruning", "path-search"]
        self.execution_times = [[[],[]]]
        self.sim_delay = sim_delay
        self.rounds = rounds
        self.wins = [0,0]
        self.hist_per_round = [[(SIZE, WALLS)],]

        # self.quick_run("path-search", "path-search")
        self.initialize_sim()


    def quick_run(self, bot1, bot2):
        Game.print_colored_output("### Quick Running rounds ###", Color.CYAN)
        self.player_simulation_algorithms[0] = bot1
        self.player_simulation_algorithms[1] = bot2
        Game.print_colored_output("Chosen algorithm for player 1 is {0:30}".format(self.player_simulation_algorithms[0].upper()), Color.CYAN)
        Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(self.player_simulation_algorithms[1].upper()), Color.CYAN)


    def initialize_sim(self):
        Game.print_colored_output("### WELCOME TO QUORIDOR ###", Color.CYAN)
        print("\n")
        print("Choose the players algorithms[first_player, second_player]")
        for i, player in enumerate(self.algorithms):
            print(f"{str(i+1)}) {player}")
        while True:
            x = input("\nChoose: ")
            # x = "1,2"
            if not len(x.split(",")) == 2 and x != "x" and x != "X":
                Game.print_colored_output("Illegal input!", Color.RED)
            elif x == "x" or x == "X":
                exit(0)
            else:
                one, two = x.split(",")
                if 0 <= int(one) - 1 < len(self.algorithms) and 0 <= int(two) - 1 < len(self.algorithms):
                    self.player_simulation_algorithms[0] = self.algorithms[int(one) - 1]
                    self.player_simulation_algorithms[1] = self.algorithms[int(two) - 1]
                    Game.print_colored_output("Chosen algorithm for player 1 is {0:30}".format(
                        self.player_simulation_algorithms[0].upper()), Color.CYAN)
                    Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(
                        self.player_simulation_algorithms[1].upper()), Color.CYAN)
                    break
                else:
                    Game.print_colored_output("Illegal input!", Color.RED)


    def choose_best_from_actions(self, d : dict):
        '''
        returns a move with the highest value
        '''
        if len(d.keys()) == 0:
            return None

        max_value = max(d.values())

        top_actions = [action for action, value in d.items() if value == max_value]
        moves_in_top_actions = [t for t in top_actions if len(t) == 2]
        if moves_in_top_actions:
            top_actions = moves_in_top_actions
        best_action = random.choice(top_actions)

        return best_action


    def minimax_agent(self, game_state : GameState, is_player1_turn, depth = 2):
        alpha = -np.inf
        beta = np.inf
        flip = 1 if is_player1_turn else -1
        best_val = -np.inf

        d = {}
        game_state.check_wall_blocks_exit_on_gen = False

        moves = game_state.get_available_moves()
        for move in moves:

            temp_pos = game_state.get_current_player_pos()

            # step into make move
            game_state.move_piece(move, False)
            game_state.player1 = not game_state.player1

            ev = flip * minimax_search(game_state, depth - 1, alpha, beta, game_state.player1)

            # undo move
            game_state.player1 = not game_state.player1
            game_state.move_piece(temp_pos, False)

            if (ev >= best_val):
                best_val = ev
                d[move] = ev

        # wall placements
        wall_placements = game_state.get_available_wall_placements()
        for wall in wall_placements:

            x, y, is_horizontal = wall
            pos = (x,y)

            game_state.temp_set_wall(pos, is_horizontal)
            game_state.player1 = not game_state.player1

            ev = flip * minimax_search(game_state, depth - 1, alpha, beta, game_state.player1)

            game_state.player1 = not game_state.player1
            game_state.unset_wall(pos)

            game_state.saved_wall_placements = wall_placements  # Reinstate free walls

            if (ev >= best_val):
                best_val = ev
                d[wall] = ev

        game_state.check_wall_blocks_exit_on_gen = True
        return self.choose_best_from_actions(d)


    def pathsearch_agent(self, game_state : GameState, is_player1_turn):


        flip = 1 if is_player1_turn else -1
        best_val = -np.inf
        d = {}

        max_reward_func = lambda p1,p2, flip : (6*p2-5*p1) if flip == 1 else (6*p1-5*p2)

        moves = game_state.get_available_moves()
        for move in moves:

            temp_pos = game_state.get_current_player_pos()

            # step into make move
            game_state.move_piece(move, False)
            game_state.player1 = not game_state.player1

            p1, p2 = astar_search(game_state)

            # undo move
            game_state.player1 = not game_state.player1
            game_state.move_piece(temp_pos, False)

            ev = max_reward_func(p1,p2, flip)

            if (ev >= best_val):
                best_val = ev
                d[move] = ev

        # wall placements
        wall_placements = game_state.get_available_wall_placements()
        for wall in wall_placements:

            x, y, is_horizontal = wall
            pos = (x,y)

            game_state.temp_set_wall(pos, is_horizontal)
            game_state.player1 = not game_state.player1

            p1, p2 = astar_search(game_state)

            game_state.player1 = not game_state.player1
            game_state.unset_wall(pos)

            game_state.saved_wall_placements = wall_placements  # Reinstate free walls

            ev = max_reward_func(p1,p2, flip)

            if (p1 != np.inf and p2 != np.inf):
                if (ev >= best_val):
                    best_val = ev
                    d[wall] = ev

        return self.choose_best_from_actions(d)

    def randombot_agent(self, game_state):
        action = randombot_action(game_state, 0.4)
        return action

    def impatientbot_agent(self, game_state):
        action = randombot_action(game_state, 0.7)
        return action

    def player_simulation(self):
        index = 1*(not self.game_state.player1)
        player_number = index + 1


        if self.verbose:
            print("Player {} ({}) is thinking...".format(player_number, self.player_simulation_algorithms[index]), end='', flush = True)

        cop = self.game_state#.copy()
        t1 = time.time()
        if self.player_simulation_algorithms[index] == "minimax-alpha-beta-pruning":
            action = self.minimax_agent(cop, cop.player1)
        elif self.player_simulation_algorithms[index] == "path-search":
            action = self.pathsearch_agent(cop, cop.player1)
        # elif self.player_simulation_algorithms[index] == "monte-carlo-tree-search":
        #     start = SearchNode(state=self.game_state, player1_maximizer=maximizer)
        #     selected_node = start.best_action()
        #     action = selected_node.parent_actiond
        elif self.player_simulation_algorithms[index] == "randomBot":
            action = self.randombot_agent(cop)
        elif self.player_simulation_algorithms[index] == "impatientBot":
            action = self.impatientbot_agent(cop)
        else:
            print('No bot configured')
            exit(1)

        if action is not None:
            self.game_state.execute_action(action)
            t2 = time.time()
            self.execution_times[-1][index].append(round(t2 - t1, 4))
            loggable_action = tuple(x if not isinstance(x, bool) else int(x) for x in action)
            self.hist_per_round[-1].append(loggable_action)

            if self.verbose:
                if len(action) == 2:
                    self.print_colored_output("\nPlayer {} ({}) has moved his piece to {}.".format(player_number, self.player_simulation_algorithms[index], action), Color.CYAN, True, '')
                else:
                    orientation = "HORIZONTAL" if action[2] else "VERTICAL"
                    loc = (chr(ord('a') + action[0]), chr(ord('a') + action[1]))
                    self.print_colored_output("Player {} ({}) has placed a {} wall at {}.".format(player_number, self.player_simulation_algorithms[index], orientation, str(loc).replace("'", "")), Color.CYAN, True, '')

                self.print_colored_output("     This took " + str(round(t2 - t1, 4)) + " seconds.", Color.CYAN, _end = '')
            return True
        else:
            self.print_colored_output("Player {} ({}) has no moves left.".format(player_number, self.player_simulation_algorithms[index]), Color.CYAN)
            return False

    def play(self):
        '''
        plays a number of rounds
        '''
        print()
        self.print_game_stats()
        print("\n")
        self.print_board()
        print()
        while self.rounds:
            self.game_state.get_available_wall_placements()
            start_time = time.time()
            if self.game_state.is_end_state():
                execution_times = [sum(i) / max(1,len(i)) for i in self.execution_times[-1]]
                num_moves = len(self.hist_per_round[-1])


                if self.verbose:
                    print("Execution averages this round: ", execution_times)
                    print("Number of moves this round: ", num_moves//2)

                winner_ind = self.game_state.get_winner()
                self.wins[winner_ind] += 1
                self.rounds -= 1
                self.game_state.reinitialize()


                winner = '  P1' if winner_ind == 0 else 'P2'
                self.print_colored_output("The winner is " + winner + ".", Color.CYAN)

                if(self.rounds > 0):
                    self.hist_per_round.append([(SIZE, WALLS)])
                    self.execution_times.append([[],[]])
                continue

            res = self.player_simulation()
            if res == None:
                print("Bot has returned something unholy")
                exit(1)

            if self.verbose:
                print()
                self.print_game_stats()
                print("\n")
                self.print_board()
                print()
            while (time.time() - start_time < self.sim_delay):
                continue

            self.game_state.player1 = not self.game_state.player1

    def print_game_stats(self):
        '''
        prints the game stats like walls left
        '''
        if not self.verbose:
            return

        g = self.game_state

        print(PLAYER1COLOR + "{0:<15}".format("Player 1 walls") + Color.WHITE +
              "|" + PLAYER2COLOR + "{0:<15}".format(
            "Player 2 walls") + Color.RESET,
              end="|\n")
        print("{0:-<15}|{1:-<15}".format("", ""), end="|\n")
        print("{0:<15}|{1:<15}|".format(g.player1_walls_num, g.player2_walls_num))


    def print_board(self):
        '''
        prints the game board in console text readable format'''
        if not self.verbose:
            return

        g = self.game_state

        # print(g.wallboard)

        for i in range(g.size):
            if i == 0:
                print("      {0:<2} ".format(i),
                      end=DEFAULT_WALL_COLOR + chr(ord('a') + i).lower() + Color.RESET)
            elif i == g.size - 1:
                print("  {0:<3}".format(i), end=" ")
            else:
                print("  {0:<2} ".format(i),
                      end=DEFAULT_WALL_COLOR + chr(ord('a') + i).lower() + Color.RESET)
        print()
        print()

        for i in range(g.rows + g.size):
            if i % 2 == 0:
                print("{0:>2}  ".format(i//2), end="")
            else:
                print(DEFAULT_WALL_COLOR + "{0:>2}  ".format(chr(ord('a') + i//2).lower()) + Color.RESET, end="")

            for j in range(g.cols + g.size):

                # (i%2 , j%2):
                # (0,0) means a cell
                # (0,1) means a possible ver border wall
                # (1,0) means a possible hor border wall
                # (1,1) means a intersection point of borders (corner of cell)

                if i % 2 == 0:
                    x = i//2
                    y = j//2
                    if j%2 == 0:
                        if np.array_equal(g.player1_pos, [x,y]):
                            print(PLAYER1COLOR + " {0:2} ".format("P1") + Color.RESET, end="")
                        elif np.array_equal(g.player2_pos, [x,y]):
                            print(PLAYER2COLOR + " {0:2} ".format("P2") + Color.RESET, end="")
                        else:
                            print("{0:4}".format(""), end="")
                    else:
                        if g.verwalls[min(g.rows-1, x), y]:
                            print(self.get_wall_color(g, min(g.rows-1, x), y) + " \u2503" + Color.RESET, end="")
                        elif g.verwalls[max(0,x-1), y]:
                            print(self.get_wall_color(g, max(0,x-1), y) + " \u2503" + Color.RESET, end="")
                        else:
                            print(" |", end="")
                else:
                    if j%2 == 0:
                        x = i//2
                        y = j//2
                        if g.horwalls[x, min(g.cols-1,y)]:
                            line = ""
                            for k in range(5):
                                line += "\u2501"
                            print(self.get_wall_color(g, x, min(g.cols-1,y)) + line + Color.RESET, end="")

                        elif g.horwalls[x, max(0,y-1)]:
                            line = ""
                            for k in range(5):
                                line += "\u2501"
                            print(self.get_wall_color(g, x, max(0,y-1)) + line + Color.RESET, end="")
                        else:
                            line = ""
                            for k in range(5):
                                line += "\u23AF"
                            print(line, end="")
                    else:
                        if not g.verwalls[i//2, j//2]  and not g.horwalls[i//2, j//2]:
                            print("o", end="")
                        else:
                            print(self.get_wall_color(g, i//2, j//2) + "o" + Color.RESET, end="")
            print()


    @staticmethod
    def get_wall_color(g : GameState, i : int, j : int):
        return PLAYER1COLOR if g.wall_colors_board[i,j] == WallColor.PLAYER1 else PLAYER2COLOR

    @staticmethod
    def print_colored_output(text, color, wipe=False, _end = '\n'):
        if wipe:
            print('\r' + color + text + Color.RESET, end = _end, flush=True)
        else:
            print(color + text + Color.RESET, end = _end, flush=True)
