from console.states.game_state import GameState, WallPieceStatus
from time import time, sleep
from console.util.color import Color
from console.algorithms.minimax import minimax
from console.algorithms.minimax_alpha_beta_pruning import minimax_alpha_beta_pruning
from console.algorithms.expectimax import expectimax
from console.algorithms.randombot import randombot_action
from console.algorithms.impatientbot import impatientbot_action
from console.algorithms.monte_carlo_tree_search import SearchNode
import math
import numpy as np

Wallcolor = Color.PINK

SIZE = 15
WALLS = 30

MoveKeyValues = "".join([str(i) for i in range(SIZE)])
WallKeyValues = "".join([chr(ord('a') + i).lower() for i in range(SIZE-1)])

class Game:
    def __init__(self, user_sim = False, rounds = 1, verbose = True, sim_delay = 0.5):

        self.player_simulation_algorithms = ["randomBot", "randomBot"]
        self.game_state = GameState(SIZE, WALLS)
        self.verbose = verbose
        self.is_user_sim = user_sim
        self.algorithms = ["randomBot", "impatientBot"]
        self.execution_times = []
        self.sim_delay = sim_delay
        self.rounds = rounds

        if user_sim:
            self.initialize_sim()
        else:
            self.sim_delay = 0.0
            self.quick_run("randomBot", "impatientBot")
            

    def print_commands(self):
        print(
            "1. You can move your piece by entering" + Color.CYAN + " mx,y " + Color.RESET + "where x is the row number and y column number")
        print(
            "2. You can place a wall by entering" + Color.CYAN + " wa,bd " + Color.RESET + "where a is the row letter and b column letter, and where d represents the wall direction and")
        print("it can be either [V,H]. They represent the vertical or horizontal orientations.")
        print("3. When it's your turn, you can also press " + Color.CYAN + " x " + Color.RESET + " to exit the game.")
        print("4. When it's your turn, you can also type " + Color.CYAN + " help " + Color.RESET + " to print this guide again.")

    def quick_run(self, bot1, bot2):
        Game.print_colored_output("### Quick Running rounds ###", Color.CYAN)
        self.is_user_sim = False
        self.player_simulation_algorithms[0] = bot1
        self.player_simulation_algorithms[1] = bot2
        Game.print_colored_output("Chosen algorithm for player 1 is {0:30}".format(self.player_simulation_algorithms[0].upper()), Color.CYAN)
        Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(self.player_simulation_algorithms[1].upper()), Color.CYAN)


    def initialize_sim(self):
        Game.print_colored_output("### WELCOME TO QUORIDOR ###", Color.CYAN)
        print("\n")
        print("First the commands [they are case insensitive]: ")
        self.print_commands()
        print("{0:-<100}".format(""))

        a = 'N'#input("\nDo you want to play against a computer?[Y/n]: ")
        if a == "Y" or a == "y":
            self.is_user_sim = True

            print("Choose the second player algorithm: ")
            print("1. randomBot")
            print("2. impatientBot")
            while True:
                x = input("Choose: ")
                if not x.isdigit() and x != "x" and x != "X":
                    Game.print_colored_output("Illegal input!", Color.RED)
                elif x == "x" or x == "X":
                    exit(0)
                else:
                    if 0 <= int(x) - 1 < len(self.algorithms):
                        self.player_simulation_algorithms[1] = self.algorithms[int(x) - 1]
                        Game.print_colored_output("Chosen algorithm for player 2 is {0:30}".format(
                            self.player_simulation_algorithms[1].upper()), Color.CYAN)
                        break
                    else:
                        Game.print_colored_output("Illegal input!", Color.RED)
        else:
            self.is_user_sim = False
            # print("Choose the players algorithms[first_player, second_player]")
            # print("1. minimax")
            # print("2. minimax with alpha beta pruning")
            # print("3. expectimax")
            # print("4. monte carlo tree search")
            while True:
                # x = input("Choose: ")
                x = "1,2"
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

    def choose_action(self, action):
        if len(action) == 2:
            self.game_state.move_piece(action)
        else:
            self.game_state.place_wall(action)
        return action

    # def minimax_agent(self, player1_minimax, is_alpha_beta):
    #     d = {}
    #     for child in self.game_state.get_all_child_states(player1_minimax):
    #         if not is_alpha_beta:
    #             value = minimax(child[0], 3, maximizing_player=False, player1_minimax=player1_minimax)
    #         else:
    #             value = minimax_alpha_beta_pruning(child[0], 3, -math.inf, math.inf, maximizing_player=False,
    #                                                player1_minimax=player1_minimax)
    #         d[value] = child
    #     return self.choose_action(d)

    # def expectimax_agent(self, player1_maximizer):
    #     d = {}
    #     for child in self.game_state.get_all_child_states(player1_maximizer):
    #         value = expectimax(child[0], 2, False, player1_maximizer)
    #         d[value] = child
    #     return self.choose_action(d)

    def randombot_agent(self):
        action = randombot_action(self.game_state)
        return self.choose_action(action)

    def impatientbot_agent(self):
        action = impatientbot_action(self.game_state)
        return self.choose_action(action)

    def player1_user(self):
        while True:
            value = input("Enter move: ")
            if value == "x" or value == "X":
                exit(0)
            elif value.lower() == "help":
                print()
                self.print_commands()
                print()
            else:
                if value.upper().startswith("M"):
                    x_string, y_string = value[1:].split(",")
                    if x_string.upper() not in MoveKeyValues or y_string.upper() not in MoveKeyValues:
                        Game.print_colored_output("Illegal move!", Color.RED)
                    else:
                        x_int = int(x_string)
                        y_int = int(y_string)
                        available_moves = self.game_state.get_available_moves(False)
                        move = (x_int, y_int)
                        if move not in available_moves:
                            Game.print_colored_output("Illegal move!", Color.RED)
                        else:
                            self.game_state.move_piece(move)
                            break

                elif value.upper().startswith("W"):
                    # place a wall
                    x_string, y_string = value[1:len(value) - 1].split(",")
                    if x_string.upper() not in WallKeyValues or y_string.upper() not in WallKeyValues:
                        Game.print_colored_output("Illegal wall placement!", Color.RED)
                    else:
                        orientation_string = value[-1]
                        if orientation_string.upper() in ["V", "H"]:

                            if orientation_string.upper() == "V":
                                orientation = WallPieceStatus.VERTICAL
                            elif orientation_string.upper() == "H":
                                orientation = WallPieceStatus.HORIZONTAL

                            x_int = ord(x_string) - ord('a')
                            y_int = ord(y_string) - ord('a')
                            is_placement_valid, coords = self.game_state.check_wall_placement((x_int, y_int),
                                                                                              orientation)
                            if not is_placement_valid:
                                Game.print_colored_output("Illegal wall placement!", Color.RED)
                            else:
                                self.game_state.place_wall(coords)
                                break

                else:
                    Game.print_colored_output("Illegal command!", Color.RED)

    def player_simulation(self, player_number):
        if player_number == 1:
            index = 0
            maximizer = True
        else:
            index = 1
            maximizer = False
        t1 = time()
        # print("Player {0:1} is thinking...\n".format(player_number))
        action = (0, 0)
        # if self.player_simulation_algorithms[index] == "minimax":
        #     action = self.minimax_agent(maximizer, is_alpha_beta=False)
        # elif self.player_simulation_algorithms[index] == "minimax-alpha-beta-pruning":
        #     action = self.minimax_agent(maximizer, is_alpha_beta=True)
        # elif self.player_simulation_algorithms[index] == "expectimax":
        #     action = self.expectimax_agent(maximizer)
        # elif self.player_simulation_algorithms[index] == "monte-carlo-tree-search":
        #     start = SearchNode(state=self.game_state, player1_maximizer=maximizer)
        #     selected_node = start.best_action()
        #     action = selected_node.parent_action
        #     self.game_state.execute_action(action, False)
        if self.player_simulation_algorithms[index] == "randomBot":
            action = self.randombot_agent()
        elif self.player_simulation_algorithms[index] == "impatientBot":
            action = self.impatientbot_agent()
        else:
            print('No bot configured')

        if action is not None:
            if len(action) == 2:
                self.print_colored_output("Player {} has moved his piece to {}.".format(player_number, action), Color.CYAN)
            else:
                orientation = "HORIZONTAL" if action[2] == WallPieceStatus.HORIZONTAL else "VERTICAL"
                loc = (chr(ord('a') + action[0]), chr(ord('a') + action[1]))
                self.print_colored_output("Player {} has placed a {} wall at {}.".format(player_number, orientation, loc), Color.CYAN)
            t2 = time()
            self.execution_times.append(t2 - t1)
            self.print_colored_output("It took him " + str(round(t2 - t1, 2)) + " seconds.", Color.CYAN)
            return True
        else:
            self.print_colored_output("Player {} has no moves left.".format(player_number), Color.CYAN)
            return False

    def check_end_state(self):
        if self.game_state.is_end_state():
            winner = self.game_state.get_winner()
            if self.is_user_sim:
                if winner == "P1":
                    self.print_colored_output("You won!", Color.GREEN)
                else:
                    self.print_colored_output("You lost!", Color.RED)
            else:
                self.print_colored_output("The winner is " + winner + ".", Color.CYAN)
            return True
        else:
            return False

    def play(self):
        while self.rounds:
            print()
            self.print_game_stats()
            print("\n")
            self.print_board()
            print()

            if self.check_end_state():
                print("Execution average: ", sum(self.execution_times) / len(self.execution_times))
                self.rounds -= 1
                self.game_state.reinitialize()
                continue

            if self.game_state.player1:
                if self.is_user_sim:
                    self.player1_user()
                else:
                    res = self.player_simulation(1)
                    sleep(self.sim_delay)
                    if not res:
                        break
            else:
                res = self.player_simulation(2)
                sleep(self.sim_delay)
                if not res:
                        break

            self.game_state.player1 = not self.game_state.player1

    def print_game_stats(self):
        if not self.verbose:
            return
        
        g = self.game_state
        
        print(Color.GREEN + "{0:<15}".format("Player 1 walls") + Color.WHITE +
              "|" + Color.RED + "{0:<15}".format(
            "Player 2 walls") + Color.RESET,
              end="|\n")
        print("{0:-<15}|{1:-<15}".format("", ""), end="|\n")
        print("{0:<15}|{1:<15}|".format(g.player1_walls_num, g.player2_walls_num))


    def print_board(self):
        if not self.verbose:
            return
        
        g = self.game_state

        # print(g.wallboard)

        for i in range(g.size):
            if i == 0:
                print("      {0:<2} ".format(i),
                      end=Wallcolor + chr(ord('a') + i).lower() + Color.RESET)
            elif i == g.size - 1:
                print("  {0:<3}".format(i), end=" ")
            else:
                print("  {0:<2} ".format(i),
                      end=Wallcolor + chr(ord('a') + i).lower() + Color.RESET)
        print()
        print()

        for i in range(g.rows + g.size):
            if i % 2 == 0:
                print("{0:>2}  ".format(i//2), end="")
            else:
                print(Wallcolor + "{0:>2}  ".format(chr(ord('a') + i//2).lower()) + Color.RESET, end="")

            for j in range(g.cols + g.size):

                # (i%2 , j%2):
                # (0,0) means a cell
                # (0,1) means a possible ver wall
                # (1,0) means a possible hor wall
                # (1,1) means a intersection of walls
                
                if i % 2 == 0:
                    x = i//2
                    y = j//2
                    if j%2 == 0:
                        if np.array_equal(g.player1_pos, [x,y]):
                            print(Color.GREEN + " {0:2} ".format("P1") + Color.RESET, end="")
                        elif np.array_equal(g.player2_pos, [x,y]):
                            print(Color.RED + " {0:2} ".format("P2") + Color.RESET, end="")
                        else:
                            print("{0:4}".format(""), end="")
                    else:
                        if g.wallboard[min(g.rows-1, x), y] == WallPieceStatus.VERTICAL or g.wallboard[max(0,x-1), y] == WallPieceStatus.VERTICAL:
                            print(Wallcolor + " \u2503" + Color.RESET, end="")
                        else:
                            print(" |", end="")
                else:
                    if j%2 == 0:
                        x = i//2
                        y = j//2
                        if g.wallboard[x,min(g.cols-1,y)] == WallPieceStatus.HORIZONTAL or g.wallboard[x, max(0,y-1)] == WallPieceStatus.HORIZONTAL:
                            line = ""
                            for k in range(5):
                                line += "\u2501"
                            print(Wallcolor + line + Color.RESET, end="")
                        else:
                            line = ""
                            for k in range(5):
                                line += "\u23AF"
                            print(line, end="")
                    else:
                        if g.wallboard[i//2, j//2] == WallPieceStatus.FREE_WALL:
                            print("o", end="")
                        else:
                            print(Wallcolor + "o" + Color.RESET, end="")        
            print()

    @staticmethod
    def print_colored_output(text, color):
        print(color + text + Color.RESET)
