from console.util.wall_direction import WallDirection
import numpy as np
from copy import copy
from console.search.astar import astar
from console.search.bfs import bfs_check_if_paths_exist
from console.util.color import Color
import threading


class WallPieceStatus:
    FREE_WALL = 0
    HORIZONTAL = 1
    VERTICAL = 2


Wallcolor = Color.PINK


class GameState:
    def __init__(self, verbose=False, size = 9, walls = 10):
        self.verbose = verbose
        self.player1 = True
        self.size = size
        self.rows = self.size-1
        self.cols = self.size-1
        self.walls = walls
        self.player1_walls_num = walls
        self.player2_walls_num = walls
        self.lock = threading.Lock()
        self.walls_can_be_placed = []

        self.player1_pos = np.array([self.size-1, self.size//2])
        self.player2_pos = np.array([0, self.size//2])

        self.wallboard = np.zeros((self.rows, self.cols), dtype=int)   
        self.set_up_board()


    def reinitialize(self):
        self.player1_pos = np.array([self.size-1, self.size//2])
        self.player2_pos = np.array([0, self.size//2]) 
        self.set_up_board()

        self.player1 = True
        self.size = self.size
        self.rows = self.size-1
        self.cols = self.size-1
        self.player1_walls_num = self.walls
        self.player2_walls_num = self.walls
        self.lock = threading.Lock()
        self.walls_can_be_placed = []

    def set_up_board(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.wallboard[i,j] = WallPieceStatus.FREE_WALL

    def copy(self):
        game_state = copy(self)
        game_state.player1_pos = copy(self.player1_pos)
        game_state.player2_pos = copy(self.player2_pos)
        game_state.wallboard = copy(self.wallboard)
        return game_state

    def print_game_stats(self):
        if not self.verbose:
            return
        
        print(Color.GREEN + "{0:<15}".format("Player 1 walls") + Color.WHITE +
              "|" + Color.RED + "{0:<15}".format(
            "Player 2 walls") + Color.RESET,
              end="|\n")
        print("{0:-<15}|{1:-<15}".format("", ""), end="|\n")
        print("{0:<15}|{1:<15}|".format(self.player1_walls_num, self.player2_walls_num))

    def print_board(self):
        if not self.verbose:
            return

        # print(self.wallboard)

        for i in range(self.size):
            if i == 0:
                print("      {0:<2} ".format(i),
                      end=Wallcolor + chr(ord('a') + i).lower() + Color.RESET)
            elif i == self.size - 1:
                print("  {0:<3}".format(i), end=" ")
            else:
                print("  {0:<2} ".format(i),
                      end=Wallcolor + chr(ord('a') + i).lower() + Color.RESET)
        print()
        print()

        for i in range(self.rows + self.size):
            if i % 2 == 0:
                print("{0:>2}  ".format(i//2), end="")
            else:
                print(Wallcolor + "{0:>2}  ".format(chr(ord('a') + i//2).lower()) + Color.RESET, end="")

            for j in range(self.cols + self.size):

                # (i%2 , j%2):
                # (0,0) means a cell
                # (0,1) means a possible ver wall
                # (1,0) means a possible hor wall
                # (1,1) means a intersection of walls
                
                if i % 2 == 0:
                    x = i//2
                    y = j//2
                    if j%2 == 0:
                        if np.array_equal(self.player1_pos, [x,y]):
                            print(Color.GREEN + " {0:2} ".format("P1") + Color.RESET, end="")
                        elif np.array_equal(self.player2_pos, [x,y]):
                            print(Color.RED + " {0:2} ".format("P2") + Color.RESET, end="")
                        else:
                            print("{0:4}".format(""), end="")
                    else:
                        if self.wallboard[min(self.rows-1, x), y] == WallPieceStatus.VERTICAL or self.wallboard[max(0,x-1), y] == WallPieceStatus.VERTICAL:
                            print(Wallcolor + " \u2503" + Color.RESET, end="")
                        else:
                            print(" |", end="")
                else:
                    if j%2 == 0:
                        x = i//2
                        y = j//2
                        if self.wallboard[x,min(self.cols-1,y)] == WallPieceStatus.HORIZONTAL or self.wallboard[x, max(0,y-1)] == WallPieceStatus.HORIZONTAL:
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
                        if self.wallboard[i//2, j//2] == WallPieceStatus.FREE_WALL:
                            print("o", end="")
                        else:
                            print(Wallcolor + "o" + Color.RESET, end="")        
            print()

    def is_piece_occupied(self, i, j):
        return np.array_equal(self.player1_pos, [i,j]) or np.array_equal(self.player2_pos, [i,j])

    def is_not_piece_occupied(self, i, j):
        return not self.is_piece_occupied(i, j)

    def is_wall_blocking_path(self, pos, new_pos):
        # move sideways (check for vertical wall)
        if pos[0] == new_pos[0]:
            y = min(pos[1], new_pos[1], self.cols-1)
            if self.wallboard[min(self.rows-1, pos[0]), y] == WallPieceStatus.VERTICAL \
            or self.wallboard[max(0, pos[0]-1), y] == WallPieceStatus.VERTICAL: 
                return True
        # move vertical (check for horizontal wall)
        elif pos[1] == new_pos[1]:
            x = min(pos[0], new_pos[0], self.rows-1)
            if self.wallboard[x, min(self.cols-1, pos[1])] == WallPieceStatus.HORIZONTAL \
            or self.wallboard[x, max(0, pos[1]-1)] == WallPieceStatus.HORIZONTAL: 
                return True
        return False

    def is_not_wall_blocking_path(self, pos, new_pos):
        return not self.is_wall_blocking_path(pos, new_pos)

    def is_goal_state(self):
        if self.player1:
            return self.player1_pos[0] == 0
        else:
            return self.player2_pos[0] == self.rows

    def distance_to_goal(self):
        if self.player1:
            return self.player1_pos[0]
        else:
            return self.rows - self.player2_pos[0]

    def get_child_states_with_moves(self):
        print('run')
        available_moves = self.get_available_moves(False)
        children = []
        for move in available_moves:
            child = self.copy()
            child.move_piece(move)
            cost = 1000
            if self.is_jump(move):
                cost = 500
            elif self.is_diagonal(move):
                cost = 500
            if child.player1:
                pos = child.player1_pos
            else:
                pos = child.player2_pos
            simplified_child_state = ((pos[0], pos[1]), (move[0], move[1]), cost)

            children.append((child, simplified_child_state))
        return children

    def get_all_child_states(self, player1_maximizer, include_state=True):

        children = []
        available_moves = self.get_available_moves(include_state)
        for move in available_moves:
            children.append(move)

        available_wall_placements = []
        if not self.player1 and not player1_maximizer:
            available_wall_placements = self.get_available_wall_placements(include_state)

        if self.player1 and player1_maximizer:
            available_wall_placements = self.get_available_wall_placements(include_state)

        for wall_placement in available_wall_placements:
            children.append(wall_placement)

        return children
    
    def check_valid_move(self, pos, new_pos):
        # Out of bounds
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return False
    
        # no wall between player and new_pos
        if self.is_wall_blocking_path(pos, new_pos):
            return False

        return True

   
    def get_available_moves(self, include_state=False):
        available_moves = []
                    

        dirs = [(1,0), (0,1), (-1,0), (0,-1)]
        if self.player1:
            pos = self.player1_pos
            otherpos = self.player2_pos
        else:
            pos = self.player2_pos
            otherpos = self.player1_pos
        
        #standard moves
        for x,y in [(pos[0]+i, pos[1] + j) for (i,j) in dirs]:
            if self.check_valid_move(pos, (x,y)) and not np.array_equal((x,y), otherpos):
                copy_state = self.copy()
                if not include_state:
                    available_moves.append((x,y))
                else:
                    available_moves.append((copy_state, (x,y)))
        
        # hor jump
        x_diff = abs(otherpos[0] - pos[0])
        y_diff = abs(otherpos[1] - pos[1])
        
        if (x_diff == 0 and y_diff == 1) or (x_diff == 1 and y_diff == 0):
            x_dir = otherpos[0]-pos[0]
            y_dir = otherpos[1]-pos[1]
            x = otherpos[0] + x_dir
            y = otherpos[1] + y_dir

            
            if self.check_valid_move(pos, otherpos):
                # vert or hor jump
                if self.check_valid_move(otherpos, (x,y)):
                    copy_state = self.copy()
                    if not include_state:
                        available_moves.append((x,y))
                    else:
                        available_moves.append((copy_state, (x,y)))  

                # diag jump
                if x_dir == 0:
                    for x,y in [(otherpos[0]+i, otherpos[1]) for i in [1,-1]]:
                        if self.check_valid_move(otherpos, (x,y)):
                            copy_state = self.copy()
                            if not include_state:
                                available_moves.append((x,y))
                            else:
                                available_moves.append((copy_state, (x,y)))
                if y_dir == 0:
                    for x,y in [(otherpos[0], otherpos[1]+i) for i in [1,-1]]:
                        if self.check_valid_move(otherpos, (x,y)):
                            copy_state = self.copy()
                            if not include_state:
                                available_moves.append((x,y))
                            else:
                                available_moves.append((copy_state, (x,y)))
        # print(available_moves)
        return available_moves

    def is_wall_placement_valid(self, pos, orientation):

        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return False
        
        if self.wallboard[pos[0], pos[1]] != WallPieceStatus.FREE_WALL:
            return False

        if orientation == WallPieceStatus.HORIZONTAL:
            if pos[1] > 0 and self.wallboard[pos[0], pos[1]-1] == WallPieceStatus.HORIZONTAL:
                return False
            if pos[1] < self.cols-1 and self.wallboard[pos[0], pos[1]+1] == WallPieceStatus.HORIZONTAL:
                return False
        if orientation == WallPieceStatus.VERTICAL:
            if pos[0] > 0 and self.wallboard[pos[0]-1, pos[1]] == WallPieceStatus.VERTICAL:
                return False
            if pos[0] < self.rows-1 and self.wallboard[pos[0]+1, pos[1]] == WallPieceStatus.VERTICAL:
                return False
            
        return True

    def is_wall_blocking_exit(self, pos, orientation):
        cop = self.copy()
        cop.place_wall((*pos, orientation))
        return not bfs_check_if_paths_exist(cop)

    def get_available_wall_placements(self, include_state=False):
        wall_placements = []
        if self.player1:
            if self.player1_walls_num <= 0:
                return wall_placements
        elif self.player2_walls_num <= 0:
                return wall_placements

        # possibly already found
        # if self.walls_can_be_placed:
        #     return self.walls_can_be_placed

        for i in range(self.rows):
            for j in range(self.cols):

                pos = (i,j)
                orientation = WallPieceStatus.HORIZONTAL

                # check horizontals
                if self.is_wall_placement_valid(pos, orientation):
                    copy_state = self.copy()
                    if not copy_state.is_wall_blocking_exit(pos, orientation):
                        if not include_state:
                            wall_placements.append((*pos, orientation))
                        else:
                            wall_placements.append((copy_state, (*pos, orientation)))

                orientation = WallPieceStatus.VERTICAL
                # check verticals
                if self.is_wall_placement_valid(pos, orientation):
                    copy_state = self.copy()
                    if not copy_state.is_wall_blocking_exit(pos, orientation):
                        if not include_state:
                            wall_placements.append((*pos, orientation))
                        else:
                            wall_placements.append((copy_state, (*pos, orientation)))

        # save for reuse
        # self.walls_can_be_placed = wall_placements

        return wall_placements


    def execute_action(self, action, execute_on_copy=True):
        if execute_on_copy:
            state = self.copy()
        else:
            state = self
        if len(action) == 2:
            state.move_piece(action)
        else:
            state.place_wall(action)

        if execute_on_copy:
            state.player1 = not self.player1
        return state
    

    def place_wall(self, inp):
        x, y, orientation = inp
        pos = x,y

        if self.player1:
            self.player1_walls_num -= 1
        else:
            self.player2_walls_num -= 1

        self.wallboard[pos[0], pos[1]] = orientation
        self.walls_can_be_placed = []
            

    def move_piece(self, new_pos):
        if self.player1:
            self.player1_pos = np.array(new_pos)
        else:
            self.player2_pos = np.array(new_pos)

    def is_end_state(self):
        return self.player1_pos[0] == 0 or self.player2_pos[0] == self.size-1

    def game_result(self, player1_maximizer=False):
        if player1_maximizer:
            if self.player1_pos[0] == 0:
                return 1
            else:
                return -1
        else:
            if self.player2_pos[0] == 16:
                return 1
            else:
                return -1

    def get_winner(self):
        if self.player1_pos[0] == 0:
            return "P1"
        else:
            return "P2"
