import numpy as np


from copy import *
from time import *
from console.search.astar import astar_search
import threading


class WallColor:
    FREE_WALL = 0
    PLAYER1 = 1
    PLAYER2 = 2


class GameState:
    def __init__(self, size = 9, walls = 10):
        self.player1 = True
        self.size = size
        self.rows = self.size-1
        self.cols = self.size-1
        self.walls = walls
        self.player1_walls_num = walls
        self.player2_walls_num = walls
        self.saved_wall_placements = []

        self.check_wall_blocks_exit_on_gen = True

        self.player1_pos: tuple = tuple([self.size-1, self.size//2])
        self.player2_pos: tuple = tuple([0, self.size//2])

        self.verwalls = np.zeros((self.rows, self.cols), dtype=bool)
        self.horwalls = np.zeros((self.rows, self.cols), dtype=bool)
        self.wall_colors_board = np.zeros((self.rows, self.cols), dtype=int)


    def reinitialize(self):
        '''
        Sets up the game state to starting
        '''
        self.player1_pos = tuple([self.size-1, self.size//2])
        self.player2_pos = tuple([0, self.size//2])

        self.verwalls = np.zeros((self.rows, self.cols), dtype=bool)
        self.horwalls = np.zeros((self.rows, self.cols), dtype=bool)
        self.wall_colors_board = np.zeros((self.rows, self.cols), dtype=int)

        self.player1 = True
        self.size = self.size
        self.rows = self.size-1
        self.cols = self.size-1
        self.player1_walls_num = self.walls
        self.player2_walls_num = self.walls
        self.saved_wall_placements = []
        self.saved_wall_placements = self.get_available_wall_placements()


    def copy(self):
        game_state = GameState()
        game_state.player1_pos = copy(self.player1_pos)
        game_state.player2_pos = copy(self.player2_pos)
        game_state.verwalls = copy(self.verwalls)
        game_state.horwalls = copy(self.horwalls)
        game_state.wall_colors_board = copy(self.wall_colors_board)
        game_state.saved_wall_placements = deepcopy(self.saved_wall_placements)
        return game_state


    def is_goal_state(self):
        if self.player1:
            return self.player1_pos[0] == 0
        else:
            return self.player2_pos[0] == self.rows

    def get_current_player_pos(self):
        if self.player1:
            return copy(self.player1_pos)
        else:
            return copy(self.player2_pos)

    def is_valid_move(self, pos : tuple, new_pos : tuple):
        '''
        Checks if a given direct move is possible (in bounds, no walls blocking)
        '''
        # Out of bounds
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return False

        # no wall between player and new_pos
        # move sideways (check for vertical wall)
        if pos[0] == new_pos[0]:
            y = min(pos[1], new_pos[1], self.cols-1)
            if self.verwalls[min(self.rows-1, pos[0]), y] or self.verwalls[max(0, pos[0]-1), y]:
                return False

        # move vertical (check for horizontal wall)
        elif pos[1] == new_pos[1]:
            x = min(pos[0], new_pos[0], self.rows-1)
            if self.horwalls[x, min(self.cols-1, pos[1])] or self.horwalls[x, max(0, pos[1]-1)]:
                    return False

        return True


    def get_available_moves(self):
        '''
        Gets a list of all player moves for current player, including jumps
        '''
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
            if self.is_valid_move(pos, (x,y)) and not np.array_equal((x,y), otherpos):
                available_moves.append((x,y))

        # hor jump
        x_diff = abs(otherpos[0] - pos[0])
        y_diff = abs(otherpos[1] - pos[1])

        if (x_diff == 0 and y_diff == 1) or (x_diff == 1 and y_diff == 0):
            x_dir = otherpos[0]-pos[0]
            y_dir = otherpos[1]-pos[1]
            x = otherpos[0] + x_dir
            y = otherpos[1] + y_dir


            if self.is_valid_move(pos, otherpos):
                # vert or hor jump
                if self.is_valid_move(otherpos, (x,y)):
                    available_moves.append((x,y))

                # diag jump
                elif x_dir == 0:
                    for x,y in [(otherpos[0]+i, otherpos[1]) for i in [1,-1]]:
                        if self.is_valid_move(otherpos, (x,y)):
                            available_moves.append((x,y))
                elif y_dir == 0:
                    for x,y in [(otherpos[0], otherpos[1]+i) for i in [1,-1]]:
                        if self.is_valid_move(otherpos, (x,y)):
                            available_moves.append((x,y))

        return available_moves

    def is_wall_placement_valid(self, pos : tuple, isHorizontal : bool):
        '''
        Gets a list of all available wall placements as (i, j, isHor)

        Uses the boolean 'self.check_wall_blocks_exit_on_gen' to do astar recursive, be careful if set
        '''

        # out of bounds
        if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
            return False

        # already occupied
        if self.horwalls[pos[0], pos[1]] or self.verwalls[pos[0], pos[1]]:
            return False


        if isHorizontal:
            if pos[1] > 0 and self.horwalls[pos[0], pos[1]-1]:
                return False
            if pos[1] < self.cols-1 and self.horwalls[pos[0], pos[1]+1]:
                return False
        else:
            if pos[0] > 0 and self.verwalls[pos[0]-1, pos[1]]:
                return False
            if pos[0] < self.rows-1 and self.verwalls[pos[0]+1, pos[1]]:
                return False

        if self.check_wall_blocks_exit_on_gen:
            if self.is_wall_blocking_exit(pos, isHorizontal):
                return False

        return True

    def temp_set_wall(self, pos : tuple, isHorizontal : bool):
        '''
        Temp sets a wall, but not player colors
        '''
        if(isHorizontal):
            self.horwalls[pos[0], pos[1]] = True
        else:
            self.verwalls[pos[0], pos[1]] = True

    def unset_wall(self, pos : tuple):
        '''
        Clears a wall from pos, clears both ver and hor (only one should have been set in pos)
        '''
        self.horwalls[pos[0], pos[1]] = False
        self.verwalls[pos[0], pos[1]] = False

    def is_wall_blocking_exit(self, pos : tuple, isHorizontal : bool):
        '''
        Checks if a wall would block the exit
        '''
        self.temp_set_wall(pos, isHorizontal)

        # check paths
        p1, p2 = astar_search(self)
        exit_blocked = (p1 == np.inf or p2 == np.inf)

        self.unset_wall(pos)
        return exit_blocked

    def update_available_wall_placements(self, pos):
        '''
        Goes through all saved available wall placements, removes them if they do not work
        '''
        wall_placements = []
        for i, j, isHorizontal in self.saved_wall_placements:
            pos = (i,j)
            if self.is_wall_placement_valid(pos, isHorizontal):
                wall_placements.append((*pos, isHorizontal))
        self.saved_wall_placements = wall_placements


    def get_available_wall_placements(self):
        '''
        Gets a list of available wall placements
        '''
        wall_placements = []
        if self.player1:
            if self.player1_walls_num <= 0:
                return wall_placements
        elif self.player2_walls_num <= 0:
                return wall_placements

        # valid wall placement already known already found
        if len(self.saved_wall_placements) > 0:
            return self.saved_wall_placements

        # calculate wall placements
        for i in range(self.rows):
            for j in range(self.cols):
                pos = (i,j)

                # check horizontals
                if self.is_wall_placement_valid(pos, isHorizontal = True):
                    wall_placements.append((*pos, True))

                # check verticals
                if self.is_wall_placement_valid(pos, isHorizontal = False):
                    wall_placements.append((*pos, False))

        # save for reuse
        self.saved_wall_placements = wall_placements

        return wall_placements


    def execute_action(self, action):
        '''
        executes an action, does not change which player turn
        '''
        if len(action) == 2:
            self.move_piece(action)
        else:
            self.place_wall(action)


    def place_wall(self, inp : tuple, check_if_valid = True, update_wall_placements = True):
        '''
        Places a wall and updates player colors, and updates player walls

        More permanent than temp_set_wall
        '''
        x, y, isHorizontal = inp
        pos = x,y

        if check_if_valid:
            if not self.is_wall_placement_valid(pos, isHorizontal):
                print('CRAP', inp)
                exit(2)

        if self.player1:
            self.player1_walls_num -= 1
        else:
            self.player2_walls_num -= 1

        self.temp_set_wall(pos, isHorizontal)
        self.wall_colors_board[pos[0], pos[1]] = WallColor.PLAYER1 if self.player1 else WallColor.PLAYER2

        # update wall placements
        if update_wall_placements:
            self.update_available_wall_placements(pos)
        else:
            self.saved_wall_placements = []



    def move_piece(self, new_pos : tuple, compute_new_wall_placements = True):
        '''
        Moves the current player to a new_pos
        '''

        if self.player1:
            pos = self.player1_pos
            otherpos = self.player2_pos
        else:
            pos = self.player2_pos
            otherpos = self.player1_pos

        if not self.is_valid_move(pos, new_pos) or new_pos == otherpos:
            print('CRAP', new_pos)
            exit(2)

        if self.player1:
            self.player1_pos = copy(new_pos)
        else:
            self.player2_pos = copy(new_pos)

        if compute_new_wall_placements:
            wall_placements = self.saved_wall_placements

            for i in [-1, 0]:
                for j in [-1,0]:
                    w1 = ((new_pos[0] + i, new_pos[1] + j), True)
                    w2 = ((new_pos[0] + i, new_pos[1] + j), False)
                    w3 = ((pos[0] + i, pos[1] + j), True)
                    w4 = ((pos[0] + i, pos[1] + j), False)
                    for w in [w1, w2, w3, w4]:
                        wall = (*w[0],w[1])
                        isValid = self.is_wall_placement_valid(*w)
                        if wall in wall_placements:
                            if not isValid:
                                wall_placements.remove(wall)
                        elif isValid:
                            wall_placements.append(wall)

            self.saved_wall_placements = wall_placements

    def is_end_state(self):
        return self.player1_pos[0] == 0 or self.player2_pos[0] == self.size-1

    def get_winner(self):

        if self.player1_pos[0] == 0:
            return 0
        elif self.player2_pos[0] == self.rows:
            return 1
        else:
            return -1
