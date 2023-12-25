from console.util.wall_direction import WallDirection
import numpy as np
from copy import copy
from console.search.astar import astar
from console.search.dfs_check_exit import dfs_check_if_exit_paths_exist
from console.util.color import Color
import threading


class WallPieceStatus:
    FREE_WALL = 0
    HORIZONTAL = 1
    VERTICAL = 2





class GameState:
    def __init__(self, size = 9, walls = 10):
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
        
    def is_jump(self, new_pos):
        if self.player1:
            pos = self.player1_pos
        else:
            pos = self.player2_pos

        x_diff = abs(new_pos[0] - pos[0])
        y_diff = abs(new_pos[1] - pos[1])
        
        return (x_diff, y_diff) in [(2,0), (0,2), (1,1)]

    def is_valid_move(self, pos, new_pos):
        # Out of bounds
        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            return False
    
        # no wall between player and new_pos
        if self.is_wall_blocking_path(pos, new_pos):
            return False

        return True

    def get_all_child_states(self, include_move = False):

        children = []
        available_moves = self.get_available_moves(True)
        for child, move in available_moves:
            if not include_move:
                children.append(child)
            else:
                children.append((child, move))

        available_wall_placements = []
        available_wall_placements = self.get_available_wall_placements(True)

        for child, wall_placement in available_wall_placements:
            if not include_move:
                children.append(child)
            else:
                children.append((child, wall_placement))

        return children
    
   
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
            if self.is_valid_move(pos, (x,y)) and not np.array_equal((x,y), otherpos):
                if not include_state:
                    available_moves.append((x,y))
                else:
                    child = self.copy()
                    child.move_piece((x,y))
                    available_moves.append((child, (x,y)))
        
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
                    if not include_state:
                        available_moves.append((x,y))
                    else:
                        child = self.copy()
                        child.move_piece((x,y))
                        available_moves.append((child, (x,y)))  

                # diag jump
                elif x_dir == 0:
                    for x,y in [(otherpos[0]+i, otherpos[1]) for i in [1,-1]]:
                        if self.is_valid_move(otherpos, (x,y)):
                            if not include_state:
                                available_moves.append((x,y))
                            else:
                                child = self.copy()
                                child.move_piece((x,y))
                                available_moves.append((child, (x,y)))
                elif y_dir == 0:
                    for x,y in [(otherpos[0], otherpos[1]+i) for i in [1,-1]]:
                        if self.is_valid_move(otherpos, (x,y)):
                            if not include_state:
                                available_moves.append((x,y))
                            else:
                                child = self.copy()
                                child.move_piece((x,y))
                                available_moves.append((child, (x,y)))
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
        cop.place_wall((*pos, orientation), False)
        return not dfs_check_if_exit_paths_exist(cop)

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
                    if not self.is_wall_blocking_exit(pos, orientation):
                        if not include_state:
                            wall_placements.append((*pos, orientation))
                        else:
                            copy_state = self.copy()
                            copy_state.place_wall((*pos, orientation), False)
                            wall_placements.append((copy_state, (*pos, orientation)))

                orientation = WallPieceStatus.VERTICAL
                # check verticals
                if self.is_wall_placement_valid(pos, orientation):
                    if not self.is_wall_blocking_exit(pos, orientation):
                        if not include_state:
                            wall_placements.append((*pos, orientation))
                        else:
                            copy_state = self.copy()
                            copy_state.place_wall((*pos, orientation), False)
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
    

    def place_wall(self, inp, check_valid = True):
        x, y, orientation = inp
        pos = x,y

        if check_valid:
            if not self.is_wall_placement_valid(pos, orientation) or self.is_wall_blocking_exit(pos, orientation):
                print('CRAP', inp)
                exit(2)

        if self.player1:
            self.player1_walls_num -= 1
        else:
            self.player2_walls_num -= 1

        self.wallboard[pos[0], pos[1]] = orientation
        self.walls_can_be_placed = []
            

    def move_piece(self, new_pos):
        
        if self.player1:
            pos = self.player1_pos
            otherpos = self.player2_pos
        else:
            pos = self.player2_pos
            otherpos = self.player1_pos

        if not self.is_valid_move(pos, new_pos) or np.array_equal(new_pos, otherpos):
            print('CRAP', new_pos)
            exit(2)

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
