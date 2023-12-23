#! /usr/bin/env python3
from time import time
from getopt import getopt
from select import select


EMPTY = 0
WHITE = 1
BLACK = 2

noneORIENTATION = 0
HORIZONTAL = 1
VERTICAL = 2


class Wall(object):
    x = 0                           # wall's x coordinate
    y = 0                           # wall's y coordinate
    orientation = noneORIENTATION   # walls's orientation

    def __init__(self, x, y, orientation):
        self.x = x
        self.y = y
        self.orientation = orientation

class Player(object):
    x = 0
    y = 0
    name = ""
    imported_player = None
    walls = 10
    color = None
    size = 9
    wins = (0,0)
    losses = (0,0)
    ties = (0,0)
    time_left = 0                       # time left for the player to complete all his moves
    last_command = ""                   # last message sent to player's stdin
    running = False

    def __init__(self, color):
        self.color = color

    def set_player(self, player):
        self. player_class = player

    def readMove(self, time, *kwargs):
        #implement check on whether time has elapsed

        return self.player_class.readMove(*kwargs)
    
    def start(self, *kwargs):
        return self.player_class.start(*kwargs)

    def kill(self, *kwargs):
        return self.player_class.kill(*kwargs)
    
    def getTitle(self):
        return self.color
    
    def update_score(self, wins_diff, ties_diff, loss_diff):
        c = 0 if (self.color == WHITE) else 1
        self.wins[c] += wins_diff
        self.ties[c] += ties_diff
        self.losses[c] += loss_diff
        return (self.wins, self.ties, self.losses)

    def swap_color(self):
        self.color = BLACK if (self.color == WHITE) else WHITE

    def winner(self):
        if (self.color==BLACK and self.y == 0) or (self.color==WHITE and self.y==self.size-1):
            return True
        return False



class Board(object):
    p1 = Player(WHITE)
    p2 = Player(BLACK)
    size = 9
    walls = []

    def coordsToString(self, x, y):
        return "%c%d" % (ord("A")+x, self.size-y)

    def stringToCoords(self, s):
        try:
            x = ord(s[0].lower())-ord('a')
            y = self.size-int(s[1:3])
            return x, y
        except:
            return None

    def orientationToString(self, orientation):
        if orientation==HORIZONTAL:
            return "horizontal"
        elif orientation==VERTICAL:
            return "vertical"
        else:
            return "unknown orientation"

    def wallExists(self, x, y, orientation):
        for wall in self.walls:
            if (wall.x == x and wall.y == y and wall.orientation == orientation):
                return True
        return False

    def print_board(self):
        print("\n     ")
        for j in range(0, self.size):
            print("%c   " % (ord('A')+j))
        print("")
        for i in range(0, self.size):
            print("   ")
            for j in range(0, self.size):
                if self.wallExists(j-1, i-1, HORIZONTAL):
                    print('=')
                elif self.wallExists(j-1, i-1, VERTICAL):
                    print('H')
                else:
                    print('+')
                if self.wallExists(j-1, i-1, HORIZONTAL) or self.wallExists(j, i-1, HORIZONTAL):
                    print("===")
                else:
                    print("---")
            print("+")
            print("%2d " % (self.size-i))
            for j in range(0, self.size):
                if self.wallExists(j-1, i-1, VERTICAL) or self.wallExists(j-1, i, VERTICAL):
                    print('H')
                else:
                    print('|')
                if self.player_1.x==j and self.player_1.y==i:
                    print(" B ")
                elif self.player_2.x==j and self.player_2.y==i:
                    print(" W ")
                else:
                    print("   ")
            print("| %d" % (self.size-i))
            if i==0:
                print("  black walls: %2d" % self.player_1.walls)
            elif i==1:
                print("  white walls: %2d" % self.player_2.walls)
            else:
                print("")
        print("   ")
        for j in range(0, self.size):
            print("+---")
        print("+\n     ")
        for j in range(0, self.size):
            print("%c   " % (ord('A')+j))
        print("\n")


    def thereIsWall(self, p_x, p_y, x, y):
        if p_x==x:
            m = min(p_y, y)
            M = max(p_y, y)
            for i in range(m, M):
                if self.wallExists(p_x, i, HORIZONTAL) or (p_x>0 and self.wallExists(p_x-1, i, HORIZONTAL)):
                    return True
        elif p_y==y:
            m = min(p_x, x)
            M = max(p_x, x)
            for i in range(m, M):
                if self.wallExists(i, p_y, VERTICAL) or (p_y>0 and self.wallExists(i, p_y-1, VERTICAL)):
                    return True
        else:
            return True
        return False

    def checkValidMove(self, x, y, player):
        if x<0 or y<0 or x>=self.size or y>=self.size:
            return False
        if (self.player_1.x==x and self.player_1.y==y) or (self.player_2.x==x and self.player_2.y==y):
            return False
        distance = abs(player.x-x)+abs(player.y-y)
        if distance==1:
            if self.thereIsWall(player.x, player.y, x, y):
                return False
            return True
        elif distance==2 and (player.x==x or player.y==y):
            x_middle = (player.x+x)/2
            y_middle = (player.y+y)/2
            if not ((self.player_1.x==x_middle and self.player_1.y==y_middle) or (self.player_2.x==x_middle and self.player_2.y==y_middle)):
                return False
            if self.thereIsWall(player.x, player.y, x, y):
                return False
            return True
        elif distance==2:
            if player.color==BLACK:
                opponent=self.player_2
            else:
                opponent=self.player_1
            if (abs(player.x-opponent.x)+abs(player.y-opponent.y))!=1:
                return False
            xBehind=player.x
            yBehind=player.y
            if player.x+1==opponent.x:
                xBehind=player.x+2
            elif player.x-1==opponent.x:
                xBehind=player.x-2
            elif player.y+1==opponent.y:
                yBehind=player.y+2
            elif player.y-1==opponent.y:
                yBehind=player.y-2
            else:
                return False
            if (xBehind < 0 or xBehind >= self.size or yBehind < 0 or yBehind >= self.size or self.thereIsWall(opponent.x, opponent.y, xBehind, yBehind)) and not self.thereIsWall(player.x, player.y, opponent.x, opponent.y) and not self.thereIsWall(opponent.x, opponent.y, x, y):
                return True
        return False

    def playmove(self, x, y, player):
        if not self.checkValidMove(x, y, player):
            return False
        player.x = x
        player.y = y
        return True


    def tryToReachOpponentSide(self, player, tmpBoard):

        if player.winner():
            return True
        x = player.x
        y = player.y
        tmpBoard[x][y] = 1
        if player.color==BLACK:
            sign=-1
        else:
            sign=1
        if self.playmove(x, y+sign, player):
            if not tmpBoard[x][y+sign] and self.tryToReachOpponentSide(player, tmpBoard):
                return True
        player.y=y
        if self.playmove(x+1, y, player):
            if not tmpBoard[x+1][y] and self.tryToReachOpponentSide(player, tmpBoard):
                return True
        player.x=x
        if self.playmove(x-1, y, player):
            if not tmpBoard[x-1][y] and self.tryToReachOpponentSide(player, tmpBoard):
                return True
        player.x=x
        if self.playmove(x, y-sign, player):
            if not tmpBoard[x][y-sign] and self.tryToReachOpponentSide(player, tmpBoard):
                return True
        player.y=y
        return False

    def checkValidBoard(self):
        tmpBoard = [[0 for i in range(self.size)] for j in range(self.size)]
        XsYs = self.player_1.x, self.player_1.y, self.player_2.x, self.player_2.y
        self.player_2.x = -1
        self.player_2.y = -1
        if not self.tryToReachOpponentSide(self.player_1, tmpBoard):
            self.player_1.x, self.player_1.y, self.player_2.x, self.player_2.y = XsYs
            return False
        tmpBoard = [[0 for i in range(self.size)] for j in range(self.size)]
        _, _, self.player_2.x, self.player_2.y = XsYs
        self.player_1.x = -1
        self.player_1.y = -1
        if not self.tryToReachOpponentSide(self.player_2, tmpBoard):
            self.player_1.x, self.player_1.y, self.player_2.x, self.player_2.y = XsYs
            return False
        self.player_1.x, self.player_1.y, self.player_2.x, self.player_2.y = XsYs
        return True

    def checkValidWall(self, x, y, orientation):
        if x<0 or y<0 or x>self.size-2 or y>self.size-2:
            return False
        for wall in self.walls:
            if ((wall.x==x and wall.y==y) or
                (wall.x-1==x and wall.y==y and wall.orientation==orientation and orientation==HORIZONTAL) or
                (wall.x+1==x and wall.y==y and wall.orientation==orientation and orientation==HORIZONTAL) or
                (wall.x==x and wall.y-1==y and wall.orientation==orientation and orientation==VERTICAL) or
                (wall.x==x and wall.y+1==y and wall.orientation==orientation and orientation==VERTICAL)):
                return False
        return True

    def playwall(self, x, y, orientation, player):
        if player.walls <= 0 or not self.checkValidWall(x, y, orientation):
            return False
        self.walls.append(Wall(x, y, orientation))
        if self.checkValidBoard():
            player.walls -= 1
            return True
        else:
            self.walls.pop()
        return False
    
    def check_playwall(self, x, y, orientation, player):
        if player.walls <= 0 or not self.checkValidWall(x, y, orientation):
            return False
        self.walls.append(Wall(x, y, orientation))
        val = self.checkValidBoard()
        self.walls.pop()
        return val


    
