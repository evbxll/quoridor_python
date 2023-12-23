#! /usr/bin/env python3

import sys
import os
import subprocess
import signal
import atexit
from time import time
from getopt import getopt
from select import select
import importlib
from other.game_imp import *


move_time = 30.0                    # allowed maximum time for a movement - default 30 seconds
verbose = 2                        # default 0, 1 to list moves, 2 to draw board
games = 1                           # how many games to play - default 1
gameCounter = 0
seed = int(time())                  # seed to pass to players - default current unix timestamp
starting_walls = 10                # how many walls each player has available at the start of the game - default(-1) 7/4*boardsize-23/4
maximum_time = -1                 # maximum number of seconds players can consume for all their moves - default(-1) 20*boardsize
memory_limit = None                 # memomy limit of player's subrocesses, if zero not checking memory usage - default 950MB



p1 = "players/random_player.py"
p1 = importlib.import_module(p1)

p2 = "players/random_player.py"
p2 = importlib.import_module(p2)

board = Board()
board.player1.player_class = getattr(p1 , "Bot")(WHITE)
board.player2.player_class = getattr(p2 , "Bot")(BLACK)

def usage():
    print("SETUP parameters incorrect")
    sys.exit(2)



def invalidMove(color, x, y, orientation):
    if color==BLACK:
        winner=board.player_2
        loser=board.player_1
    else:
        winner=board.player_1
        loser=board.player_2
    if orientation:
        move = "%s %s" % (board.coordsToString(x,y), board.orientationToString(orientation))
    else:
        move = "%s" % board.coordsToString(x,y)
    print("\n%s tried an invalid move: %s" % (loser.getTitle(), move))
    print("%s wins game %d\n" % (winner.getTitle(), gameCounter+1))
    print("--------------------")


def memoryExceed(color):
    if color==BLACK:
        winner=board.player_2
        loser=board.player_1
    else:
        winner=board.player_1
        loser=board.player_2
    print("\n%s exceeded the memory limit" % loser.getTitle())
    print("%s wins game %d\n" % (winner.getTitle(), gameCounter+1))
    print("--------------------")
    loser.kill()


def timeOut(color):
    if color==BLACK:
        winner=board.player_2
        loser=board.player_1
    else:
        winner=board.player_1
        loser=board.player_2
    print("\n%s ran out of time" % loser.getTitle())
    print("%s wins game %d\n" % (winner.getTitle(), gameCounter+1))
    print("--------------------")
    loser.kill()


def unexpectedOutput(player, s):
    if(verbose>0):
        print("unexpected output from %s: \"%s\" after command: \"%s\"" % (player.getTitle(), s, player.last_command))


def winner(player):
    print("\n%s wins game %d" % (player.getTitle(), gameCounter+1))
    print("%s took %.2fs to decide its moves"%(board.player_1.getTitle(), maximum_time-board.player_1.time_left))
    print("%s took %.2fs to decide its moves"%(board.player_2.getTitle(), maximum_time-board.player_2.time_left))
    print("\n--------------------")


def initialize_game():
    if not board.player_1.running:
        board.player_1.run()
    if not board.player_2.running:
        board.player_2.run()
    board.walls = []
    board.player_1.walls = starting_walls
    board.player_2.walls = starting_walls

    board.player_1.time_left = maximum_time
    board.player_2.time_left = maximum_time

    board.player_1.y = int(0)
    board.player_1.x = int(board.size/2)

    board.player_2.y = int(board.size-1)
    board.player_2.x = int(board.size/2)


def playGame():
    initialize_game()
    print("\n\nGame %d starting now!\n" % (1+gameCounter))
    if verbose>1:
        board.print_board()
    while True:
        start_time = time()
        move = board.player_1.readMove(min(move_time, board.player_1.time_left), board, board.player1.x, board.player1.y, board.player1)
        board.player_1.time_left -= time()-start_time
        if move: #no timeout
            if move[0]:  # wall
                if not board.playwall(move[1]):
                    invalidMove(WHITE, move[1], move[2], move[0])
                    return BLACK
                if verbose>0:
                    print("%s places %s wall at %s" % (board.player_1.getTitle(), board.orientationToString(move[0]), board.coordsToString(move[1], move[2])))
            else:  # move
                if not board.playmove(move[1]):
                    invalidMove(WHITE, move[1])
                    return BLACK
                if verbose>0:
                    print("%s moves to %s" % (board.player_1.getTitle(), board.coordsToString(*move[1])))
            if verbose>1:
                board.print_board()
            if board.player_1.winner():
                winner(board.player_1)
                return WHITE
        else:
            timeOut(WHITE)
            return WHITE

        start_time = time()
        move = board.player_2.readMove(min(move_time, board.player_2.time_left), board, board.player2.x, board.player2.y, board.player2)

        board.player_2.time_left -= time()-start_time
        if move:  # no timeout
            if move[0] == 'w':  # wall
                if not board.playwall(move[1], move[2], move[0], board.player_2):
                    invalidMove(BLACK, move[1], move[2], move[0])
                    return WHITE
                if verbose>0:
                    print("%s places %s wall at %s" % (board.player_2.getTitle(), board.orientationToString(move[0]), board.coordsToString(move[1], move[2])))
            elif move[0] == 'm':  # move
                if not board.playmove(*move[1]):
                    invalidMove(BLACK, move[1])
                    return WHITE
                if verbose>0:
                    print("%s moves to %s" % (board.player_2.getTitle(), board.coordsToString(move[1])))
            else:
                invalidMove(WHITE, move[1], move[2], move[0])
            if(verbose>1):
                board.print_board()
            if board.player_2.winner():
                winner(board.player_2)
                return WHITE
        else:
            timeOut(WHITE)
            return BLACK


print("\n--------------------\n")

####read parameters####
if maximum_time<0: # default=20*boardsize
    maximum_time=board.size*20
if starting_walls<0: # default=7/4*boardsize-23/4
    starting_walls=int(7.0/4*board.size-23.0/4)

if board.size<3 or board.size>25 or board.size%2==0 or starting_walls<0 or games<=0 or maximum_time<=0 or move_time<=0:
    usage()

print("boardsize set to %d" % board.size)
print("walls set to %d" % starting_walls)
print("verbose set to %d" % verbose)
print("games set to %d" % games)
print("maximum_time set to %d" % maximum_time)
print("move_time set to %.1f" % move_time)
if memory_limit:
    print("memory_limit set to %d" % memory_limit)
print("seed set to %d" % seed)

####check files exist and are executable####

while not os.path.exists(board.player_1.path):
    print("File %s does not exist, please specify the filename correctly" % board.player_1.path)
    board.player_1.path = input('> ')
if not os.access(board.player_1.path, os.X_OK):
    print("File %s is not executable, quiting..." % board.player_1.path)
    sys.exit(3)

while not os.path.exists(board.player_2.path):
    print("File %s does not exist, please specify the filename correctly" % board.player_2.path)
    board.player_2.path = input('> ')
if not os.access(board.player_2.path, os.X_OK):
    print("File %s is not executable, quiting..." % board.player_2.path)
    sys.exit(3)

####start subprocesses####
board.player_1.start()
board.player_2.start()


####at exit####

def cleanup():
    board.player_1.kill()
    board.player_2.kill()


####Save game script####

f = open("last_game.sh","w")
f.write("#!/bin/bash\n\n")
if "--seed" not in sys.argv:
    sys.argv.append("--seed %d"%seed)
f.write(" ".join(sys.argv))
f.write("\n")
f.close()
mode = os.stat("last_game.sh").st_mode
mode |= (mode & 0o444) >> 2
os.chmod("last_game.sh", mode)

####Games starting####

print("\n--------------------")
print("%s vs %s" % (board.player_1.getTitle(), board.player_2.getTitle()))
print("----------------------")


while gameCounter < games:
    i = playGame()
    if i==BLACK:
        board.player_1.gamesWon += 1
    else:
        board.player_2.gamesWon += 1
    gameCounter+=1
    # if abs(board.player_1.gamesWon-board.player_2.gamesWon)>games-gameCounter:
    #     break

if games > 1:
    print("%s won %d games" % (board.player_1.getTitle(), board.player_1.gamesWon))
    print("%s won %d games" % (board.player_2.getTitle(), board.player_2.gamesWon))
    if board.player_1.gamesWon > board.player_2.gamesWon:
        print("\n%s wins!!" % board.player_1.getTitle())
    elif board.player_1.gamesWon < board.player_2.gamesWon:
        print("\n%s wins!!" % board.player_2.getTitle())
    else:
        print("\nIt's a tie!!")
    print("--------------------\n")


