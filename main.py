from console.game import Game

if __name__ == '__main__':
    g = Game(False, True, 10, 0.2)
    g.play()
    print(g.wins)

