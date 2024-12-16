from enum import Enum

class Cell(Enum):

    BLANK = 3
    WALL = 1
    GATE = 2
    BALL = 0
    POWERBALL = 4
    PACMAN = 5
    GHOST = 6
