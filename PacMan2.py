import sys
import os

from Cell import Cell

# maze = []
def read_maze(file_path):
    maze = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [Cell(int(char)) for char in line.strip()]
            maze.append(row)
    return maze

file_path = "map.txt"
maze = read_maze(file_path)

# for row in maze:
#     print([element.name for element in row])
pacman_start = [18,14]
maze[18][14] = Cell.PACMAN


for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i][j] == Cell.PACMAN:
            print('Pacman at: ', i, j)