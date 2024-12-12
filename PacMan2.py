import sys
import os

from Cell import Cell

maze = []
def read_maze(file_path):
    maze = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [Cell(int(char)) for char in line.strip()]
            maze.append(row)
    return maze

file_path = "map.txt"
maze_2d_list = read_maze(file_path)

for row in maze_2d_list:
    print([element.name for element in row])