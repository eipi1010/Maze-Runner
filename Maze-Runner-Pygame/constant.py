import numpy as np
from enum import Enum,IntEnum
MAZE = np.array([
    [-3,-4,-7],
    [-2,-5,-6],
    [-10,-3,10],
    ])

COORDINATES = {
0: (85, 85),
1: (85+150, 85),
2: (85+150*2, 85),
3: (85, 85+150),
4: (85+150, 85+150),
5: (85+150*2, 85+150),
6: (85, 85+150*2),
7: (85+150, 85+150*2),
8: (85+150*2, 85+150*2),
}

INDEX = {
0:(0,0),
1:(0,1),
2:(0,2),
3:(1,0),
4:(1,1),
5:(1,2),
6:(2,0),
7:(2,1),
8:(2,2),
}

    # Colors
class Color():
    WHITE = (255, 255, 255)
    RED_OLD = (255, 100, 100)   # Old Q-Value
    GREEN_R = (0, 255, 0)       # Reward
    YELLOW_A = (255, 255, 0)    # Alpha
    CYAN_G = (0, 255, 255)      # Gamma
    MAGENTA_MAX = (255, 0, 255) # Max Future Q

class Action(IntEnum):
    LEFT = -1
    RIGHT = 1
    UP = -MAZE.shape[0]
    DOWN = MAZE.shape[1]

class Index(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3