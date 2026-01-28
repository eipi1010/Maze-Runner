from constant import MAZE,Action,Index,INDEX
import numpy as np
import random



class Player():
    def __init__(self,state,end):
        self.state = state
        self.start_state = state
        self.end = end
        self.action_record = []
        self.prev_actions = []
 
        self.length = len(MAZE)
        self.width = len(MAZE[0])
        self.q_table = np.zeros((self.length*self.width,4))

        self.gamma = 0.9
        self.alpha = 0.90
        self.epsilon = 1
        self.min_epsilon = 0.1
        

    def __str__(self):
        return str(self.q_table)
    
    def reward(self,state):
        return MAZE[INDEX[state]]
    
    def move(self,action):
        self.state += action
    
    
    def safe_move(self,action):
        if action == Index.UP:
            if self.legal(self.state,Action.UP):
                self.move(Action.UP)
                self.action_record.append("U")
        elif action == Index.DOWN:
            if self.legal(self.state,Action.DOWN):
                self.move(Action.DOWN)
                self.action_record.append("D")
        elif action == Index.LEFT:
            if self.legal(self.state,Action.LEFT):
                self.move(Action.LEFT)
                self.action_record.append("L")
        else:
            if self.legal(self.state,Action.RIGHT):
                self.move(Action.RIGHT)
                self.action_record.append("R")
        if self.state in self.end:
            self.prev_actions = self.action_record.copy()
            self.action_record.clear()


    
    def dead(self,action) -> bool:
        if action == Index.UP:
            if self.legal(self.state,Action.UP):
                return False
        elif action == Index.DOWN:
            if self.legal(self.state,Action.DOWN):
                return False
        elif action == Index.LEFT:
            if self.legal(self.state,Action.LEFT):
                return False
        else:
            if self.legal(self.state,Action.RIGHT):
                return False
        return True
    
    def actions(self,state) -> list[int]:
        self.legal(state)
        a: list[int] = []
        if state % self.width != 1:
            a.append(Action.LEFT) 
        if state % self.width != 0:
            a.append(Action.RIGHT) 
        if state > self.length:
            a.append(Action.UP)
        if state < self.length*self.width - self.length:
            a.append(Action.DOWN) 

        return a

    def legal(self,state,action):
        if action == Action.LEFT:
            return state % self.width != 0
        elif action == Action.RIGHT:
            return state % self.width != 2
        elif action == Action.UP:
            return state >= self.length
        else:
            return state < self.length*self.width-self.length
        

    def reset(self):
        self.state = self.start_state



