import chess
import numpy as np
from trainer.utils import c

class Node(object):
    def __init__(self, state: chess.Board, parent, move: str):
        self.state = state.fen()
        self.legal_moves = list(state.generate_legal_moves())
        self.turn = state.turn
        self.children = []
        self.arrival_move = move
        self.W = 0
        self.N = 0
        self.Q = 0
        self.parent = parent
        
    
    def serialize(self):
        pass # TODO

    def position_score(self):
        return (
            self.Q + # exploitation factor
            c * np.sqrt(np.log(self.parent.N + 0.001)/ (self.N + 1)) # exploration factor
        ) 
        
