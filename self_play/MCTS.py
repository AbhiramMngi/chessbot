from self_play.node import Node
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import chess
from typing import Callable
import random
# from opponent_engine.engine import get_best_move
from opponent_engine.engine import close as close_engine
from trainer.utils import mcts_depth_limit
from chess.pgn import Game, FileExporter
import datetime
import tqdm
from logger import setup_logger


logger = setup_logger(__name__)
class MCTS:
    # white - 1
    # black - 0
    def __init__(self, predict: Callable, root: Node = None, color: int = 1, mode: str = "train"):
        self.game_path = []
        self.color = color
        self.predict = predict
        self.depth_limit = mcts_depth_limit # default depth limit
        # logger.info("MCTS initialized")
        self.root = Node(chess.Board(), parent=None, move = "") if root is None else root
        self.board = chess.Board()
        self.node_count = 1
        self.mode = mode
        self.move_stack = []
        self.df = {"fen": [], "eval": []}

    def run_simulations(self, n_simulations):
        try:
            for i in tqdm.tqdm(range(n_simulations)):
                self.board = chess.Board()
                leaf = self.select_leaf(self.root)
                value = self.expand_node(leaf)
                self.backpropagation(value)

            
            logger.info(f"total nodes: {self.node_count}")
            if n_simulations < 2: 
                self.plot_tree()
        finally:
            close_engine()
            self.save_tree(self.root)
            import pandas as pd
            pd.DataFrame(self.df).to_csv(f"{datetime.datetime.now()}.csv")

    def select_leaf(self, root: Node) -> Node:

        self.game_path.append(root)
        if len(root.arrival_move) > 0:
            self.board.push(chess.Move.from_uci(root.arrival_move))
        if root.legal_moves == [] or root.children == []: # terminal
            return root
        
        scores = np.array([child.position_score() for child in root.children])
        dirichlet = np.random.dirichlet([0.3] * scores.shape[0]) * scores
        
        best_child = root.children[dirichlet.argmax()]

        return self.select_leaf(best_child)

    def expand_node(self, leaf: Node):
        if leaf.legal_moves == []: # terminal
            return self.simulate_game(leaf) # +1 if white, -1 if black
        
        max_node, max_move = None, None

        dirichlet = np.random.dirichlet([0.3] * len(leaf.legal_moves))
        for i, move in enumerate(leaf.legal_moves):
            self.board.push(move)
            new_node = Node(self.board, leaf, move.uci())
            self.node_count += 1
            leaf.children.append(new_node)
            self.board.pop()
        scores = dirichlet * np.array([child.position_score() for child in leaf.children])
        idx = scores.argmax()
        max_node, max_move = leaf.children[idx], leaf.legal_moves[idx]

        self.board.push(max_move)

        self.game_path.append(max_node)
        return self.simulate_game(leaf)

    def simulate_game(self, node: Node) -> float:
        while True:
            if self.board.is_game_over(): 
                # logger.info(f"Game over: {self.board.fen()}")
                break
            move = None
            # if self.board.turn == chess.WHITE:
            move, _ = self.predict(self.board)
            # else:
            # move = get_best_move(self.board.fen())  # Replace with your opponent's engine
            self.board.push(move)
        
        self.save_sim()

        return 0.5 if self.board.outcome() is None or self.board.outcome().winner is None else self.board.outcome().winner * 2 - 1

    def backpropagation(self, value: float):
        for node in self.game_path[::-1]:
            node.N += 1
            node.W += value 
            node.Q = node.W / node.N

    def save_sim(self):
        game = Game()
        
        node = game
        for move in self.board.move_stack:
            # logger.info(f"{move.uci()}")
            node = node.add_variation(move)

        with open(f"games/{datetime.datetime.now()}", "w") as f:
            exporter = FileExporter(f)
            game.accept(exporter)
        
        # close()
        
    
    def _add_edge(self, graph: nx.Graph, node: Node, parent: Node = None):
        if parent:
            graph.add_edge(str(parent), str(node))
        for child in node.children:
            self._add_edge(graph, child, parent=node)
    def plot_tree(self):
        G = nx.DiGraph()

        self._add_edge(G, self.root)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        
        nx.draw(G, pos, with_labels=True, node_color="lightblue",arrows=True)
        
        plt.show()
    
    def save_tree(self, root: Node) -> None:
        if root is None:
            return 
        self.df["fen"].append(root.state)
        self.df["eval"].append(root.Q)
        for child in root.children:
            self.save_tree(child)
        


        