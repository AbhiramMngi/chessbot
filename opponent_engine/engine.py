import chess
from chess.engine import SimpleEngine
from os import environ

import chess.engine

engine = SimpleEngine.popen_uci(environ["CHESS_ENGINE_PATH"])

def get_best_move(fen: str) -> str:
    return engine.play(chess.Board(fen=fen), limit=chess.engine.Limit(time=0.05)).move

close = lambda: engine.close()

    