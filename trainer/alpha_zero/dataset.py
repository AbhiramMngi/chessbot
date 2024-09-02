import torch 
from torch.utils.data import Dataset
import chess.pgn
from trainer.input_processor import InputProcessor
from trainer.output_processor import OutputProcessor


# TODO: Complete the dataset implementation after MCTS
class AlphaZeroDataset(Dataset):
    def __init__(
        self,
        file_path: str
    ):
        super().__init__()
        self.games = []

        with open(file_path, 'r') as file:
            while True:
                game = self.games.append(chess.pgn.read_game(file))
                if not game:
                    break
                self.games.append(game)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        board = game.board()
        move = game.fullmove_number
        input_processor = InputProcessor()
        output_processor = OutputProcessor()
        input = input_processor.position_to_input(board)
        output = output_processor.get_output_tensor_from_move(move)
        return input, output
