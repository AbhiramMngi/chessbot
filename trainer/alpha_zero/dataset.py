from torch.utils.data import Dataset
import chess
import torch
import pandas as pd
from trainer.utils import file_path
from opponent_engine.engine import get_best_move, close
from trainer.alpha_zero.input_processor import AlphaZeroInputProcessor
from trainer.alpha_zero.output_processor import AlphaZeroOutputProcessor

class AlphaZeroDataset(Dataset):
    def __init__(
        self
    ):
        super().__init__()
        self.data = pd.read_csv(file_path, index_col="Unnamed: 0")
        self.op = AlphaZeroOutputProcessor()
        self.ip = AlphaZeroInputProcessor()
        self.offset = 0

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        fen, eval = self.data.iloc[idx]
        
        board = chess.Board(fen)


        inputs = self.ip.position_to_input(board)
        outputs = [self.op.get_output_tensor_from_move(get_best_move(fen))]
        outputs.append(torch.Tensor([eval]).squeeze())

        return inputs[0], outputs
    
