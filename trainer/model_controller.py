import torch 
import chess
from torch import nn
from typing import Callable
from trainer.input_processor import InputProcessor
from trainer.output_processor import OutputProcessor
from utils import config, TrainingConfig
from trainer.alpha_zero.model import build_alphazero_model

class ModelController(object):
    def __init__(
        self, 
        model_type: str,
        model_path: str | None, 
        input_processor: InputProcessor, 
        output_processor: OutputProcessor,
        training_config: TrainingConfig,  
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.ip = input_processor
        self.op = output_processor
        self.model, self.loss = self.model_init()
        self.training_config = training_config
    
    def pipeline(self, input_pos: chess.Board) -> chess.Move:
        input = self.ip.position_to_input(input_pos)
        output = self.model(input)
        return self.op.output_to_position(output, input_pos)
    
    def pipeline_batch(self, inputs: list[chess.Board]) -> list[chess.Move]:
        return [
            self.pipeline(board) for board in inputs
        ]
    
    def load_model(self):
        if self.model_path is not None:
            self.model = torch.load(self.model_path)
    
    def model_init(self) -> tuple[nn.Module, Callable]:
        if self.model_type == "AlphaZero":
            return build_alphazero_model(config)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
    def train(self, dataset: list[tuple[chess.Board, chess.Move]]):
        pass


    def test(self, dataset: list[tuple[chess.Board, chess.Move]]):
        pass
    
    

        

