import torch 
import chess
from torch import nn
from typing import Callable
from trainer.input_processor import InputProcessor
from trainer.output_processor import OutputProcessor
from trainer.utils import config, TrainingConfig
from trainer.model_instances import alphazero_model, az_loss_fn, az_model_save
from logger import setup_logger

logger = setup_logger(__name__)
class ModelController(object):
    def __init__(
        self, 
        model_type: str,
        input_processor: InputProcessor, 
        output_processor: OutputProcessor,
        training_config: TrainingConfig,  
    ):
        self.model_type = model_type
        self.ip = input_processor
        self.op = output_processor
        self.model, self.loss, self.save = self.model_init()
        self.training_config = training_config
    
    def pipeline(self, input_pos: chess.Board) -> tuple[chess.Move, torch.Tensor]:
        input = self.ip.position_to_input(input_pos)
        # logger.info(f"Input sending of shape: {input.shape}")
        output = self.model(input)
        return self.op.output_to_position(output[0], input_pos), output[1]
    
    def pipeline_batch(self, inputs: list[chess.Board]) -> list[chess.Move]:
        return [
            self.pipeline(board) for board in inputs
        ]
    
    def predict(self, input_pos: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_pos = chess.Board(input_pos)
        input = self.ip.position_to_input(input_pos)
        output = self.model(input)
        return output[0], output[1]
    
    def model_init(self) -> tuple[nn.Module, Callable, Callable]:
        if self.model_type == "AlphaZero":
            return alphazero_model, az_loss_fn, az_model_save
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
    
    def train(self, dataset: list[tuple[chess.Board, chess.Move]]):
        pass


    def test(self, dataset: list[tuple[chess.Board, chess.Move]]):
        pass
    
    

        

