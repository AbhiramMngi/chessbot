import torch
from trainer.output_processor import OutputProcessor


class AlphaZeroOutputProcessor(OutputProcessor):
    def __init__(self):
        super().__init__()
        pass 
    def _output_to_fen(self, output: torch.Tensor) -> str:
        pass