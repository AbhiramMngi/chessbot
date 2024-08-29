from torch._tensor import Tensor
from trainer.input_processor import InputProcessor

class AlphaZeroInputProcessor(InputProcessor):
    def __init__(self):
        super().__init__()
    def _fen_to_input(self, input_fen: str) -> Tensor:
        pass        
