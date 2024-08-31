import torch
from torch._tensor import Tensor
from trainer.input_processor import InputProcessor
import chess
from trainer.utils import config, n_past_moves
from logger import setup_logger

logger = setup_logger(__name__)

class AlphaZeroInputProcessor(InputProcessor):
    def __init__(self):
        super().__init__()

    def position_to_input(self, board: chess.Board) -> Tensor:
        turn = torch.ones((1, *config.input_shape[1:])) * (board.turn == chess.WHITE) 

        positions = self._get_past_positions(board)

        castling_rights = torch.concat(
            [
                torch.ones((1, *config.input_shape[1:])) * board.has_kingside_castling_rights(chess.WHITE),
                torch.ones((1, *config.input_shape[1:])) * board.has_queenside_castling_rights(chess.WHITE),
                torch.ones((1, *config.input_shape[1:])) * board.has_kingside_castling_rights(chess.BLACK),
                torch.ones((1, *config.input_shape[1:])) * board.has_queenside_castling_rights(chess.BLACK),
            ],
            axis = 0
        )

        en_passant = torch.zeros((1, *config.input_shape[1:]))

        if board.ep_square is not None:
            en_passant[0, board.ep_square // 8, board.ep_square % 8] = 1
        
        counter = torch.ones((1, *config.input_shape[1:])) * board.can_claim_draw()

        return torch.cat([turn, castling_rights, en_passant, counter, positions], axis=0)

    
    def _get_past_positions(self, board: chess.Board) -> Tensor:
        positions = torch.zeros((n_past_moves * 14, *config.input_shape[1:]))
        stack = []

        for i in range(n_past_moves):
            if board.move_stack != []:
                move = board.pop()
                stack.append(move)

            for color in chess.COLORS[::-1]:
                
                for piece in chess.PIECE_TYPES:

                    piece_positions = list(board.pieces(piece, color))
                    for pos in piece_positions:

                        r, c = pos // 8, pos % 8
                        positions[i * 14 + len(chess.PIECE_NAMES[1:]) * color + piece - 1, r, c] = 1
            
            repetition_index = i * 14 + len(chess.PIECE_NAMES[1:]) * (len(chess.COLORS))
            positions[repetition_index] = board.can_claim_threefold_repetition() * torch.ones(8, 8) 
            positions[repetition_index + 1] = board.can_claim_fifty_moves() * torch.ones(8, 8) 

            if board.move_stack == []:
                break
        while not (stack == []):
            board.push(stack.pop())
        
        return positions
        

if __name__ == "__main__":
    input_processor = AlphaZeroInputProcessor()
    board = chess.Board()
    input_tensor = input_processor.position_to_input(board)
    logger.info(f"Output Tensor shape: {input_tensor.shape}")





