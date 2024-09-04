import torch
from trainer.output_processor import OutputProcessor
import chess
from trainer.utils import config, knight_move_indices, underpromotion_move_indices, PROMOTION_MAP
from logger import setup_logger

logger = setup_logger(__name__)

class AlphaZeroOutputProcessor(OutputProcessor):
    
    def __init__(self):
        super().__init__() 
        self.move_dict = {}
    def output_to_position(self, output: torch.Tensor, board: chess.Board) -> chess.Move:
        list_of_moves = list(output.squeeze())
        
        legal_moves = board.generate_legal_moves()
        move_final = ["", 0]

        for move in legal_moves:
            move_prob = list_of_moves[self._get_output_index_from_move(move)]
            if move_final[1] < move_prob:
                move_final = [move, move_prob]
        return move_final[0]
    
    def _is_under_promotion(self, promotion_piece: int) -> bool:

        return promotion_piece not in [None, 5]
    
    def _is_queenlike_move(self, from_square: int, to_square: int) -> bool:
        diff = abs(from_square - to_square)
        return diff % 7 == 0 or diff % 9 == 0 or diff % 8 == 0 or ((from_square // 8) == (to_square // 8))

    def _get_output_index_from_move(self, move: chess.Move) -> torch.Tensor:
        
        promotion_piece = move.promotion
        from_square, to_square = move.from_square, move.to_square
        # logger.info(f"from_square: {from_square} to {to_square} promotion piece: {promotion_piece}")
        
        if self._is_queenlike_move(from_square, to_square) and (not self._is_under_promotion(promotion_piece)): # QueenLike Move

            from_square_idx = from_square * 56
            diff = to_square - from_square
            steps, direction = None, None

            if diff % 7 == 0 and -7 <= (diff // 7 ) <= 7: # nortwest Diagonal
                steps = diff // 7
                direction = 0
                if steps > 0: direction = 5
                else: direction = 6
                
            elif diff % 9 == 0 and -9 <= (diff // 7) <= 9: # northeast Diagonal
                steps = diff // 9
                direction = 1
                if steps > 0: direction = 4
                else: direction = 7

            elif diff % 8 == 0: # up/down
                steps = diff // 8
                direction = 2
                if steps > 0: direction = 2
                else: direction = 3

            else: # left/right
                steps = diff
                direction = 0
                if steps > 0: direction = 1
                else: direction = 0

            return from_square_idx + direction * 8 + steps - 1
        
        elif self._is_under_promotion(promotion_piece): # Under Promotion
            direction = to_square - from_square - 8 # -1, 0, 1
            promotion_piece -= 2
            from_square_idx = underpromotion_move_indices[0] + from_square * 9
            # logger.info(f"promotion piece: {PROMOTION_MAP[promotion_piece]}")
            return from_square_idx + promotion_piece * 3 + direction

        else: # Knight Move

            from_square_idx = knight_move_indices[0] + from_square * 8
            diff = to_square - from_square
            # logger.info(f"From square: {from_square} to_square: {to_square} diff: {diff}")

            if diff == -10: # LLEFT_SDOWN
                return from_square_idx
            elif diff == 6: # LLEFT_SUP
                return from_square_idx + 1
            elif diff == -6: # LRIGHT_SDOWN
                return from_square_idx + 2
            elif diff == 10: # LRIGHT_SUP
                return from_square_idx + 3
            elif diff == -17: # SLEFT_LDOWN
                return from_square_idx + 4
            elif diff == 15: # SLEFT_LUP
                return from_square_idx + 5
            elif diff == -15: # SRIGHT_LDOWN
                return from_square_idx + 6
            elif diff == 17: # SRIGHT_LUP
                return from_square_idx + 7

    def get_output_tensor_from_move(self, move: chess.Move) -> torch.Tensor:
        arr = torch.zeros(config.output_shape)
        arr[self._get_output_index_from_move(move)] = 1
        return arr
    
if __name__ == "__main__":
    output_processor = AlphaZeroOutputProcessor()

    idx = output_processor.get_output_tensor_from_move(chess.Move.from_uci("g7f5")).argmax()
    logger.info(f"Index: {idx}, Move: {chess.Move.from_uci('g7f5')}")
    t = torch.zeros(config.output_shape)
    t[idx, :] = 1
    logger.info(output_processor.output_to_position(t, chess.Board("8/6N1/8/8/8/K7/8/k7 w - - 0 1")))