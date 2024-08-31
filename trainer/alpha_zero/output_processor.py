import torch
from trainer.output_processor import OutputProcessor
import chess
from multiprocessing import Process, Array, Manager
from multiprocessing.managers import DictProxy
from multiprocessing.sharedctypes import SynchronizedArray
from trainer.utils import config, QueenLikeDirection, KnightMove, n, n_underpromotion_moves, n_queenlike_moves, n_possible_outputs, queenlike_move_indices, knight_move_indices, underpromotion_move_indices, n_knight_moves, PROMOTION_MAP
from logger import setup_logger

logger = setup_logger(__name__)
class AlphaZeroOutputProcessor(OutputProcessor):
    def __init__(self):
        super().__init__() 
    def output_to_position(self, output: torch.Tensor, board: chess.Board) -> chess.Move:
        list_of_moves = list(output.squeeze())
        
        manager = Manager()

        move_dict = manager.dict()
        legal_moves = set([i.uci() for i in board.legal_moves])
        processes = [
            Process(
                target = self._output_to_move, 
                args=(list_of_moves, move_dict, i, legal_moves)
            )
            for i in range(8)
        ]

        for p in processes:
            p.start()
            p.join()

        max_prob, max_move = 0, ""
        for val in move_dict.values():
            if max_prob < val[1]:
                max_move, max_prob = val[0], val[1]
        return max_move
    
    def _output_to_move(self, array: list, move_dict: DictProxy, part: int, legal_moves: set) -> chess.Move:
        between = lambda arr, val: arr[0] <= val <= arr[1]

        start = part * (n_possible_outputs * n)
        end = (part + 1) * (n_possible_outputs * n) - 1
        max_prob, max_move = 0, ""
        moveq = None
        for i in range(start, end + 1):
            
            if between(queenlike_move_indices, i):
                
                move= self._get_queenlike_move(i)
                moveq = chess.Move.from_uci(move.uci() + "q") if move else None

            elif between(knight_move_indices, i):

                move = self._get_knight_move(i - knight_move_indices[0])
                moveq = move

            elif between(underpromotion_move_indices, i):

                move = self._get_underpromotion_move(i - underpromotion_move_indices[0])
                moveq = move

            if move is not None and move.uci() in legal_moves:
                if max_prob < array[i]:
                    max_move, max_prob = move, array[i]

            elif moveq is not None and moveq.uci() in legal_moves:
                if max_prob < array[i]:
                    max_move, max_prob = moveq, array[i]
        
        move_dict[part] = [max_move, max_prob]
    
    def _get_queenlike_move(self, i: int) -> chess.Move | None: 
        square = i //  n_queenlike_moves
        move_nr = i % n_queenlike_moves
        
        direction = move_nr // 7    
        steps = ((move_nr % 8) + 1)
        end_square = None
        if direction == QueenLikeDirection.LEFT.value: # LEFT
            end_square = square - steps
        if direction == QueenLikeDirection.RIGHT.value: # RIGHT
            end_square = square + steps
        if direction == QueenLikeDirection.UP.value: # UP
            end_square = square + 8 * steps
        if direction == QueenLikeDirection.DOWN.value: # DOWN
            end_square = square - 8 * steps
        if direction == QueenLikeDirection.NORTH_EAST.value: # NORTHEAST
            end_square = square + 8 * steps + steps
        if direction == QueenLikeDirection.NORTH_WEST.value: # NORTHWEST
            end_square = square + 8 * steps - steps
        if direction == QueenLikeDirection.SOUTH_EAST.value: # SOUTHEAST
            end_square = square - 8 * steps + steps
        if direction == QueenLikeDirection.SOUTH_WEST.value: # SOUTHWEST
            end_square = square - 8 * steps - steps

        move = self._create_move(square, end_square, "")
        return move
            
    def _get_knight_move(self, i: int) -> chess.Move | None:
        square = i //  n_knight_moves
        direction = i % n_knight_moves
        
        end_square = None
        if direction == KnightMove.LLEFT_SDOWN.value:
            end_square = square - 8 - 2
        if direction == KnightMove.LLEFT_SUP.value:
            end_square = square + 8 - 2
        if direction == KnightMove.LRIGHT_SDOWN.value:
            end_square = square - 8 + 2
        if direction == KnightMove.LRIGHT_SUP.value:
            end_square = square + 8 + 2
        if direction == KnightMove.SLEFT_LDOWN.value:
            end_square = square - 16 - 1
        if direction == KnightMove.SLEFT_LUP.value:
            end_square = square + 16 - 1
        if direction == KnightMove.SRIGHT_LDOWN.value:
            end_square = square - 16 + 1
        if direction == KnightMove.SRIGHT_LUP.value:
            end_square = square + 16 + 1

        return self._create_move(square, end_square, "")
        
    def _get_underpromotion_move(self, i: int) -> chess.Move | None:
        square = i //  n_underpromotion_moves
        move_nr = i % n_underpromotion_moves
        promotion_piece = move_nr // 3
        direction = move_nr % 3

        end_square = 8 + direction - 1

        return self._create_move(square, end_square, promotion_piece=PROMOTION_MAP[promotion_piece])
    
    def _create_move(self, start_square: int, end_square: int, promotion_piece: str = "") -> chess.Move | None:
        move = None
        try:
            move =  chess.Move.from_uci(f"{chess.square_name(start_square)}{chess.square_name(end_square)}{promotion_piece}")
        except ValueError:
            return None
        except IndexError:
            return None
        return move
    
    def _is_under_promotion(self, promotion_piece: int) -> bool:
        return promotion_piece not in [None, 5]
    
    def _is_queenlike_move(self, from_square: int, to_square: int) -> bool:
        diff = abs(from_square - to_square)
        return diff % 7 == 0 or diff % 9 == 0 or diff % 8 == 0 or ((from_square // 8) == (to_square // 8))
        #      northwest diagonal , northeast diagonal or up/down or left/right

    def get_output_tensor_from_move(self, move: chess.Move) -> torch.Tensor:
        
        promotion_piece = move.promotion
        from_square, to_square = move.from_square, move.to_square

        output = torch.zeros(config.output_shape)
        
        if self._is_queenlike_move(from_square, to_square): # QueenLike Move

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

            output[from_square_idx + direction * 8 + steps - 1] = 1
        
        elif self._is_under_promotion(promotion_piece): # Under Promotion
            direction = to_square - from_square - 8 # -1, 0, 1

            from_square_idx = underpromotion_move_indices[0] + from_square * 9

            output[from_square_idx + promotion_piece * 3 + direction] = 1

        else: # Knight Move
            from_square_idx = knight_move_indices[0] + from_square * 8

            diff = to_square - from_square

            if diff == -10: # LLEFT_SDOWN
                output[from_square_idx] = 1
            elif diff == 6: # LLEFT_SUP
                output[from_square_idx + 1] = 1
            elif diff == -6: # LRIGHT_SDOWN
                output[from_square_idx + 2] = 1
            elif diff == 10: # LLEFT_SUP
                output[from_square_idx + 3] = 1
            elif diff == -17: # SLEFT_LDOWN
                output[from_square_idx + 4] = 1
            elif diff == 15: # SLEFT_LUP
                output[from_square_idx + 5] = 1
            elif diff == -15: # SRIGHT_LDOWN
                output[from_square_idx + 6] = 1
            elif diff == 17: # SRIGHT_LUP
                output[from_square_idx + 7] = 1

        return output
    
if __name__ == "__main__":
    output_processor = AlphaZeroOutputProcessor()

    logger.info(output_processor.get_output_tensor_from_move(chess.Move.from_uci("g7g8")).argmax()) 
    t = torch.zeros(config.output_shape)
    t[3040, :] = 1
    logger.info(output_processor.output_to_position(t, chess.Board("8/6P1/8/8/8/K7/8/k7 w - - 0 1")))