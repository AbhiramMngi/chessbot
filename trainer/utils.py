from enum import Enum
class ModelConfig:
    def __init__(self, filters, resid_blocks, input_shape, output_shape):
        self.n_filters = filters
        self.resid_blocks = resid_blocks
        self.n = 8
        self.input_shape = input_shape
        self.output_shape = output_shape


config = ModelConfig(filters=256, resid_blocks=5, input_shape=(20, 8, 8), output_shape=(4672, 1))

n_past_moves = 1 # no of past moves to include in the network input 1 or 8

n_possible_outputs = 73
n = 8


n_queenlike_moves = 56
n_knight_moves = 8
n_underpromotion_moves = 9

queenlike_move_indices = [0, 56 * 64 - 1]
knight_move_indices = [56 * 64, 64 * 64 - 1]
underpromotion_move_indices = [64 * 64, 73 * 64 - 1]

class QueenLikeDirection(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NORTH_EAST = 4
    NORTH_WEST = 5
    SOUTH_EAST = 6
    SOUTH_WEST = 7


class KnightMove(Enum):
    SRIGHT_LUP = 0 # Short right, Long up
    SRIGHT_LDOWN = 1 # Short right, Long down
    SLEFT_LUP = 2 # Short left, Long up
    SLEFT_LDOWN = 3 # Short left, Long down
    LRIGHT_SUP = 4 # Long right, Short up
    LRIGHT_SDOWN = 5 # Long right, Short down
    LLEFT_SUP = 6 # Long left, Short up
    LLEFT_SDOWN = 7 # Long left, Short down

# class UnderPromotion(Enum):
#     KNIGHT = 0
#     BISHOP = 1
#     ROOK = 2

PROMOTION_MAP = {
    0: "b",
    1: "n",
    2: "r"
}

class TrainingConfig:
    def __init__(
        self,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

default_training_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=1,
)

c = 1.4 # exploration factor 

mcts_depth_limit = 200 # mcts search depth limit

n_simulations = 100 # number of simulations