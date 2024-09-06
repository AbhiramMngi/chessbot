from trainer.model_controller import ModelController
from trainer.alpha_zero.input_processor import AlphaZeroInputProcessor
from trainer.alpha_zero.output_processor import AlphaZeroOutputProcessor
from trainer.utils import default_training_config, n_simulations
from self_play.MCTS import MCTS

class Agent(object):
    def __init__(self):
        self.model_controller = ModelController(
            model_type="AlphaZero",
            input_processor=AlphaZeroInputProcessor(),
            output_processor=AlphaZeroOutputProcessor(),
            training_config=default_training_config,
        )
        
        self.mcts = MCTS(predict=self.model_controller.pipeline)
    
    def run_simulations(self):
        self.mcts.run_simulations(n_simulations=n_simulations)

agent = Agent()

agent.run_simulations()


    



