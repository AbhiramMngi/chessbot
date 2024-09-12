import torch 
import chess
from torch import nn
from typing import Callable
from trainer.alpha_zero.input_processor import AlphaZeroInputProcessor
from trainer.alpha_zero.output_processor import AlphaZeroOutputProcessor
from trainer.input_processor import InputProcessor
from trainer.output_processor import OutputProcessor
from trainer.utils import config, TrainingConfig, default_training_config
from opponent_engine.engine import close
from trainer.model_instances import alphazero_model, az_loss_fn, az_model_save
from logger import setup_logger
from trainer.alpha_zero.dataset import AlphaZeroDataset
from torch.utils.data import DataLoader

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.training_config = training_config
    
    def pipeline(self, input_pos: chess.Board) -> tuple[chess.Move, torch.Tensor]:

        input = self.ip.position_to_input(input_pos)
        input = input.to(self.device)
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
    
    def train(self):

        train_dataset = AlphaZeroDataset()
        train_loader = DataLoader(train_dataset, batch_size=self.training_config.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)
        
        for epoch in range(self.training_config.num_epochs):
            for batch_idx, (inputs, outputs) in enumerate(train_loader):
                optimizer.zero_grad()
                inputs = inputs.to(self.device)
                outputs_pred = self.model(inputs)
                # logger.info(f"Predictions: {outputs_pred[0].shape}, Targets: {outputs[0].sha}")
                loss = self.loss(outputs_pred[0], outputs[0], outputs_pred[1], outputs[1])
                loss.backward()
                optimizer.step()

            if epoch % self.training_config.save_interval == 0:
                self.save()
                logger.info(f"Epoch {epoch + 1} - Loss: {loss.item()}")
    


if __name__ == "__main__":
    model_controller = ModelController(
        model_type="AlphaZero",
        input_processor=AlphaZeroInputProcessor(),
        output_processor=AlphaZeroOutputProcessor(),
        training_config=default_training_config,
    )

    model_controller.train()
    close()
