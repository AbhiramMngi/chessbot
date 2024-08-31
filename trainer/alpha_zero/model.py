import os
from torchviz import make_dot
from torch import nn
from trainer.utils import config, ModelConfig
import torch
from torchsummary import summary
from logger import setup_logger

logger = setup_logger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ResidualBlock, self).__init__()
        self.conv_layer = build_convolutional_layer(config)
    
    def forward(self, x):
        return x + self.conv_layer(x)

def build_convolutional_layer(config: ModelConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels=config.n_filters, out_channels=config.n_filters, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(config.n_filters),
        nn.ReLU()
    )

def build_policy_block(config: ModelConfig) -> nn.Sequential:
    return nn.Sequential(
        build_convolutional_layer(config),
        nn.Flatten(),
        nn.Linear(in_features=config.n_filters * config.n * config.n,out_features=config.output_shape[0]),
        nn.Softmax(dim=-1)
    )

def build_value_block(config: ModelConfig) -> nn.Sequential:
    return nn.Sequential(
        build_convolutional_layer(config),
        nn.Flatten(),
        nn.Linear(in_features=config.n_filters * config.n * config.n, out_features=config.output_shape[1]),
        nn.Tanh()
    )


class AlphaZero(nn.Module):
    def __init__(self, config: ModelConfig):
        super(AlphaZero, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=config.input_shape[0], out_channels=config.n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.n_filters),
            nn.ReLU()
        )
        self.residual_layers = [ResidualBlock(config) for i in range(config.resid_blocks)]
        self.policy = build_policy_block(config)
        self.value = build_value_block(config)
    
    def forward(self, x):
        x = self.conv_layer(x)
        for layer in self.residual_layers:
            x = layer(x)
        return self.policy(x), self.value(x)
    

mse_loss = nn.MSELoss()
cce_loss = nn.CrossEntropyLoss()

def build_model(path: str):
    
    model = AlphaZero(config)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    
    return model, total_loss_func

# TODO: Loss function check
def total_loss_func(predicted_policy, target_policy, predicted_value, target_value):
    mse = mse_loss(predicted_value, target_value)
    cce = cce_loss(predicted_policy, target_policy)
    return (mse + cce) * 0.5
    

path = "../../alphazero_model.pth"

if __name__ == "__main__":
    model, _ = build_model(path)
    input = torch.randn(1, *config.input_shape)
    output = model(input)
    dot = make_dot(output, params = dict(model.named_parameters()))
    dot.format = "png"
    dot.render("model_architecture")
    
    summary(model, input_size=config.input_shape)

    if not os.path.exists(path): 
        torch.save(model.state_dict(), path)
