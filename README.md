# Chess Bot

A chess engine based on the [Alphazero Paper](https://arxiv.org/pdf/1712.01815).

## Description

This is an implementation of the alphazero algorithm that uses MCTS Search to find optimal moves and generate training set for the Neural Network. 

## Model Architecture

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 256, 8, 8]          46,336
       BatchNorm2d-2            [-1, 256, 8, 8]             512
              ReLU-3            [-1, 256, 8, 8]               0
            Conv2d-4            [-1, 256, 8, 8]         590,080
       BatchNorm2d-5            [-1, 256, 8, 8]             512
              ReLU-6            [-1, 256, 8, 8]               0
           Flatten-7                [-1, 16384]               0
            Linear-8                 [-1, 4672]      76,550,720
           Softmax-9                 [-1, 4672]               0
           Conv2d-10            [-1, 256, 8, 8]         590,080
      BatchNorm2d-11            [-1, 256, 8, 8]             512
             ReLU-12            [-1, 256, 8, 8]               0
          Flatten-13                [-1, 16384]               0
           Linear-14                    [-1, 1]          16,385
             Tanh-15                    [-1, 1]               0
================================================================
Total params: 77,795,137
Trainable params: 77,795,137
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.45
Params size (MB): 296.76
Estimated Total Size (MB): 298.22
----------------------------------------------------------------
```

The model replicates the design of the Alpha Zero model by google. See [Paper](https://arxiv.org/pdf/1712.01815).

The model outputs 2 heads, 

* **Value** - The evaluation of the given position. Uses the **Mean Squared Error** Loss.
* **Next Move** - A softmax vector containing all the moves with the highest probable move being the best move in the position. Uses the **Categorical cross-entropy** Loss.

The losses are added in equal ratios to get the final Loss function.


The system also uses Monte Carlo Tree Search (MCTS), which is used to generate training data for the neural network by self play, change the [n_simulations]() parameter in the trainer/utils.py file to change the number of games played in self play.

Due to lack of compute, MCTS is only able to generate a few thousand positions for training, to increase the model's effectiveness, the model is pretrained on previous positions with evaluations given by lc0 from Lichess Evaluations Database.



## How to Use

The trained model is not hosted on github because of file size limitations (298MB).

However, You can load and train your own model by doing the following:

After you've cloned the model, run the following command
``` 
$ python -m trainer.alpha_zero.model
```

to create an instance of the model.

Then, you can train the model by downloading the lichess evaluation database, download [here](https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz). This is a huge database containing over 37M positions.


To train the model, run the following command

```
$ python -m trainer.model_controller
```

and to generate games, run the following command:

```
$ python -m self_play.game_env.agent
```

then copy the games available in the games/ directory and paste the pgn in [here](https://lichess.org/paste) to view the game.


