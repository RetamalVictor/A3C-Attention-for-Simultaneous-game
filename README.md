# A3C-Attention-for-Simultaneous-games

This is the implementation of a version of Advantage Asynchronous Actor Critic (A3C) using Attention as a way to score the opponent importance during the game. The enviroment used to run the experiments is a slightly modified environment from a [Cythonized version](https://github.com/tambetm/pommerman-baselines/tree/master/cython_env) of [Pommerman Env](https://github.com/MultiAgentLearning/playground). 

## Getting started

### installation 
To run experiments and train new agents, at least Pommerman environment needs to be installed. 
The installation guidelines are available at [Installation Guidelines](https://github.com/MultiAgentLearning/playground/tree/master/docs) 

To improve performance the Cython environment can be used. 
To install it, please refer to  [Cythonized version installation](https://github.com/tambetm/pommerman-baselines/tree/master/cython_env)

A set of requirements is specified in the requirement.txt file.
```
$ pip install -r requirement.txt
```
## Learning
### Training an agent
To train an agent, run the file ```learn.py ```. The default params will run the ```baseline A3C model```, in a FFA 4 player match, with a training duration of 800 steps. 
```
$ python3 -u learn.py --experiment-name <exp-name> --save-path <save_path>
```
More about default parameters in the docs.

To launch training with the ```Attention version``` with Attention parameter specifications.
```
$ python3 -u learn.py --nb-players 4 --nb-soft-attention-heads 5 --hard-attention-rnn-hidden-size 128 --experiment-name <exp-name> --save-path <save_path>
```
As an approximation to run ~500.000 episodes, ```A3C versions``` takes 100h and ```Attention Version``` takes 200h in Snellius Cluster with 1/3 tcn node.
The trained agent checkpoints will be stored in a directory named ```/saved-models/<opponent_classes>-<experiment_name>```

### Visualization
To visualize the performance of your agent while playing the game. A visualization script is include in the ```pommerman_env``` directory.
To run the visualization
```
$ python3 visualization.py --version "Att" --model-path <model-path>
```
For custom training parameters, you will need to modify the model parameters used to load the checkpoints.


## Planning
To tune a method (e.g., vanilla SMMCTS), you can run the following command:
```
$ python3  planning/tune_smmmcts.py
```
For instance, if I want to choose the best parameter for vanilla SMMCTS where the trained model is not used, you should add the following options to the launch command
```
--no-pw --search-opponent-actions --policy-estimation uniform --opponent-classes "simple, simple, simple" --nb-games 200 --nb-plays 1 --mcts-iterations 500
```
Note that you have to modify planning/tune_smmcts.py to choose the parameter of interest. Tuning those parameters might
take weeks for the neural network model.
Finally, to compute the win rate, tie rate and lose rate, you can run the following command with the desired parameter:
```
$ python3 planning/evaluate_smmmcts.py
```
which will compute the above-mentioned metrics. This will take days for the neural network model.
