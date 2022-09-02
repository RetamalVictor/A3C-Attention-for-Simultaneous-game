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
