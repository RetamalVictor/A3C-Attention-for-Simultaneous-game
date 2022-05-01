import pommerman
from pommerman import agents
from pathlib import Path
import sys
PATH = "/home/baierh/tu-eind-AGSMCTS/tu-eind-AGSMCTS/src/"
sys.path.append(PATH)

from config import cpu, gpu
from pommerman_env.agents.rl_agent import  RLAgent
from learning.model.agent_model import create_agent_model
import torch


def main():
    #Prepare the agents

    #Need the model for this
    device = 'cpu'

    model_spec = {
            "nb_conv_layers": 4,
            "nb_filters": 32,
            "latent_dim": 128,
            "nb_soft_attention_heads": 5,
            "hard_attention_rnn_hidden_size": 128,
            "approximate_hard_attention": True,
        }
    nb_actions = 6
    nb_opponents = 3
    shared_model = create_agent_model(rank=1,
                                        seed=1,
                                        nb_actions=nb_actions,
                                        nb_opponents=nb_opponents,
                                        device=device,
                                        train=False,
                                        **model_spec)

    shared_model.load_state_dict(torch.load(r'/home/baierh/tu-eind-AGSMCTS/tu-eind-AGSMCTS/src/saved_models/simple,simple,simple_exp2/agent_model_5.pt'))


    my_agent = RLAgent(0,shared_model)
    agent_list = [
            my_agent,
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
            ]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)

# Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
           #env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()

if __name__ == "__main__":
    main()