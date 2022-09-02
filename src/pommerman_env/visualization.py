import pommerman
from pommerman import agents
from pathlib import Path
import sys

PATH = r"C:\Users\victo\Desktop\VU master\pommerman\tu-eind-AGSMCTS\src"
sys.path.append(PATH)

# from config import cpu, gpu
from learning.pommerman_env_utils import create_env
from learning.model.shared_adam import SharedAdam
from pommerman_env.agents.rl_agent import RLAgent
from learning.model.agent_model import create_agent_model
import torch
import torch.nn.functional as F

<<<<<<< Updated upstream

def main():
    # Prepare the agents
=======
parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default='Att', choices=['Att', 'A3C'])
parser.add_argument("--model-path", type=str, default='~/A3C-Attention-for-Simultaneous-game/src/saved_models/simple,simple,simple_Attention_version/agent_model_Att_version.pt')
>>>>>>> Stashed changes

    # Need the model for this
    device = "cpu"

    # model_spec = {
    #         "nb_conv_layers": 4,
    #         "nb_filters": 32,
    #         "latent_dim": 128,
    #         "nb_soft_attention_heads": 5,
    #         "hard_attention_rnn_hidden_size": 128,
    #         "approximate_hard_attention": True,
    #     }

    model_spec = {
        "nb_conv_layers": 4,
        "nb_filters": 32,
        "latent_dim": 128,
        "nb_soft_attention_heads": None,
        "hard_attention_rnn_hidden_size": None,
        "approximate_hard_attention": True,
    }
    nb_actions = 6
    nb_opponents = 3
    shared_model = create_agent_model(
        rank=1 + 1,
        seed=1,
        nb_actions=nb_actions,
        nb_opponents=nb_opponents,
        device=device,
        train=True,
        **model_spec,
    )
    optimizer = SharedAdam(
        shared_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )

    # checkpoint_dict = torch.load(r'C:\Users\victo\Desktop\VU master\pommerman\tu-eind-AGSMCTS\src\saved_models\saved_models\simple,simple,simple\agent_model_2697.pt')
    checkpoint_dict = torch.load(
        r"C:\Users\victo\Desktop\VU master\pommerman\tu-eind-AGSMCTS\src\saved_models\saved_models\simple,simple,simple_A3C\agent_model_1124.pt"
    )
    shared_model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    shared_model.eval()
    agents, env = create_env(
        1,
        1,
        False,
        model_spec,
        nb_actions,
        nb_opponents,
        ["simple", "simple", "simple"],
        device,
        train=True,
    )
    agent_model = agents[0].agent_model
    agent_model.load_state_dict(shared_model.state_dict())
    agent = agents[0]
    # my_agent = RLAgent(0,shared_model)
    # agent_list = [
    #         my_agent,
    #         agents.SimpleAgent(),
    #         agents.SimpleAgent(),
    #         agents.SimpleAgent()
    #         ]

    # env = pommerman.make('PommeFFACompetition-v0', agents)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            obs = env.get_features(state).to(device)
            (
                agent_policy,
                agent_value,
                opponent_log_prob,
                opponent_influence,
            ) = agent.estimate(obs)
            agent_prob = F.softmax(agent_policy, dim=-1)
            agent_action = agent_prob.multinomial(num_samples=1).detach()
            opponent_actions = env.act(state)
            agent_action = agent_action.item()
            actions = [agent_action, *opponent_actions]
            state, rewards, done = env.step(actions)
            env.render()
<<<<<<< Updated upstream
        print("Episode {} finished".format(i_episode))
=======
        print(f"Episode {i_episode} finished with reward {rewards}")
>>>>>>> Stashed changes
    env.close()


if __name__ == "__main__":
<<<<<<< Updated upstream
    main()
=======
    args = parser.parse_args()
    version = args.version
    model_path = args.model_path

    main(version, model_path)
>>>>>>> Stashed changes
