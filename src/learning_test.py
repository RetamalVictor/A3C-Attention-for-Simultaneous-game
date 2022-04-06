import argparse
import os
import warnings
from multiprocessing import cpu_count
from pathlib import Path
from random import randint
import sys
PATH = "/home/baierh/tu-eind-AGSMCTS/tu-eind-AGSMCTS"
sys.path.append(PATH)

import torch
import torch.multiprocessing as mp
from multiprocessing import Process


from learning.model.agent_model import create_agent_model
from config import cpu, gpu
from learning.monitor import monitor
from learning.model.shared_adam import SharedAdam
from learning.train import train


if __name__ == '__main__':
    SEED = randint(1,10000)
    RANK = 1
    NB_PROCESSES = 16
    print(NB_PROCESSES)
    NB_PLAYERS  = 4
    NB_OPPONENTS= NB_PLAYERS -1
    OPPONENT_CLASSES = ["static", "static", "static"]
    COMBINED_OPPONENT_CLASSES = ",".join(OPPONENT_CLASSES)
    Path(f"../saved_models/{COMBINED_OPPONENT_CLASSES}").mkdir(exist_ok=True, parents=True)
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    USE_CYTHON = True
    NB_STEPS = 800
    SAVE_INTERVAL = 2500
    NB_FILTERS = 4
    NB_CONV_LAYERS = 32 
    LATENT_DIM = 128
    NB_SOFT_ATTENTION_HEADS = 5
    HARD_ATTENTION_RNN_HIDDEN_SIZE = 128
    APPROXIMATE_HARD_ATTENTION = 'store_true'
    EXACT_HARD_ATTENTION = None
    MAX_GRAD_NORM = 0.8
    REWARD_SHAPERS = ["enemy_killed", "mobility", "picking_powerup", "avoiding_illegal_moves"]
    DEVICE = torch.device("cpu")
    CHECK_POINT = None
    INCLUDE_OPPONENT_LOSS = 'store_true'

    MODEL_SPECS = {
        "nb_conv_layers": NB_CONV_LAYERS,
            "nb_filters": NB_FILTERS,
            "latent_dim": LATENT_DIM,
            "nb_soft_attention_heads": NB_SOFT_ATTENTION_HEADS,
            "hard_attention_rnn_hidden_size": HARD_ATTENTION_RNN_HIDDEN_SIZE,
            "approximate_hard_attention": APPROXIMATE_HARD_ATTENTION,
        }

    NB_ACTIONS = 6

    shared_model = create_agent_model(rank=NB_PROCESSES + 1,
                                    seed=SEED,
                                    nb_actions=NB_ACTIONS,
                                    nb_opponents=NB_OPPONENTS,
                                    device=DEVICE,
                                    train=True,
                                    **MODEL_SPECS)

    shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(),
                            lr= 1e-4,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=1e-5)

    optimizer.share_memory()
    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    args = (shared_model,
            OPPONENT_CLASSES,
            SAVE_INTERVAL)
    p1 = mp.Process(target=monitor, args=args)
    p1.start()
    processes.append(p1)
    print("Started training.")
    args = (RANK,
            SEED,
            USE_CYTHON,
            shared_model,
            optimizer,
            counter,
            lock,
            MODEL_SPECS,
            NB_STEPS,
            NB_ACTIONS,
            NB_OPPONENTS,
            OPPONENT_CLASSES,
            REWARD_SHAPERS,
            MAX_GRAD_NORM,
            INCLUDE_OPPONENT_LOSS,
            DEVICE)
    p2 = mp.Process(target=train, args=args)
    p2.start()
    processes.append(p2)
    for p in processes:
        p.join()
