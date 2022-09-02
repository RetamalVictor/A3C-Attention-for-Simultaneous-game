#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Obada Aljabasini
# Last modification Date: 03/08/2022
#  - Victor Retamal --  Added documentation, fixed imports and saving paths
# version ='1.0'
# ---------------------------------------------------------------------------
""" Main Learning Script """
# ---------------------------------------------------------------------------
# # heavily inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/main.py
import argparse
import os
import warnings

from multiprocessing import cpu_count
from pathlib import Path
from random import randint
import sys

PATH = r"C:\Users\victo\Desktop\VU master\pommerman\tu-eind-AGSMCTS\src"
sys.path.append(PATH)

import torch
import torch.multiprocessing as mp

from learning.model.agent_model import create_agent_model
from config import cpu, gpu
from learning.monitor import monitor
from learning.model.shared_adam import SharedAdam
from learning.train import train

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

cpu_count = 8 #int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=randint(1, 100000))
parser.add_argument("--nb-processes", type=int, default=cpu_count)
parser.add_argument("--nb-players", type=int, default=4, choices=[2, 4])
ss = "simple, simple, simple"
parser.add_argument(
    "--opponent-classes",
    type=lambda s: [str(item).strip().lower() for item in s.split(",")],
    default=ss,
)
parser.add_argument("--nb-steps", type=int, default=32)
parser.add_argument("--save-interval", type=int, default=2460)
parser.add_argument("--nb-conv-layers", type=int, default=4)
parser.add_argument("--nb-filters", type=int, default=32)
parser.add_argument("--latent-dim", type=int, default=128)
parser.add_argument("--nb-soft-attention-heads", type=int, default=None)
parser.add_argument("--hard-attention-rnn-hidden-size", type=int, default=None)
parser.add_argument(
    "--approximate-hard-attention",
    dest="approximate_hard_attention",
    action="store_true",
)
parser.add_argument(
    "--exact-hard-attention", dest="approximate_hard_attention", action="store_false"
)
parser.add_argument("--max-grad-norm", type=float, default=0.8)
d = "enemy_killed, mobility, picking_powerup, planting_bomb_near_enemy, catching_enemy, planting_bomb_near_wall, avoiding_illegal_moves"
# d = "enemy_killed, mobility, picking_powerup, avoiding_illegal_moves"
parser.add_argument(
    "--reward-shapers",
    type=lambda s: [str(item).strip().lower() for item in s.split(",")],
    default=d,
)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--check-point", type=str, default=None)
parser.add_argument("--save-path",  type=str, default=None)
parser.add_argument("--experiment-name",  type=str, default=None)
parser.add_argument(
    "--include-opponent-loss", dest="include_opponent_loss", action="store_true"
)
parser.add_argument(
    "--ignore-opponent-loss", dest="include_opponent_loss", action="store_false"
)
parser.set_defaults(approximate_hard_attention=True, include_opponent_loss=True)

if __name__ == "__main__":

    # SAVING_PATH = "/home/hbaier/Pommerman-project/tu-eind-AGSMCTS/output/bug_fixed/results/training-4.txt"
    args = parser.parse_args()
    reward_shapers = args.reward_shapers
    opponent_classes = args.opponent_classes
    combined_opponent_classes = ",".join(opponent_classes)
    experiment_name = args.experiment_name
    Path(f"../saved_models/{combined_opponent_classes}_{experiment_name}").mkdir(
        exist_ok=True, parents=True
    )
    os.environ["OMP_NUM_THREADS"] = "1"
    mp.set_start_method("spawn")
    device = gpu if args.device.lower() == "gpu" else cpu
    seed = args.seed
    # use_cython = args.nb_players == 4
    use_cython = False
    max_grad_norm = args.max_grad_norm
    nb_processes = args.nb_processes
    nb_opponents = args.nb_players - 1
    nb_steps = args.nb_steps
    save_interval = args.save_interval
    save_path = args.save_path
    include_opponent_loss = args.include_opponent_loss
    model_spec = {
        "nb_conv_layers": args.nb_conv_layers,
        "nb_filters": args.nb_filters,
        "latent_dim": args.latent_dim,
        "nb_soft_attention_heads": args.nb_soft_attention_heads,
        "hard_attention_rnn_hidden_size": args.hard_attention_rnn_hidden_size,
        "approximate_hard_attention": args.approximate_hard_attention,
    }
    nb_actions = 6

    # Creating the model all agent will share
    shared_model = create_agent_model(
        rank=nb_processes + 1,
        seed=seed,
        nb_actions=nb_actions,
        nb_opponents=nb_opponents,
        device=device,
        train=True,
        **model_spec,
    )
    # Initializing optimizer
    optimizer = SharedAdam(
        shared_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )

    # Reloading training if checkpoint provided
    if args.check_point is not None:
        checkpoint_dict = torch.load(args.check_point)
        shared_model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        shared_model.train()

    # Sharing parameters for multiprocessing
    shared_model.share_memory()
    optimizer.share_memory()
    processes = []
    counter = mp.Value("i", 0)
    lock = mp.Lock()

    # Initializing monitoring process
    args = (shared_model, optimizer, opponent_classes, save_interval, experiment_name)
    p = mp.Process(target=monitor, args=args)
    p.start()
    processes.append(p)

    # Initializing Training processes
    for rank in range(nb_processes - 1):
        args = (
            rank,
            seed,
            use_cython,
            shared_model,
            optimizer,
            counter,
            lock,
            model_spec,
            nb_steps,
            nb_actions,
            nb_opponents,
            opponent_classes,
            reward_shapers,
            max_grad_norm,
            include_opponent_loss,
            device,
            save_path,
        )
        p = mp.Process(target=train, args=args)
        p.start()
        processes.append(p)
    print("Started training.")
    print(f"CPU_count {cpu_count}")
    print(f"Number of processes: {nb_processes}, initialized: {len(processes)}")
    print(f"Cython: {use_cython}")
    for p in processes:
        p.join()
