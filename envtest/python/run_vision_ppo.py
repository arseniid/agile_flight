#!/usr/bin/env python3
import argparse
import math
#
import os
import subprocess


import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy
from sb3_contrib.ppo_recurrent import MlpLstmPolicy

from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.common.ppo_recurrent import RecurrentPPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    parser.add_argument("--recurrent", type=int, default=0, help="Use recurrent LSTM-based policy or not")
    parser.add_argument("--pretrained", type=int, default=0, help="Pre-trained PPO trial number for transfer learning")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )

    if not args.train:
        cfg["simulation"]["num_envs"] = 1

    # create training environment
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"

    # create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    cfg["simulation"]["num_envs"] = old_num_envs

    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    #
    if args.train:
        if args.recurrent:
            model_type = RecurrentPPO
            policy_type = "MlpLstmPolicy"
        else:
            model_type = PPO
            policy_type = "MlpPolicy"

        if args.pretrained:
            model = model_type.load(
                rsg_root + f"/../ros/rl_policy/PPO_{args.pretrained}/ppo_{args.pretrained}_model",
                tensorboard_log=log_dir,
                env=train_env,
                eval_env=eval_env,
                env_cfg=cfg,
                verbose=1,
            )
            print("loaded pre-trained model")
        else:
            model = model_type(
                tensorboard_log=log_dir,
                policy=policy_type,
                policy_kwargs=dict(
                    activation_fn=torch.nn.ReLU,
                    net_arch=[dict(pi=[256, 256], vf=[512, 512])],
                    log_std_init=-0.5,
                ),
                learning_rate=5e-5,
                env=train_env,
                eval_env=eval_env,
                use_tanh_act=True,
                gae_lambda=0.95,
                gamma=0.99,
                n_steps=500,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                batch_size=50000,
                clip_range=0.2,
                use_sde=False,  # don't use (gSDE), doesn't work
                env_cfg=cfg,
                verbose=1,
            )

        #
        model.learn(total_timesteps=int(37.5 * 1e7), log_interval=(10, 50))
        model.save(rsg_root + f"/saved/{'Recurrent' if args.recurrent else ''}PPO_{args.trial}/ppo_{args.trial}_model")
    else:
        if args.render:
            proc = subprocess.Popen(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64")
        #
        weight = (
            rsg_root
            + f"/../ros/rl_policy/{'Recurrent' if args.recurrent else ''}PPO_{args.trial}/Policy/iter_{args.iter:05d}.pth"
        )
        env_rms = (
            rsg_root
            + f"/../ros/rl_policy/{'Recurrent' if args.recurrent else ''}PPO_{args.trial}/RMS/iter_{args.iter:05d}.npz"
        )

        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        # Create policy object
        policy = (
            MlpLstmPolicy(**saved_variables["data"])
            if args.recurrent
            else MlpPolicy(**saved_variables["data"])
        )
        #
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        # Load weights
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)
        #
        eval_env.load_rms(env_rms)
        test_policy(eval_env, policy, render=args.render, recurrent=args.recurrent)

        if args.render:
            proc.terminate()


if __name__ == "__main__":
    main()
