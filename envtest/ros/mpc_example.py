import torch
import numpy as np

from utils import AgileCommand

# learned (n)mpc models
from learn_mpc.mpc_nn import MPCLearnedControl, MPCLearnedControlSmall



def mpc_example(state, obstacles, learned_mpc=None):
    model = learned_mpc

    # Convert obstacles to vector observation
    obs_vec = []
    for obstacle in obstacles.obstacles:
        obs_vec.append(obstacle.position.x + state.pos[0])
        obs_vec.append(obstacle.position.y + state.pos[1])
        obs_vec.append(obstacle.position.z + state.pos[2])
        obs_vec.append(obstacle.linear_velocity.x)
        obs_vec.append(obstacle.linear_velocity.y)
        obs_vec.append(obstacle.linear_velocity.z)
        obs_vec.append(obstacle.scale)
    obs_vec = np.array(obs_vec)

    obs = np.concatenate([obs_vec, state.pos, state.vel], axis=0).astype(np.single)
    obs_batched = torch.from_numpy(np.expand_dims(obs, axis=0))

    action = model(obs_batched)

    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = action[0, :3]
    command.yawrate = 0
    return command


def load_learned_mpc(mpc_path):
    model = MPCLearnedControl(use_batch_normalization=True)
    model.load_state_dict(torch.load(mpc_path))
    model.eval()
    return model
