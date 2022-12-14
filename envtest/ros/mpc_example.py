import torch
import numpy as np

from utils import AgileCommand, get_goal_direction, transform_obstacles

# learned (n)mpc models
from learn_mpc import mpc_nn


class PathModelAlignment:
    def __init__(self, hyperdata_path="learned_mpc/hyperdata.txt"):
        self.hyperdata_path = hyperdata_path
        self.alignment = dict()

        self._read_hyper()

    def _read_hyper(self):
        with open(self.hyperdata_path, "r") as hyperfile:
            class_name = None
            for line in hyperfile.read().splitlines():
                if "filename" in line:
                    filename = line.split()[-1]
                if "model_name" in line:
                    class_name = line.split("'")[-2].split(".")[-1]
                if "Identity()" in line:
                    use_batch_norm = False
                elif "BatchNorm1d" in line:
                    use_batch_norm = True
                if "(2)" in line and "Linear" not in line:
                    non_linearity_name = line.split()[-1].split("(")[0]

                if "===" in line and class_name:  # next class
                    self._append_model(class_name, non_linearity_name, filename, use_batch_norm)
                    class_name = None
            else:  # end of file
                self._append_model(class_name, non_linearity_name, filename, use_batch_norm)
        print(self.alignment)

    def _append_model(self, *args):
        cls = getattr(mpc_nn, args[0])
        non_linearity_fn = getattr(torch.nn, args[1])
        self.alignment[args[2]] = cls(use_batch_normalization=args[3], activation_fn=non_linearity_fn)


def mpc_example(state, obstacles, learned_mpc=None):
    model = learned_mpc

    obstacles_arr = transform_obstacles(state=state, obstacles=obstacles, absolute=False, as_np=True)
    if isinstance(model, mpc_nn.LearnedMPCShortControlFirstDeepObstaclesOnly):
        obs = obstacles_arr.astype(np.single)
    else:
        goal_direction = get_goal_direction(state=state)
        obs = np.concatenate((obstacles_arr, goal_direction), axis=None).astype(np.single)
    obs_batched = torch.from_numpy(np.expand_dims(obs, axis=0))

    action = model(obs_batched)

    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    if "Control" in model.__class__.__name__:
        command.velocity = action[0, :]
    else:
        command.velocity = action.reshape((11, 9))[0, 3:6]
    command.yawrate = 0
    return command


def load_learned_mpc(mpc_path):
    path_to_model = PathModelAlignment()

    model = path_to_model.alignment[mpc_path.split("/")[-1]]
    model.load_state_dict(torch.load(mpc_path))
    model.eval()
    return model
