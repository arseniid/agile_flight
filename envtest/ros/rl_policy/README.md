# Trained Deep RL policies
The best trained deep reinforcement learning policies for the DodgeDrone-Challenge 2022 will be kept here.

Each policy contains 'Policy/' and 'RMS/' folders with the trained network, as well as used `config.yaml` and Tensorboard logs.
Additionally, some policies also store the full model as `.zip` archives (for example, for further training).

## Best policies (ranking)
Here, no evaluation will be given, but only a ranking. Keep in mind that this ranking is not absolute, and some policies might behave better on different
environments.
It is also worth to mention that the models got *much* better for the second evaluation.

PPO_1: This policy was already trained by the maintainers, but is rather sub-optimal.

> **_NOTE:_** Other trained policies (which have poorer performance) will anyway stay on the lab computer and *might be* available upon request.

### After first evaluation

1. PPO_30: best time on medium environments, still some collisions on hard
2. PPO_26: quite good on medium, still 2-5 (sometimes 7) collisions on hard
3. PPO_23: 3-5 (sometimes 9) collisions on hard
4. PPO_29: a lot of very occasional collision

### After second evaluation

1. PPO_39: best so far, with no more than 3 collisions on hard (0 quite often)
2. PPO_37: quite reproducible with mostly 2 collisions (0 sometimes), but cannot 'solve' medium environments!
3. PPO_43 (iter=5k | 7.5k): can have up to 5 collisions on hard | worse than 5.000 iterations
4. PPO_35: base for all of the above
5. PPO_41: should be re-evaluated
