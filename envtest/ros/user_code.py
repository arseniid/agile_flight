#!/usr/bin/python3

import cvxpy as cp
import numpy as np

from pickle import NONE
from utils import AgileCommandMode, AgileCommand
from rl_example import rl_example


def compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command vision-based!")
    # print(state)
    # print("Image shape: ", img.shape)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    ################################################
    # !!! End of user code !!!
    ################################################

    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]
 
    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    # If you want to test your RL policy
    if rl_policy is not None:
        command = rl_example(state, obstacles, rl_policy)
    else:
        command.velocity = solve_mpc_state_based(state, obstacles)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command


def outer(a, b):
    """ Inspired by https://github.com/cvxpy/cvxpy/issues/1724 """
    a = cp.Expression.cast_to_const(a)  # if a is an Expression, return it unchanged.
    assert a.ndim == 1
    b = cp.Expression.cast_to_const(b)
    assert b.ndim == 1
    a = cp.reshape(a, (a.size, 1))
    b = cp.reshape(b, (1, b.size))
    expr = a @ b
    return expr


def solve_mpc_state_based(state, obstacles):
    """
    Solves (convex) linear MPC using CVXPY library.
    As obstacle avoidance constraints cannot be convex,
    this solution uses SDR (semi-definite relaxation) of the problem.

    If the problem is detected to be infeasible, the default control velocity [1.0, 0.0, 0.0] is returned.
    """
    obstacles_full_state = []
    for obstacle in obstacles.obstacles:
        if obstacle.scale != 0:
            obs_rel_pos = np.array([obstacle.position.x, obstacle.position.y, obstacle.position.z])
            obs_vel = np.array([obstacle.linear_velocity.x, obstacle.linear_velocity.y, obstacle.linear_velocity.z])
            obstacles_full_state.append((obs_rel_pos + state.pos, obs_vel, obstacle.scale))

    n = 3
    m = 3
    T = 50
    dt = 0.01

    A = np.eye(n)
    B = np.eye(m) * dt

    # start and end position
    x_0 = state.pos
    x_goal = 60

    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T))

    cost = 0
    constraints = []

    # start and goal position constraint
    constraints.append(x[:3, 0] == x_0)

    # bounding box
    constraints.extend([
        x[0, :] >= -5.0,
        x[0, :] <= 65.0,
        x[1, :] >= -10.0,
        x[1, :] <= 10.0,
        x[2, :] >= 0.0,
        x[2, :] <= 10.0,
    ])

    # TODO: adjust control constraints
    constraints.extend([
        u[0, :] >= -5.0,
        u[0, :] <= 5.0,  # 50 is possible for velocity
        u[1, :] >= -5.0,
        u[1, :] <= 5.0,
        u[2, :] >= -5.0,
        u[2, :] <= 5.0,
    ])

    cost += cp.pos(x_goal - x[0, T])
    for t in range(T):
        constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t])  # x_t+1 = A * x_t + B * u_t

        # obstacle avoidance
        for abs_pos, abs_vel, radius in obstacles_full_state:
            safe_radius = radius + 0.5

            obs_rel_pos = cp.reshape(abs_pos + abs_vel * t * dt - x[:3, t], (n, 1))
            obs_rel_pos_T = cp.reshape(abs_pos + abs_vel * t * dt - x[:3, t], (1, n))
            m_identity_matrix = -np.eye(3)
            P = outer(obs_rel_pos, obs_rel_pos)

            constraints.append(
                cp.trace(m_identity_matrix @ P) + safe_radius ** 2 <= 0
            )

            schur = cp.bmat([[P, obs_rel_pos], [obs_rel_pos_T, np.eye(1)]])
            constraints.append(
                schur >> 0
            )

    problem = cp.Problem(cp.Minimize(cost), constraints)

    try:
        problem.solve(max_iters=500, verbose=True)
    except cp.error.SolverError as e:
        print(e)
        return np.array([1.0, 0.0, 0.0])

    print(f"MPC problem status: {problem.status}")
    print(f"Optimal value at time {state.t} is {problem.value}")
    print(f"(All) optimal states {x.value}")
    print(f"(Current) optimal controls {u.value[:, 0] if u.value is not None else u.value}")

    if problem.status in ["optimal", "optimal_inaccurate"]:
        return u.value[:3, 0]
    elif problem.status in ["infeasible", "infeasible_inaccurate"]:
        return np.array([1.0, 0.0, 0.0])
