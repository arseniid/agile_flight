#!/usr/bin/python3

import casadi
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


def compute_command_state_based(state, obstacles, rl_policy=None, mpc_dt=None, predicted=None):
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
    commands_list = []
    if rl_policy is not None:
        command = rl_example(state, obstacles, rl_policy)
    else:
        predicted_controls, predicted_all_last = solve_nmpc(
            state, obstacles, dt=mpc_dt, warm_start_predictions=predicted
        )

        for i, cmd_control in enumerate(predicted_controls):
            command = AgileCommand(command_mode)
            command.t = state.t + i * mpc_dt
            command.velocity = cmd_control
            command.yawrate = 0.0

            commands_list.append(command)

    ################################################
    # !!! End of user code !!!
    ################################################

    return (commands_list, predicted_all_last) if commands_list else command


def get_obstacle_absolute_states(state, obstacles, qty=15):
    """ Parses ROS obstacle message into list of absolute obstacle states """
    obstacles_list = []
    for obstacle in obstacles.obstacles[:qty]:
        if obstacle.scale != 0:
            obs_rel_pos = np.array(
                [obstacle.position.x, obstacle.position.y, obstacle.position.z]
            )
            obs_vel = np.array(
                [
                    obstacle.linear_velocity.x,
                    obstacle.linear_velocity.y,
                    obstacle.linear_velocity.z,
                ]
            )
            obstacles_list.append((obs_rel_pos + state.pos, obs_vel, obstacle.scale))
    return obstacles_list


def solve_mpc_state_based(state, obstacles):
    """
    Solves (convex) linear MPC using CVXPY library.
    As obstacle avoidance constraints cannot be convex,
    this solution uses SDR (semi-definite relaxation) of the problem.

    If the problem is detected to be infeasible, the default control velocity [1.0, 0.0, 0.0] is returned.
    """
    obstacles_full_state = get_obstacle_absolute_states(state, obstacles, qty=5)

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
    constraints.extend(
        [
            x[0, :] >= -5.0,
            x[0, :] <= 65.0,
            x[1, :] >= -10.0,
            x[1, :] <= 10.0,
            x[2, :] >= 0.0,
            x[2, :] <= 10.0,
        ]
    )

    # TODO: adjust control constraints
    constraints.extend(
        [
            u[0, :] >= -5.0,
            u[0, :] <= 5.0,  # ~32 seems to be max. for velocity
            u[1, :] >= -5.0,
            u[1, :] <= 5.0,
            u[2, :] >= -5.0,
            u[2, :] <= 5.0,
        ]
    )

    for t in range(T):
        cost += cp.pos(x_goal - x[0, t + 1]) / T  # TODO: add soft cost as distance to obstacle?
        constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t])  # x_t+1 = A * x_t + B * u_t

        # obstacle avoidance
        m_identity_matrix = -np.eye(3)
        for abs_pos, abs_vel, radius in obstacles_full_state:
            safe_radius = radius + 0.5
            obs_rel_pos = cp.reshape(abs_pos + abs_vel * t * dt - x[:3, t], (n, 1))
            obs_rel_pos_T = cp.reshape(abs_pos + abs_vel * t * dt - x[:3, t], (1, n))

            P = cp.Variable((n, n))
            constraints.append(cp.trace(m_identity_matrix @ P) + safe_radius ** 2 <= 0)

            schur = cp.bmat([[P, obs_rel_pos], [obs_rel_pos_T, np.eye(1)]])
            constraints.append(schur >> 0)

    problem = cp.Problem(cp.Minimize(cost), constraints)

    try:
        problem.solve(max_iters=500, verbose=True)
    except cp.error.SolverError as e:
        print(e)
        return np.array([[1.0, 0.0, 0.0]])

    print(f"MPC problem status: {problem.status}")
    print(f"Optimal value at time {state.t} is {problem.value}")
    print(f"(All) optimal states {x.value}")
    print(f"(Current) optimal controls {u.value[:, 0] if u.value is not None else u.value}")

    if problem.status in ["optimal", "optimal_inaccurate"]:
        return np.expand_dims(u.value[:3, 0], axis=0)
    elif problem.status in ["infeasible", "infeasible_inaccurate"]:
        return np.array([[1.0, 0.0, 0.0]])


def solve_nmpc(state, obstacles, dt=0.05, warm_start_predictions=None):
    """
    Solves (non-convex) non-linear MPC using Ipopt solver inside of CasADi wrapper.

    If the problem is detected to be (locally) infeasible, the last (debug) control velocity is returned.
    """
    obstacles_full_state = get_obstacle_absolute_states(state, obstacles, qty=15)

    T = 25
    v_max = 3.0
    a_max = 11.3  # a_max = max_thrust (8.50 N) / mass (0.752 kg) = 11.30 m/s^2
    min_distance = 0.1
    x0 = np.zeros((1, 6))
    x0[0, :3] = state.pos
    x0[0, 3:] = state.vel

    xo = np.array([state_o[0] for state_o in obstacles_full_state])
    vo = np.array([state_o[1] for state_o in obstacles_full_state])
    xg = np.array([65.0, 0, 0])  # larger x to not stop in front of goal

    opti = casadi.Opti()

    # State consists of:
    #   x[t, 0:3] - 3-dimensional position
    #   x[t, 3:6] - 3-dimensional outer control (i.e., velocity)
    #   x[t, 6:9] - 3-dimensional inner control (i.e., acceleration)
    x = opti.variable(T, 9)

    opti.minimize(xg[0] - x[T - 1, 0])

    # initial state
    opti.subject_to(x[0, :6] == x0)

    # dynamic model
    opti.subject_to(x[1:, 0] == x[:-1, 0] + dt * x[:-1, 3] + dt * dt * x[:-1, 6] / 2.0)
    opti.subject_to(x[1:, 1] == x[:-1, 1] + dt * x[:-1, 4] + dt * dt * x[:-1, 7] / 2.0)
    opti.subject_to(x[1:, 2] == x[:-1, 2] + dt * x[:-1, 5] + dt * dt * x[:-1, 8] / 2.0)
    opti.subject_to(x[1:, 3] == x[:-1, 3] + dt * x[:-1, 6])
    opti.subject_to(x[1:, 4] == x[:-1, 4] + dt * x[:-1, 7])
    opti.subject_to(x[1:, 5] == x[:-1, 5] + dt * x[:-1, 8])

    # bounding box
    safe_dist = 0.5
    opti.subject_to(-5.0 + safe_dist < x[:, 0])
    opti.subject_to(x[:, 0] < 65)
    opti.subject_to(-10.0 + safe_dist < x[:, 1])
    opti.subject_to(x[:, 1] < 10.0 - safe_dist)
    opti.subject_to(0.0 < x[:, 2])
    opti.subject_to(x[:, 2] < 10.0 - safe_dist)

    # outer control constraints
    opti.subject_to(-v_max < x[:, 3])
    opti.subject_to(x[:, 3] < v_max)
    opti.subject_to(-v_max < x[:, 4])
    opti.subject_to(x[:, 4] < v_max)
    opti.subject_to(-v_max < x[:, 5])
    opti.subject_to(x[:, 5] < v_max)

    # inner control constraints
    opti.subject_to(-a_max < x[:, 6])
    opti.subject_to(x[:, 6] < a_max)
    opti.subject_to(-a_max < x[:, 7])
    opti.subject_to(x[:, 7] < a_max)
    opti.subject_to(-a_max < x[:, 8])
    opti.subject_to(x[:, 8] < a_max)

    # obstacle avoidance
    for i in range(xo.shape[0]):
        for k in range(T):
            constrain = False
            if i < 5 and k < 10:
                constrain = True
            elif (
                np.linalg.norm(
                    x0[0, :3] + k * dt * x0[0, 3:] - xo[i, None] - k * dt * vo[i, None]
                )
                <= obstacles_full_state[i][2] + min_distance
            ):
                constrain = True
            elif (
                warm_start_predictions is not None
                and k < warm_start_predictions.shape[0]
                and np.linalg.norm(
                    warm_start_predictions[k, :3] - xo[i, None] - k * dt * vo[i, None]
                )
                <= obstacles_full_state[i][2] + min_distance
            ):
                constrain = True
            if constrain:
                opti.subject_to(
                    casadi.sqrt(
                        casadi.sumsqr(x[k, :3] - xo[i, None] - k * dt * vo[i, None])
                    )
                    > obstacles_full_state[i][2] + min_distance
                )

    # dummy warm start
    opti.set_initial(x[:, 0], np.linspace(x0[0, 0], x0[0, 0] + v_max * dt * T, T))
    opti.set_initial(x[:, 1], x0[0, 1])
    opti.set_initial(x[:, 2], x0[0, 2])
    opti.set_initial(x[1:, 3], v_max)
    opti.set_initial(x[1:, 4], 0.0)
    opti.set_initial(x[1:, 5], 0.0)
    if abs(x0[0, 3] - v_max) <= v_max / 2:
        opti.set_initial(x[1:, 6], 0.0)
    elif x0[0, 3] < v_max:
        opti.set_initial(x[1:, 6], a_max)
    else:
        opti.set_initial(x[1:, 6], -a_max)
    opti.set_initial(x[1:, 7], 0.0)
    opti.set_initial(x[1:, 8], 0.0)
    if warm_start_predictions is not None:
        opti.set_initial(x[1:warm_start_predictions.shape[0] + 1, :], warm_start_predictions)

    silent_options = {
        "ipopt.max_iter": 40,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
    }  # print_level: 0-12 (5 by default)
    solver_options = {"ipopt.max_iter": 40, "verbose": False}
    opti.solver("ipopt", silent_options)
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(e)
        return opti.debug.value(x[1:, 3:6]), opti.debug.value(x[1:, :])

    x_optimal = sol.value(x)

    return x_optimal[1:, 3:6], x_optimal[1:, :]
