import cvxpy as cp
import numpy as np

from utils import main_test


def solve_mpc(state, obstacles):
    obstacles_full_state = []
    for obstacle in obstacles.obstacles:
        if obstacle.scale != 0:
            # TODO: everything after was 3-dimensional
            obs_rel_pos = np.array([obstacle.position.x, obstacle.position.y])
            obs_vel = np.array([obstacle.linear_velocity.x, obstacle.linear_velocity.y])
            obstacles_full_state.append((obs_rel_pos, obs_vel, obstacle.scale))  # TODO: 'state.pos' was added to obs_rel_pos 

    #print(obstacles_full_state)
    #print(state)

    n = 2  # TODO: was 3
    m = 2  # TODO: was 3
    T = 50

    A = np.eye(n)
    B = np.eye(m) * 0.1  # TODO: was 0.01 here

    # start and end state
    x_0 = state.pos
    x_goal = 10

    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T))

    cost = 0
    constraints = []

    # start and goal position constraint
    constraints.append(x[:3, 0] == x_0)
    #constraints.append(x[0, T] >= x_goal)  # TODO: questionable

    # bounding box
    safe_dist = 0.1  # TODO: was 0.5
    constraints.extend([
        x[0, :] >= -5.0,
        x[0, :] <= 15.0,
        x[1, :] >= -2.0 + safe_dist,
        x[1, :] <= 2.0 - safe_dist,
    ])

    constraints.extend([
        u[0, :] >= -5.0,
        u[0, :] <= 5.0,
        u[1, :] >= -5.0,
        u[1, :] <= 5.0,
    ])

    #cost += cp.pos(x_goal - x[0, T]) / T
    for t in range(T):
        cost += cp.pos(x_goal - x[0, t + 1]) / T  # -> THAT'S GREAT! Helps to solve the problem in the middle of the way, when controls are ~0...

        constraints.append(x[:, t + 1] == A @ x[:, t] + B @ u[:, t])  # x_t+1 = A * x_t + B * u_t

        # TODO: obstacle avoidance
        for abs_pos, abs_vel, radius in obstacles_full_state:
            safe_radius = radius + 0.3  # TODO: was 0.5
            obs_rel_pos = cp.reshape(abs_pos + abs_vel * t * 0.1 - x[:n, t], (n, 1))  # TODO: was 0.01 here
            obs_rel_pos_T = cp.reshape(abs_pos + abs_vel * t * 0.1 - x[:n, t], (1, n))  # TODO: was 0.01 here

            ## cost += cp.inv_pos(cp.norm(obs_rel_pos)) / T

            m_identity_matrix = -np.eye(n)
            P = cp.Variable((n, n))
            constraints.append(
                cp.trace(m_identity_matrix @ P) + safe_radius ** 2 <= 0
            )

            #cost += cp.trace(P)

            schur = cp.bmat([[P, obs_rel_pos], [obs_rel_pos_T, np.eye(1)]])
            constraints.append(
                schur >> 0
            )

    problem = cp.Problem(cp.Minimize(cost), constraints)

    try:
        problem.solve(max_iters=500, verbose=True)  # verbose=True
    except cp.error.SolverError as e:
        print(e)
        return np.array([0.0, 0.0])

    print(f"MPC problem status: {problem.status}")
    print(f"Optimal value at time {state.t} is {problem.value}")
    print(f"(All) optimal states {x.value}")
    print(f"(Current) optimal controls {u.value[:, 0] if u.value is not None else u.value}")

    if problem.status in ["optimal", "optimal_inaccurate"]:
        return u.value[:n, 0]
    elif problem.status in ["infeasible", "infeasible_inaccurate"]:
        return np.array([0.0, 0.0])


if __name__ == "__main__":
    main_test(solve_fn=solve_mpc)
