import casadi
import numpy as np

from utils import main_test


def solve_nmpc(state, obstacles):
    obstacle_pos = []
    obstacle_radii = []
    for obstacle in obstacles.obstacles:
        if obstacle.scale != 0:
            obstacle_pos.append(np.array([obstacle.position.x, obstacle.position.y]))
            obstacle_radii.append(obstacle.scale)

    T = 50
    dt = 0.1
    v_max = 5.0
    min_distance = 0.1
    x0 = np.zeros((1, 4))
    x0[0, :2] = state.pos

    xo = np.array(obstacle_pos)
    # xo = np.random.rand(4, 2) * 10
    xg = np.array([10.1, 0])
    xg_T = np.resize(xg, (T, 2))

    opti = casadi.Opti()
    x = opti.variable(T, 4)
    # opti.minimize(casadi.norm_2((x[:, 0] - xg_T[:, 0])**2 + (x[:, 1] - xg_T[:, 1])**2))
    opti.minimize(casadi.norm_2((x[:, 0] - xg_T[:, 0])**2))
    #print((x[:, 0] - xg_T[:, 0]).shape)
    #print(casadi.fmax(0, x[:, 0] - xg_T[:, 0]).shape)
    #print(casadi.sum1(casadi.fmax(0, x[:, 0] - xg_T[:, 0])).shape)
    """opti.minimize(casadi.sum1(casadi.fmax(np.zeros((T, 1)), xg_T[:, 0] - x[:, 0])))"""
    opti.subject_to(x[0, :] == x0)
    opti.subject_to(x[1:, 0] == x[:-1, 0] + dt * x[:-1, 2])
    opti.subject_to(x[1:, 1] == x[:-1, 1] + dt * x[:-1, 3])
    opti.subject_to(-1 < x[:, 0])
    opti.subject_to(-2 < x[:, 1])
    opti.subject_to(x[:, 0] < 15)
    opti.subject_to(x[:, 1] < 2)
    opti.subject_to(-v_max < x[:, 2])
    opti.subject_to(-v_max < x[:, 3])
    opti.subject_to(x[:, 2] < v_max)
    opti.subject_to(x[:, 3] < v_max)
    for i in range(xo.shape[0]):
        for k in range(T):
            opti.subject_to(casadi.sqrt(casadi.sumsqr(x[k, :2] - xo[i, None])) > obstacle_radii[i] + min_distance)

    opti.solver('ipopt')
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(e)
        print(opti.debug.value(x))
        #import time
        #time.sleep(3.0)
        return np.array([5.0, 0.0])
    x_optimal = sol.value(x)

    return x_optimal[1, 2:]


if __name__ == "__main__":
    main_test(solve_fn=solve_nmpc)
