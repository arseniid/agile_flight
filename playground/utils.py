from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_plots(obstacles, traj):
    ax = plt.gca()
    ax.set_xlim([-1, 11])
    ax.set_ylim([-2, 2])

    for obstacle in obstacles.obstacles:
        circle = plt.Circle((obstacle.position.x, obstacle.position.y), obstacle.scale, color='r')
        ax.add_patch(circle)

    x_points = np.array([coord[0] for coord in traj])
    y_points = np.array([coord[1] for coord in traj])

    plt.plot(x_points, y_points)
    plt.title("Drone trajectory in x-y plane & obstacles")
    plt.show()


def main_test(solve_fn, print_stats=True):
    class State:
        def __init__(self, position: np.array) -> None:
            self.pos = position
            self.t = datetime.now()

        def __repr__(self) -> str:
            return "State is " + str(self.pos) + " at time " + str(self.t)

    class ObstacleArray:
        class Obstacle:
            class Vector2d:
                def __init__(self, x, y) -> None:
                    self.x = x
                    self.y = y

                def __repr__(self) -> str:
                    return "[" + str(self.x) + ", " + str(self.y) + "]"

            def __init__(self, pos: List, vel: List, radius: float) -> None:
                self.position = self.Vector2d(pos[0], pos[1])
                self.linear_velocity = self.Vector2d(vel[0], vel[1])
                self.scale = radius

            def __repr__(self) -> str:
                return "Obstacle at " + str(self.position) + " with velocity " + str(
                    self.linear_velocity) + " of radius " + str(self.scale)

        def __init__(self) -> None:
            self.obstacles = []

        def add_obstacle(self, pos: List, vel: List, radius: float) -> None:
            self.obstacles.append(self.Obstacle(pos, vel, radius))

        def __repr__(self) -> str:
            ret = []
            for obstacle in self.obstacles:
                ret.append(str(obstacle))
            return "\n".join(ret)

    state = State(np.zeros((2,)))
    obstacles = ObstacleArray()
    obstacles.add_obstacle([3.0, 1.5], [0.0, 0.0], 1.0)
    obstacles.add_obstacle([5.0, -1.0], [0.0, 0.0], 0.7)
    obstacles.add_obstacle([9.5, 0.0], [0.0, 0.0], 0.433)

    trajectory = [state.pos]
    collision = False
    if print_stats:
        t_start = datetime.now()
    while state.pos[0] < 10:
        controls = solve_fn(state, obstacles)
        state = State(state.pos + 0.1 * controls)

        print(f"Current controls: {controls}")
        print(f"(Updated) state: {state}")
        print(obstacles)
        trajectory.append(state.pos)

        for i, o in enumerate(obstacles.obstacles):
            dist_o = np.linalg.norm(state.pos - np.array([o.position.x, o.position.y]))
            print(f"{BColors.WARNING}Distance to obstacle #{i} is {dist_o}{BColors.ENDC}")
            if dist_o <= o.scale:
                print(f"{BColors.FAIL}NOOOOOOOOOOOOOO it's a collision!!!!111!!1...{BColors.ENDC}")
                collision = True
                break
    else:
        if not collision:
            print(f"{BColors.OKGREEN}The 'drone' has reached the goal, avoiding all the collisions!{BColors.ENDC}")

    if print_stats:
        print(f"{BColors.OKCYAN}Whole execution took {(datetime.now() - t_start)} s{BColors.ENDC}")
        create_plots(obstacles, trajectory)
