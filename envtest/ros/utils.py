#!/usr/bin/python3

import numpy as np
from scipy.spatial.transform import Rotation as R


class AgileCommandMode(object):
    """Defines the command type."""
    # Control individual rotor thrusts.
    SRT = 0
    # Specify collective mass-normalized thrust and bodyrates.
    CTBR = 1
    # Command linear velocities. The linear velocity is expressed in world frame.
    LINVEL = 2

    def __new__(cls, value):
        """Add ability to create CommandMode constants from a value."""
        if value == cls.SRT:
            return cls.SRT
        if value == cls.CTBR:
            return cls.CTBR
        if value == cls.LINVEL:
            return cls.LINVEL

        raise ValueError('No known conversion for `%r` into a command mode' % value)


class AgileCommand:
    def __init__(self, mode):
        self.mode = AgileCommandMode(mode)
        self.t = 0.0

        # SRT functionality
        self.rotor_thrusts = [0.0, 0.0, 0.0, 0.0]

        # CTBR functionality
        self.collective_thrust = 0.0
        self.bodyrates = [0.0, 0.0, 0.0]

        # LINVEL functionality
        self.velocity = [0.0, 0.0, 0.0]
        self.yawrate = 0.0

    def __repr__(self):
        repr_str = "AgileCommand:\n" \
                   + " t:     [%.2f]\n" % self.t \
                   + " vel:   [%.2f, %.2f, %.2f]\n" % (self.velocity[0], self.velocity[1], self.velocity[2]) \
                   + " yawrate: [%.2f]" % self.yawrate
        return repr_str


class AgileQuadState:
    def __init__(self, quad_state):
        self.t = quad_state.t

        self.pos = np.array([quad_state.pose.position.x,
                             quad_state.pose.position.y,
                             quad_state.pose.position.z], dtype=np.float32)
        self.att = np.array([quad_state.pose.orientation.w,
                             quad_state.pose.orientation.x,
                             quad_state.pose.orientation.y,
                             quad_state.pose.orientation.z], dtype=np.float32)
        self.vel = np.array([quad_state.velocity.linear.x,
                             quad_state.velocity.linear.y,
                             quad_state.velocity.linear.z], dtype=np.float32)
        self.omega = np.array([quad_state.velocity.angular.x,
                               quad_state.velocity.angular.y,
                               quad_state.velocity.angular.z], dtype=np.float32)

    def __repr__(self):
        repr_str = "AgileQuadState:\n" \
                   + " t:     [%.2f]\n" % self.t \
                   + " pos:   [%.2f, %.2f, %.2f]\n" % (self.pos[0], self.pos[1], self.pos[2]) \
                   + " att:   [%.2f, %.2f, %.2f, %.2f]\n" % (self.att[0], self.att[1], self.att[2], self.att[3]) \
                   + " vel:   [%.2f, %.2f, %.2f]\n" % (self.vel[0], self.vel[1], self.vel[2]) \
                   + " omega: [%.2f, %.2f, %.2f]" % (self.omega[0], self.omega[1], self.omega[2])
        return repr_str


def transform_obstacles(state, obstacles, absolute, as_np, qty=None):
    """Transforms ROS obstacle message into different representation."""
    rot = (
        R.from_quat([state.att[1], state.att[2], state.att[3], state.att[0]])
        if not absolute
        else R.identity()
    )

    obstacles_list = []
    qty = qty or len(obstacles.obstacles)
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

            obs_full_state = (
                np.concatenate(
                    (
                        rot.apply(obs_rel_pos + state.pos * absolute),
                        rot.apply(obs_vel - state.vel * (not absolute)),
                        obstacle.scale,
                    ),
                    axis=None,
                )
                if as_np
                else [
                    rot.apply(obs_rel_pos + state.pos * absolute),
                    rot.apply(obs_vel - state.vel * (not absolute)),
                    obstacle.scale,
                ]
            )
            obstacles_list.append(obs_full_state)

    obstacles_arr = (
        np.pad(
            np.array(obstacles_list),
            ((0, len(obstacles.obstacles) - len(obstacles_list)), (0, 0)),
        )
        if as_np
        else None
    )
    return obstacles_arr if as_np else obstacles_list


def get_goal_direction(state, goal=60.0):
    """Returns the goal direction given drone orientation."""
    rot = R.from_quat([state.att[1], state.att[2], state.att[3], state.att[0]])
    goal_3d = np.array([goal, 0.0, 5.0])

    rel_goal = goal_3d - state.pos
    goal_direction = rot.apply(rel_goal)
    return goal_direction
