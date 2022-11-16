#!/usr/bin/python3
import argparse

import numpy as np
import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Int8

from envsim_msgs.msg import ObstacleArray
from rl_example import load_rl_policy
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState

from learn_mpc.utils import DataSaver  # flightmare/flightrl package


class AgilePilotNode:
    def __init__(self, vision_based=False, ppo_path=None, mpc_path=None, environment=None):
        print("Initializing agile_pilot_node...")
        rospy.init_node('agile_pilot_node', anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        self.environment = environment if environment else "undefined"
        if ppo_path is not None:
            self.rl_policy = load_rl_policy(ppo_path)
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None

        self.crashes = 0

        self.create_dataset = True and ppo_path is None and mpc_path is None and environment
        if self.create_dataset:
            self.data_saver = DataSaver(folder="nmpc")
            self.sequences_stored = 0
            # 200 seems to be a good upper bound for the amount of sequences in the current setting
            self.data_sequences_in = np.zeros(shape=(200, 111))
            self.data_sequences_out = np.zeros(shape=(200, 72))

        self.predicted_not_executed_states = None

        rospy.on_shutdown(self.cleanup)

        quad_name = 'kingfisher'

        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/state", QuadState, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        self.img_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/unity/depth", Image, self.img_callback,
                                        queue_size=1, tcp_nodelay=True)
        self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles", ObstacleArray,
                                             self.obstacle_callback, queue_size=1, tcp_nodelay=True)

        # Information flow subscribers
        self.crash_sub = rospy.Subscriber("/" + quad_name + "/times_crashed", Int8, self.crash_callback,
                                          queue_size=1, tcp_nodelay=True)

        # Command publishers
        self.cmd_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/feedthrough_command", Command, queue_size=1)
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/velocity_command", TwistStamped,
                                          queue_size=1)
        print("Initialization completed!")

    def img_callback(self, img_data):
        if not self.vision_based:
            return
        if self.state is None:
            return
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')
        command = compute_command_vision_based(self.state, cv_image)
        self.publish_command(command)

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)
        if self.create_dataset and self.state.pos[0] >= 60:
            log_fn = rospy.logwarn_once if self.crashes else rospy.loginfo_once
            log_fn(f"Created {self.sequences_stored} sequences, trying to save them... Times crashed: {self.crashes}")
            self.data_saver.save_data(
                input_data=self.data_sequences_in,
                output_data=self.data_sequences_out,
                environment=self.environment,
                size=self.sequences_stored,
                crashes=self.crashes,
            )
            self.create_dataset = False  # data should be already stored by now

    def obstacle_callback(self, obs_data):
        if self.vision_based:
            return
        if self.state is None or self.state.pos[0] > 60:
            return
        if self.rl_policy:
            command = compute_command_state_based(state=self.state, obstacles=obs_data, rl_policy=self.rl_policy)
            self.publish_command(command)
        else:
            if self.create_dataset:
                obstacles_arr = self._transform_obstacles(state=self.state, obstacles=obs_data)
                try:
                    self.data_sequences_in[self.sequences_stored] = np.concatenate((obstacles_arr, self.state.pos, self.state.vel), axis=None)
                except IndexError as e:  # usually, in case of no movement whysoever goes beyond of 200 (but no meaningful data)
                    rospy.logerr(e)
                    rospy.signal_shutdown("Shut down node due to no movement")
            mpc_dt = 0.08
            commands_list, not_executed = compute_command_state_based(
                state=self.state,
                obstacles=obs_data,
                rl_policy=self.rl_policy,
                mpc_dt=mpc_dt,
                predicted=self.predicted_not_executed_states,
            )
            if self.create_dataset:
                try:
                    self.data_sequences_out[self.sequences_stored] = not_executed[:, 3:6].flatten()
                except IndexError as e:  # usually, in case of no movement whysoever goes beyond of 200 (but no meaningful data)
                    rospy.logerr(e)
                    rospy.signal_shutdown("Shut down node due to no movement")
                self.sequences_stored += 1
            idx_first_not_executed = self.publish_batch(commands_list, dt=mpc_dt)
            self.predicted_not_executed_states = not_executed[idx_first_not_executed:, :]

    def _transform_obstacles(self, state, obstacles):
        """ Transforms ROS obstacle message into NumPy array of absolute obstacle states """
        obstacles_arr = np.zeros(shape=(len(obstacles.obstacles), 7))
        for idx, obstacle in enumerate(obstacles.obstacles):
            if obstacle.scale != 0:  # FIXME: be consistent -> zero-obstacles should go at the end
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
                obstacles_arr[idx] = np.concatenate((obs_rel_pos + state.pos, obs_vel, obstacle.scale), axis=None)
        return obstacles_arr

    def publish_batch(self, commands_list, dt=0.01):
        """ Wrapper around `publish_command` function to publish a batch of commands """
        executed_until = 0
        to_execute = 2
        for command in commands_list:
            if command.t >= rospy.get_time():
                if executed_until < to_execute:
                    executed_until += 1
                    self.publish_command(command)
                    if executed_until != to_execute:
                        rospy.sleep(dt - 0.001)
                else:
                    break
        return to_execute

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True

    def crash_callback(self, data):
        self.crashes = data.data

    def cleanup(self):
        rospy.logwarn("Shutdown called!")
        rospy.sleep(3.0)
        for sub in [self.start_sub, self.odom_sub, self.img_sub, self.obstacle_sub]:
            sub.unregister()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agile Pilot.')
    parser.add_argument('--vision_based', help='Fly vision-based', required=False, dest='vision_based',
                        action='store_true')
    parser.add_argument('--ppo_path', help='PPO neural network policy', required=False, default=None)
    parser.add_argument('--mpc_path', help='MPC learned neural network', required=False, default=None)
    parser.add_argument('--environment', help='Currently loaded environment', required=False, default=None)

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, ppo_path=args.ppo_path, mpc_path=args.mpc_path, environment=args.environment)
    rospy.spin()
