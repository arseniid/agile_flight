# DodgeDrone: Vision-based Agile Drone Flight (ICRA 2022 Competition)

[![IMAGE ALT TEXT HERE](docs/imgs/video.png)](https://youtu.be/LSu25NH6fW0)


Would you like to push the boundaries of drone navigation? Then participate in the dodgedrone competition!
You will get the chance to develop perception and control algorithms to navigate a drone in both static and dynamic environments. Competing in the challenge will deepen your expertise in computer vision and control, and boost your research.
You can find more information at the [competition website](https://uzh-rpg.github.io/icra2022-dodgedrone/).

This codebase provides the following functionalities:

1. A simple high-level API to evaluate your navigation policy in the Robot Operating System (ROS). This is completely independent on how you develop your algorithm.
2. Training utilities to use reinforcement learning for the task of high-speed obstacle avoidance.

All evaluation during the competition will be performed using the same ROS evaluation, but on previously unseen environments / obstacle configurations.

## Submission

- **06 May 2022** Submission is open. Please submit your version of the file [user_code.py](https://github.com/uzh-rpg/agile_flight/blob/main/envtest/ros/user_code.py) with all needed dependencies with an email to loquercio AT berkeley DOT edu. Please use as subject *ICRA 2022 Competition: Team Name*. If you have specific dependencies, please provide instructions on how to install them. Feel free to switch from python to cpp if you want. 

### Further Details

- We will only evaluate on the warehouse environment with spheres obstacles. 
- If you're using vision, you are free to use any sensor you like (depth, optical flow, RGB). The code has to run real-time on a desktop with 16 Intel Core i7-6900K and an NVIDIA Titan Xp.
- If you're using vision, feel free to optimize the camera parameters for performance (e.g. field of view). 
- We will two rankings, one for vision-based and another for state-based. The top three team for each category will qualify for the finals.

## Update

- **02 May 2022** Fix a bug in the vision racing environment when computing reward function. No need to update if you are not using RL or if you have change the reward formualtion. Related to this issue #65

- **27 March 2022** Fix a static object rendering issue. Please download the new Unity Standalone using [this](https://github.com/uzh-rpg/agile_flight/blob/main/setup_py.bash#L32-L39). Also, git pull the project.

## Flight API

This library contains the core of our testing API. It will be used for evaluating all submitted policies. The API is completely independent on how you build your navigation system. You could either use our reinforcement learning interface (more on this below) or add your favourite navigation system.

### Prerequisite

Before continuing, make sure to have g++ and gcc to version 9.3.0. You can check this by typing in a terminal `gcc --version` and `g++ --version`. Follow [this guide](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/) if your compiler is not compatible.

In addition, make sure to have ROS installed. Follow [this guide](http://wiki.ros.org/noetic/Installation/Ubuntu) and install ROS Noetic if you don't already have it.

### Installation (for ROS User)

We only support Ubuntu 20.04 with ROS noetic. Other setups are likely to work as well but not actively supported.

Start by creating a new catkin workspace.

```
cd     # or wherever you'd like to install this code
export ROS_VERSION=noetic
export CATKIN_WS=./icra22_competition_ws
mkdir -p $CATKIN_WS/src
cd $CATKIN_WS
catkin init
catkin config --extend /opt/ros/$ROS_VERSION
catkin config --merge-devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-fdiagnostics-color

cd src
git clone git@github.com:uzh-rpg/agile_flight.git
cd agile_flight
```

Run the `setup_ros.bash` in the main folder of this repository, it will ask for sudo permissions. Then build the packages.

```bash
./setup_ros.bash

catkin build
```

### Installation (for Python User)

If you want to develop algorithms using only Python, especially reinforcement learning, you need to install our library as python package.

**Make sure that you have [anaconda](https://www.anaconda.com/) installed. This is highly recommanded.**

Run the `setup_py.bash` in the main folder of this repository, it will ask for sudo permissions.

```bash
./setup_py.bash
```

### Task  

The task is to control a simulated quadrotor to fly through obstacle dense environments.
The environment contains both static and dynamic obstacles.
You can specifiy which difficulty level and environment you want to load for testing your algorithm.
The yaml configuration file is located in [this file](https://github.com/uzh-rpg/flightmare/blob/dev/version_22/flightpy/configs/vision/config.yaml). 
The goal is to proceed as fast as possible **60m in positive x-direction** without colliding into obstacles and exiting a pre-defined bounding box. 
The parameters of the goal location and the bounding box can be found [here](https://github.com/uzh-rpg/agile_flight/blob/main/envtest/ros/evaluation_config.yaml).

```yaml
environment:
  level: "medium" # three difficulty level for obstacle configurations [easy, medium, hard]
  env_folder: "environment_0" # configurations for dynamic and static obstacles, environment number are between [0 - 100]

unity:
  scene_id: 0 # 0 warehouse, 1 garage, 2 natureforest, 3 wasteland
```

### Usage

The usage of this code base entails two main aspects: writing your algorithm and testing it in the simulator.

**Writing your algorithm:**

To facilitate coding of your algorithms, we provided a simple code structure for you, just edit the following file: [envtest/ros/user_code.py](https://github.com/uzh-rpg/agile_flight/blob/main/envtest/ros/user_code.py).
This file contains two functions, [compute_command_vision_based](https://github.com/uzh-rpg/agile_flight/blob/main/envtest/ros/user_code.py#L8) and [compute_command_state_based](https://github.com/uzh-rpg/agile_flight/blob/main/envtest/ros/user_code.py#L44).
In the vision-based case, you will get the current image and state of the quadrotor. In the state-based case, you will get the metric distance to obstacles and the state of the quadrotor. We strongly reccomend using the state-based version to start with, it is going to be much easier than working with pixels!

Depending on the part of the competition you are interested in, adapt the corresponding function.
To immediately see something moving, both functions at the moment publish a command to fly straight forward, of course without avoiding any obstacles.
Note that we provide three different control modes for you, ordered with increasing level of abstraction: commanding individual single-rotor thrusts (SRT), specifying mas-normalized collective thrust and bodyrates (CTBR), and outputting linear velocity commands and yawrate (LINVEL). The choice of control modality is up to you.
Overall, the more low-level you go, the more difficult is going to be to mantain stability, but the more agile your drone will be.

**Testing your approach in the simulator:**

Make sure you have completed the installation of the flight API before continuing.
To use the competition software, three steps are required:

1. Start the simulator

   ```
   roslaunch envsim visionenv_sim.launch render:=True
   # Using the GUI, press Arm & Start to take off.
   python evaluation_node.py
   ```

   The evaluation node comes with a config file. There, the options to plot the results can be disabled if you want no plots.
2. Start your user code. This code will generate control commands based on the sensory observations. You can toggle vision-based operation by providing the argument `--vision_based`.

   ```
   cd envtest/ros
   python run_competition.py [--vision_based]
   ```

3. Tell your code to start! Until you publish this message, your code will run but the commands will not be executed. We use this to ensure fair comparison between approaches as code startup times can vary, especially for learning-based approaches.

   ```
   rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" -1
   ```

If you want to perform steps 1-3 automatically, you can use the `launch_evaluation.bash N` script provided in this folder. It will automatically perform `N` rollouts and then create an `evaluation.yaml` file which summarizes the rollout statistics.

**Using Reinforcement Learning (Optional)**
We provide an easy interface for training your navigation policy using reinforcement learning. While this is not required for the competition, it could just make your job easier if you plan on using RL.

Follow [this guide](/envtest/python/README.md) to know more about how to use the training code and some tips on how to develop reinforcement learning algorithms



## Evaluation scripts ##
Since the number of available solutions has grown quite largely, this section might be helpful to know all possibilities to evaluate them.


### launch_evaluation.bash ###
Core `.bash` script to evaluate *one* particular solution on **one** particular environment; allows for multiple rollouts.

How to (for all currently available solutions):
- Deep Reinforcement Learning (in form of a PPO algorithm): Trained policies are located in ['rl_policy/'](/envtest/ros/rl_policy/) folder. To evaluate, run:

   ```
   ./launch_evaluation.bash <N> <PPO_TRIAL_NUMBER> [<EVALUATED_ENV>]
   ```

   where N: int (number of rollouts); PPO_TRIAL_NUMBER: int (number of the trained policy to evaluate). If additionally EVALUATED_ENV: str (any string to distinguish between environments) is given, each result from [summary](evaluation.yaml) will be written to the following "$HOME/Documents/PPO-baseline/Evaluation/PPO_\<PPO_TRIAL\>.yaml" folders/files.

   > **_NOTE:_** To set the desired policy iteration, refer to the [load_rl_policy()](https://github.com/arseniid/agile_flight/blob/main/envtest/ros/rl_example.py#L72-L73) function

- Classical (N)MPC algorithm: Both linear and nonlinear MPCs are defined in [user_code.py](/envtest/ros/user_code.py). To evaluate, run:

   ```
   ./launch_evaluation.bash <N> <any_string> [<EVALUATED_ENV>]
   ```

   where N: int (number of rollouts); any_string: str (any random string, especially **not** an integer). If additionally EVALUATED_ENV: str (any string to distinguish between environments) is given, it will be only used during dataset creation to store data in the correct file.

   > **_NOTE 1:_** To choose the desired algorithm (out of two), one has to adapt [user_code.py](https://github.com/arseniid/agile_flight/blob/main/envtest/ros/user_code.py#L84-L86) manually

   > **_NOTE 2:_** By default, after each MPC run the data will be saved to a dataset (see ['datasets/'](/flightmare/flightpy/datasets/) folder). To stop this, one has to manually set `self.create_dataset = False` in [run_competition.py](https://github.com/arseniid/agile_flight/blob/main/envtest/ros/run_competition.py#L40) or **omit** the \<EVALUATED_ENV\> argument

- Learned (N)MPC: Trained models are located in ['learned_mpc/'](/envtest/ros/learned_mpc/) folder. To evaluate, run:

   ```
   ./launch_evaluation.bash <N> <MPC_MODEL_PATH>
   ```

   where N: int (number of rollouts); MPC_MODEL_PATH: str (relative path to the trained (N)MPC model -- similar to 'learned_mpc/\<model_name\>.pth').

   > **_NOTE:_** The correct MPC class will be inferenced automatically from the given MPC_MODEL_PATH (according to [hyperdata.txt](/envtest/ros/learned_mpc/hyperdata.txt))


### launch_evaluation_all.py ###
Broader `.py` script to evaluate *one* particular solution (by default: MPC) on **multiple** environments; allows for multiple rollouts for each environment.

The script is self-explanatory, uses `./launch_evaluation.bash` internally and is mainly used for dataset creation with (N)MPC.

> **_NOTE:_** The script has a lot of hard-coded parts, and it is advised *against* using this. If one would decided to use it,
please change it accordingly (for example, [subprocess spawning](https://github.com/arseniid/agile_flight/blob/main/launch_evaluation_all.py#L32-L38) -- 
no guarantees for a smooth run!).
