#!/bin/bash

# Pass number of rollouts as argument
if [ $1 ]
then
  N="$1"
else
  N=10
fi

# Pass PPO trial number as argument
if [ $2 ]
then
  PPO_TRIAL="$2"
else
  PPO_TRIAL=1
fi

# Pass difficulty level and currently tested environment as argument
# (only used during final evaluation)
if [ $3 ] && [ $4 ]
then
  DIFFICULTY_LEVEL="$3"
  TESTED_ENV="$4"
fi

# Set Flightmare Path if it is not set
if [ -z $FLIGHTMARE_PATH ]
then
  export FLIGHTMARE_PATH=$PWD/flightmare
fi

# Launch the simulator, unless it is already running
if [ -z $(pgrep visionsim_node) ]
then
  roslaunch envsim visionenv_sim.launch render:=True &
  ROS_PID="$!"
  echo $ROS_PID
  sleep 10
else
  ROS_PID=""
fi

SUMMARY_FILE="evaluation.yaml"
> $SUMMARY_FILE

# Perform N evaluation runs
for i in $(eval echo {1..$N})
do
  # Publish simulator reset
  rostopic pub /kingfisher/dodgeros_pilot/off std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/reset_sim std_msgs/Empty "{}" --once
  rostopic pub /kingfisher/dodgeros_pilot/enable std_msgs/Bool "data: true" --once
  rostopic pub /kingfisher/dodgeros_pilot/start std_msgs/Empty "{}" --once

  export ROLLOUT_NAME="rollout_""$i"
  echo "$ROLLOUT_NAME"

  cd ./envtest/ros/
  python3 evaluation_node.py &
  PY_PID="$!"

  DIR="rl_policy/PPO_${PPO_TRIAL}/"
  if [ -d "$DIR" ]
  then
    python3 run_competition.py --ppo_path "rl_policy/PPO_${PPO_TRIAL}/" &
  else
    python3 run_competition.py --ppo_path "rl_policy/RecurrentPPO_${PPO_TRIAL}/" &
  fi
  COMP_PID="$!"

  cd -

  sleep 2.0
  rostopic pub /kingfisher/start_navigation std_msgs/Empty "{}" --once

  # Wait until the evaluation script has finished
  while ps -p $PY_PID > /dev/null
  do
    sleep 1
  done

  cat "$SUMMARY_FILE" "./envtest/ros/summary.yaml" > "tmp.yaml"
  mv "tmp.yaml" "$SUMMARY_FILE"

  kill -SIGINT "$COMP_PID"
done

if [ $3 ] && [ $4 ]
then
  EVALUATION_SUMMARY_FILE="$HOME/Documents/PPO-baseline/Evaluation/PPO_${PPO_TRIAL}.yaml"
  (echo -e "\n\n---[${DIFFICULTY_LEVEL}/environment_${TESTED_ENV}]---" ; cat "$SUMMARY_FILE" ) >> "$EVALUATION_SUMMARY_FILE"
fi

if [ $ROS_PID ]
then
  kill -SIGINT "$ROS_PID"
fi
