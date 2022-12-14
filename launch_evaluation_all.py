#!/usr/bin/python3
import argparse
import subprocess
import time


def evaluate_all(difficulty="hard", start_from=0, end_with=100, env_list=None, rollouts=5):
    config_path = "flightmare/flightpy/configs/vision/config.yaml"
    summary_path = "CHANGE_MY_NAME.txt"

    iterate_over = env_list or range(start_from, end_with + 1)
    for i in iterate_over:
        with open(config_path, "r") as config_file:
            config_lines = config_file.read().splitlines()
            difficulty_line = config_lines[1]
            env_line = config_lines[2]

        difficulty_line_split = difficulty_line.split("\"")
        difficulty_line_split[1] = difficulty
        difficulty_line = "\"".join(difficulty_line_split)

        env_line_split = env_line.split("\"")
        env_line_split[1] = f"environment_{i}"
        env_line = "\"".join(env_line_split)

        config_lines[1] = difficulty_line
        config_lines[2] = env_line

        with open(config_path, "w") as config_file:
            config_file.write("\n".join(config_lines) + "\n")

        proc = None
        if "learned" in summary_path and "mpc" in summary_path.lower():
            proc = subprocess.Popen(f"./launch_evaluation.bash {rollouts} learned_mpc/nmpc_short_controls_first_model_deep_obstacles_only.pth", shell=True)
        elif "ppo" in summary_path.lower():
            proc = subprocess.Popen(f"./launch_evaluation.bash {rollouts} 39", shell=True)
        else:  # classical MPC
            proc = subprocess.Popen(f"./launch_evaluation.bash {rollouts} mpc {difficulty}_{i}", shell=True)
        try:
            proc.wait(timeout=100 * rollouts)  # timeout of 70 seconds for each episode + some overhead
        except subprocess.TimeoutExpired as e:
            print(e)
            proc.kill()

        time.sleep(15.0)

        with open(summary_path, "a+") as summary_file:
            summary_file.write(f"\n\n======={difficulty}_{i}=======\n")
            with open("evaluation.yaml", "r") as evaluation_file:
                rollouts_evaluation = evaluation_file.read()
                summary_file.write(rollouts_evaluation)


def evaluate_smart(upper_bound=1):
    metadata_path = "flightmare/flightpy/datasets/nmpc_short/metadata.txt"

    re_evaluation_needed = []
    with open(metadata_path, "r") as metadata_file:
        metadata_lines = metadata_file.read().splitlines()
        for line in metadata_lines:
            env = line.split()[2].rstrip(":")
            crashes = int(line.split()[-2])   
            if crashes > upper_bound:
                re_evaluation_needed.append(env)

    cool_nl = "\n  - "
    print(f"Re-evaluation needed for the following {len(re_evaluation_needed)} environments:{cool_nl}{cool_nl.join(re_evaluation_needed)}")
    for env_to_evaluate in re_evaluation_needed:
        difficulty, start = env_to_evaluate.split("_")
        evaluate_all(
            difficulty=difficulty,
            start_from=int(start),
            end_with=int(start),
            rollouts=10,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small Python evaluator script for evaluating multiple environments at once")
    parser.add_argument(
        "--smart",
        help="Flag for smart evaluation",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--upper-bound",
        help="In case of smart evaluation: Allowed upper bound of collisions to NOT re-evaluate",
        required=False
    )
    parser.add_argument(
        "--difficulty",
        help="Difficulty level",
        required=False,
        default="hard"
    )
    parser.add_argument(
        "--start",
        help="Starting environment",
        required=False,
        default=0
    )
    parser.add_argument(
        "--end",
        help="Last environment",
        required=False,
        default=100
    )
    parser.add_argument(
        "--eval-list",
        help="Comma-separated environment numbers to evaluate",
        required=False,
        default=None
    )
    parser.add_argument(
        "--rollouts",
        help="Number of rolouts for each environment",
        required=False,
        default=5
    )

    args = parser.parse_args()
    if args.smart:
        evaluate_smart() if args.upper_bound is None else evaluate_smart(upper_bound=int(args.upper_bound))
    else:
        evaluate_all(
            difficulty=args.difficulty,
            start_from=int(args.start),
            end_with=int(args.end),
            env_list=args.eval_list.split(",") if args.eval_list else None,
            rollouts=int(args.rollouts),
        )
