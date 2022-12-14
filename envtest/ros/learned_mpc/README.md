# Trained Learning-based MPC models
The best trained Learning-based MPC models for the DodgeDrone-Challenge 2022 will be kept here.

Each model is a `.pth` file, which can be read and prepared for inference by the PyTorch.

The information on the exact models, training hyperparameters, etc. is stored in the [hyperdata file](hyperdata.txt). The loading of the correct MPC model class for inference occurs automatically from the given model path (correct schema of the [hyperdata file](hyperdata.txt) implied!).

## Best models (ranking)
After the full evaluation of the models (see the [CSV-table](../../../final_evaluation/final_summary_learned_mpc_choice.csv)), the following ranking is given:

| Ranking | Model path                                                          | Medium environments solved | Hard environments solved | Amount of out-of-bounds | Avg. number of collisions (on medium) | Avg. number of collisions (on hard) | Note                                |
|---------|---------------------------------------------------------------------|----------------------------|--------------------------|-------------------------|---------------------------------------|-------------------------------------|-------------------------------------|
|    1    | learned_mpc/nmpc_short_controls_first_model_deep_obstacles_only.pth | 7 / 101                    | 0 / 101                  | 42 / 202                | 3.86                                  | 7.57                                | Best model; only obstacles as input |
|    2    | learned_mpc/nmpc_short_controls_first_model_deep.pth                | 2 / 101                    | 0 / 101                  | 66 / 202                | 5.33                                  | 8.76                                | 2nd best model                      |
|    3    | learned_mpc/nmpc_short_controls_first_model.pth                     | n/a                        | n/a                      | ~17 / 50                | 5.07                                  | 9.64                                | Quite a general model               |
|    4    | learned_mpc/nmpc_short_controls_first_model_wide_relu.pth           | n/a                        | n/a                      | ~23 / 50                | 3.34                                  | 10.64                               | Very few collisions on medium!   |


Additionally, the [overfitted model](overfit_controls/) was overfitted on the `hard_0` environment with very little validation error, and can solve this environment efficiently, which confirm the capabilities of the training.
