#!/bin/bash
nohup python eval_multi_distance_models_hpc.py --data_path ./data/c2p/ \
                                              --n_neighbours 5  --n_train 1000 \
                                              --output ./exps/eval_multi_distance_model/ \
                                              --name eval_c2p_rf_5_1000_single --override \
                                              > ./logs/eval_c2p_rf_5_1000_single.log 2>&1 &

nohup python eval_multi_distance_models_hpc.py --data_path ./data/p2c/ \
                                              --n_neighbours 5  --n_train 1000 \
                                              --output ./exps/eval_multi_distance_model/ \
                                              --name eval_p2c_rf_5_1000_single --override \
                                              > ./logs/eval_p2c_rf_5_1000_single.log 2>&1 &
