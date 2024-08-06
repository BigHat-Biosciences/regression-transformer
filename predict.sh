#!/bin/bash

# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm/checkpoint-rmse-min-19800"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm_oas_merged/checkpoint-rmse-min-22600"
PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm_oas_imputed/checkpoint-pearson-max-15000"

SAMPLES_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/rt/capulet-vhh-capulet_new-qc-thermostability-ml-sequences-0_21/datum.csv"

# MODEL_NAME="3ct_tm"
# MODEL_NAME="joint_oas_3ct_tm"
MODEL_NAME="joint_oas_3ct_tm_imputed"

OUTPUT_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/preds/rt/${MODEL_NAME}"

########## HYPERPARAMETERS ##########
temperature=1.0
batch_size=32
property_tokens="<tm>"

python run_predict.py \
    --model_path $PRETRAINED_MODEL_PATH \
    --samples_path $SAMPLES_PATH \
    --output_dir $OUTPUT_DIR \
    --temperature $temperature \
    --batch_size $batch_size \
    --run_name "${MODEL_NAME}_preds_on_cfps_tm" \
    --property_tokens $property_tokens