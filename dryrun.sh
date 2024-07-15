#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_rt"
export SM_CHANNEL_TRAINING="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/rt/examples"
export PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/.gt4sd/algorithms/rt/"


# ########## Train from scratch ##########
# python run_train.py \
# 	--config_name configs/rt_small.json \
#     --tokenizer_name vocabs/smallmolecules.txt \
#     --train_data_path qed_property_example.csv \
# 	--train_metadata_path qed_property_metadata.csv \
#     --test_data_path qed_property_example.csv \
# 	--do_train True \
# 	--do_eval True \
# 	--cc_loss True \
# 	--cc_loss_weight 0 \
#     --warmup_steps 0 \
#     --lr_scheduler_type linear \
#     --evaluate_during_training True \
#     --learning_rate 0.0001 \
#     --line_by_line True \
# 	--overwrite_output_dir True \
# 	--batch_size 16 \
# 	--per_device_train_batch_size 16 \
#   	--per_device_eval_batch_size 16 \
# 	--eval_steps 5 \
# 	--logging_steps 10 \
# 	--alternate_steps 1 \
# 	--gradient_accumulation_steps 1 \
# 	--eval_accumulation_steps 1 \
# 	--save_total_limit 2 \
# 	--save_steps 0 \
# 	--num_train_epochs 5 \
# 	--augment 0 \
#     --seed 42 \
#     --report_to none


export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_rt"
export SM_CHANNEL_TRAINING="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/rt/oas_and_capulet_382_3ct_tm_imputed"
export PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt_joint_oas_disp/checkpoint-rmse-min-17400"


########## Finetune from a pretrained model ##########
python run_train.py \
	--model_path $PRETRAINED_MODEL_PATH \
    --train_data_path val.csv \
    --test_data_path val.csv \
	--do_train True \
	--do_eval True \
	--cc_loss True \
	--cc_loss_weight 0 \
    --warmup_steps 0 \
	--learning_rate 0.0001 \
    --lr_scheduler_type linear \
    --evaluate_during_training True \
    --line_by_line True \
	--overwrite_output_dir True \
	--batch_size 16 \
	--per_device_train_batch_size 16 \
  	--per_device_eval_batch_size 16 \
	--eval_steps 2 \
	--logging_steps 200 \
	--alternate_steps 50 \
	--gradient_accumulation_steps 1 \
	--eval_accumulation_steps 1 \
	--save_total_limit 3 \
	--save_steps 0 \
	--num_train_epochs 100 \
	--augment 0 \
    --seed 42 \
    --report_to none
