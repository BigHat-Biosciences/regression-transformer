#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_rt"
export SM_CHANNEL_TRAINING="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/rt/oas_display_imputed"
export PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/boman"


########## Finetune from a pretrained model ##########
python run_train.py \
	--model_path $PRETRAINED_MODEL_PATH \
    --train_data_path train.csv \
	--train_metadata_path train_metadata.csv \
    --test_data_path tm_val.csv \
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
	--eval_steps 100 \
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