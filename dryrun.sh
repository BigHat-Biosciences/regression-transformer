#!/bin/bash

export SM_MODEL_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/test_rt"
export SM_CHANNEL_TRAINING="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/datasets/rt/examples"
export PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/experiments/rt/finetune_10ct_display_on_cfps_tm/checkpoint-pearson-max-20000/"

python -m pdb run_train.py \
	--config_name configs/rt_small.json \
    --tokenizer_name vocabs/smallmolecules.txt \
    --train_data_path qed_property_example.csv \
    --test_data_path qed_property_example.csv \
	--do_train True \
	--do_eval True \
	--cc_loss True \
    --warmup_steps 0 \
    --lr_scheduler_type linear \
    --evaluate_during_training True \
    --learning_rate 0.0001 \
    --line_by_line True \
	--overwrite_output_dir True \
	--batch_size 16 \
	--eval_steps 5 \
	--logging_steps 100 \
	--alternate_steps 50 \
	--gradient_accumulation_steps 1 \
	--eval_accumulation_steps 1 \
	--cc_loss_weight 1 \
	--save_total_limit 2 \
	--save_steps 0 \
	--num_train_epochs 5 \
	--augment 0 \
    --seed 42 \
    --report_to none