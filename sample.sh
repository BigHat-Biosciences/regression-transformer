#!/bin/bash

# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_joint_oas_3ct_tm_imputed_with_confidence"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_joint_oas_3ct_tm_imputed"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_joint_oas_3ct_tm"
PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_cfps_tm"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_3ct_tm"
OUTPUT_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples/vhh-capulet-001_0719_no-tag_5662/rt/effect_of_n"

########## HYPERPARAMETERS ##########
temperature=1.0
batch_size=32
tolerance=100
num_avg_mutations=6
num_samples=10000
conditioning_value="0.8"
property_tokens="<tm>"

python run_sample.py \
	--model_path $PRETRAINED_MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --search_method sample \
  --temperature $temperature \
  --batch_size $batch_size \
  --tolerance $tolerance \
  --num_avg_mutations $num_avg_mutations \
  --num_samples $num_samples \
  --run_name "cfps_p=${num_avg_mutations}_c=${conditioning_value}" \
  --property_tokens $property_tokens \
  --conditioning_value $conditioning_value \