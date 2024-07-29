#!/bin/bash

# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_joint_oas_3ct_tm_imputed"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_joint_oas_3ct_tm"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/rt_cfps_tm"

PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm/checkpoint-rmse-min-19800"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm_oas_merged/checkpoint-rmse-min-22600"
# PRETRAINED_MODEL_PATH="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/models/rt/3ct_tm_oas_imputed/checkpoint-pearson-max-15000"

MODEL_NAME="3ct_tm"
# MODEL_NAME="joint_oas_3ct_tm"
# MODEL_NAME="joint_oas_3ct_tm_imputed"
########## HYPERPARAMETERS ##########
cond_values=(0.001 0.250 0.500 0.750 1.000)

temperature=1.0
batch_size=32
tolerance=100
num_avg_mutations=6
num_samples=1000
property_tokens="<tm>"
SEED_CONSTRUCTS="vhh-capulet-001_0719_no-tag_5662 vhh-capulet-001_0719_no-tag_6440 vhh-capulet-001_0719_no-tag_7275 vhh-capulet-001_0719_no-tag_6304 vhh-capulet-001_0719_no-tag_6327 vhh-capulet-001_0719_no-tag_5682 vhh-capulet-001_0719_no-tag_6279"

for seed_construct in ${SEED_CONSTRUCTS}; do
  OUTPUT_DIR="/home/ec2-user/other/slu/projects/bh-experimental/wizard_hat/conditional_plm/res/samples/$seed_construct/rt/$MODEL_NAME"
  echo "Output directory: $OUTPUT_DIR"

  for conditioning_value in "${cond_values[@]}"; do
    python run_sample.py \
      --model_path $PRETRAINED_MODEL_PATH \
      --output_dir $OUTPUT_DIR \
      --search_method denoise \
      --temperature $temperature \
      --batch_size $batch_size \
      --tolerance $tolerance \
      --num_avg_mutations $num_avg_mutations \
      --num_samples $num_samples \
      --run_name "c=${conditioning_value}_p=${num_avg_mutations}_n=${num_samples}" \
      --property_tokens $property_tokens \
      --conditioning_value $conditioning_value \
      --seed_construct $seed_construct
      
    if [ $? -ne 0 ]; then
      echo "Error in running sample.py"
      exit 1
    fi
  done
done