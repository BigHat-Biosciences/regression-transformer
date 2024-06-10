#!/bin/bash

python scripts/run_language_modeling.py \
    --output_dir rt_example \
    --config_name configs/rt_small.json \
    --tokenizer_name ./vocabs/smallmolecules.txt \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --save_total_limit 2 \
    --save_steps 500 \
    --per_gpu_train_batch_size 16 \
    --evaluate_during_training \
    --eval_steps 5 \
    --eval_data_file ./examples/qed_property_example.txt\
     --train_data_file ./examples/qed_property_example.txt \
    --line_by_line \
    --block_size 510 \
    --seed 42 \
    --logging_steps 100 \
    --eval_accumulation_steps 2 \
    --training_config_path training_configs/qed_alternated_cc.json \
    --overwrite_output_dir \
    --no_cuda