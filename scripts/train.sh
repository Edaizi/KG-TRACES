
MODEL_PATH=model/Qwen2.5-7b-Instruct

DATASET_LIST="data/webqsp/train-reasoning_process.jsonl data/webqsp/train-relation_path.jsonl data/webqsp/train-triple_path.jsonl data/cwq/train-reasoning_process.jsonl data/cwq/train-relation_path.jsonl data/cwq/train-triple_path.jsonl" 

EVAL_DATASET_LIST="data/webqsp/validation.jsonl data/cwq/validation.jsonl"

SAVE_NAME=KG-TRACES
SAVE_PATH=models/${SAVE_NAME}
WANDB_PROJECT_NAME=KG-TRACES

export WANDB_PROJECT="KG-TRACES"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"



accelerate launch --config_file config/deepspeed_zero3_train.yml src/finetuning.py \
    --data_path_list ${DATASET_LIST}  \
    --eval_data_path_list ${EVAL_DATASET_LIST} \
    --model_name_or_path ${MODEL_PATH} \
    --output_dir ${SAVE_PATH} \
    --overwrite_output_dir False \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 1e-7 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --report_to "wandb" \
    --gradient_checkpointing True \
    --run_name ${SAVE_NAME}
