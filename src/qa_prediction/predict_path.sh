SPLIT="test"
DATASET_LIST="webqsp cwq"  # "webqsp cwq""
PATH_TYPE_LIST="relation triple" 
MODEL_NAME=KG-TRACES
MODEL_PATH=models/KG-TRACES
PROMPT_PATH=prompts/qwen2.5.txt
OUTPUT_PATH=results/gen_predict_path

BATCH_SIZE=128

export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


BEAM_LIST="1 2 3 4 5" # "1 2 3 4 5"
echo "BEAM_LIST=${BEAM_LIST}"

for PATH_TYPE in $PATH_TYPE_LIST; do
    echo "PATH_TYPE=${PATH_TYPE}"
    for DATASET in $DATASET_LIST; do
        echo "DATASET=${DATASET}"
        for N_BEAM in $BEAM_LIST; do
            echo "N_BEAM=${N_BEAM}"
            python src/qa_prediction/gen_predict_path.py \
                --dataset ${DATASET} \
                --prompt_path ${PROMPT_PATH} \
                --split ${SPLIT} \
                --model_name ${MODEL_NAME} \
                --model_path ${MODEL_PATH} \
                --path_type ${PATH_TYPE} \
                --n_beam ${N_BEAM} \
                --batch_size ${BATCH_SIZE} \
                --output_path ${OUTPUT_PATH}
        done
    done
done