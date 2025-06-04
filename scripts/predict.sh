DATASET_LIST="webqsp cwq"    ## webqsp or cwq
MODEL_TYPE=webqsp_cwq_tuned ## webqsp_tuned or un_tuned or webqsp_cwq_tuned
MODEL_NAME=KG-TRACES

MODEL_PATH=models/KG-TRACES

PRED_PATH_TYPE_LIST="relation"  ## relation or triple or relation_triple
BEAM_LIST="1 2 3 4 5"


for DATA_SET in $DATASET_LIST; do
        for PRED_PATH_TYPE in $PRED_PATH_TYPE_LIST; do
                for N_BEAM in $BEAM_LIST; do
                        echo "--------------Reasoning of dataset: [$DATA_SET] and beam: [$N_BEAM]--------------"
                        PRED_RELATION_PATH_PATH=results/gen_predict_path/${DATA_SET}/${MODEL_NAME}/test/type_relation/predictions_${N_BEAM}_False.jsonl
                        PRED_TRIPLE_PATH_PATH=results/gen_predict_path/${DATA_SET}/${MODEL_NAME}/test/type_triple/predictions_${N_BEAM}_False.jsonl
                        python src/qa_prediction/predict_answer.py \
                                --dataset=${DATA_SET} \
                                --batch_size=64 \
                                --model_name=${MODEL_NAME} \
                                --model_path=${MODEL_PATH} \
                                --model_type=${MODEL_TYPE} \
                                --pred_relation_path_path=${PRED_RELATION_PATH_PATH} \
                                --pred_triple_path_path=${PRED_TRIPLE_PATH_PATH} \
                                --n_beam=${N_BEAM} \
                                --add_path \
                                --use_pred_path \
                                --pred_path_type=${PRED_PATH_TYPE} \

                done
        done
done