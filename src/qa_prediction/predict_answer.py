import sys
import os
import torch
from datasets import load_dataset
import json
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
p_project_root = os.path.dirname(project_root) 
sys.path.extend([project_root, p_project_root])

from llms import get_registed_model
from qa_prediction.evaluate_results import eval_result
from qa_prediction.build_qa_input import PromptBuilder
from tools import logger_factory, io_file

logger = logger_factory.setup_logger("predict_answer")


def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def merge_pred_path_result(qa_dataset, relation_path_dataset, triple_path_dataset, n_proc=1, filter_empty=False):
    question_to_relation_path = dict()
    question_to_triple_path = dict()
    for data in relation_path_dataset:
        qid = data["id"]
        predicted_paths = data["prediction_paths"]
        ground_paths = data["ground_paths"]
        question_to_relation_path[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }
    for data in triple_path_dataset:
        qid = data["id"]
        predicted_paths = data["prediction_paths"]
        ground_paths = data["ground_paths"]
        question_to_triple_path[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_relation_paths"] = []
        sample["predicted_triple_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_relation_paths"] = question_to_relation_path[qid]["predicted_paths"]
        sample["predicted_triple_paths"] = question_to_triple_path[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_relation_path[qid]["ground_paths"]
        return sample

    logger.info("Merging predicted paths...")
    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset


def prediction(batch_data, processed_list, input_builder, model):
    # Extract the input data from the batch

    questions, answers, ids = [], [], []
    filter_batch_data = []
    for data in batch_data:
        if data["id"] in processed_list:
            continue
        questions.append(data["question"])
        answers.append(data["answer"])
        ids.append(data["id"])
        filter_batch_data.append(data)
    
    if len(questions) == 0:
        return None
        
    # Process the batch data
    inputs = [input_builder.process_input(data)[0] for data in filter_batch_data]
    lists_of_paths = [input_builder.process_input(data)[1] for data in filter_batch_data]
    num_of_paths = [len(lists_of_paths[i]) if lists_of_paths[i] else 0 for i in range(len(lists_of_paths))]
    
    # Get model predictions for the whole batch
    predictions = model.generate_sentence_batch(inputs)

    results = []
    for i, data in enumerate(filter_batch_data):
        # Ensure that predictions match the input order
        result = {
            "id": ids[i],
            "question": questions[i],
            "input": inputs[i],
            "prediction": predictions[i],
            "ground_truth": answers[i],
            "input_token_num":model.token_len(inputs[i]),
            "output_token_num":model.token_len(predictions[i][0]),
            "lists_of_paths": lists_of_paths[i],
            "num_of_paths": num_of_paths[i]
        }
        results.append(result)

    return results


def main(args, LLM):
    input_file = os.path.join(args.data_path, args.dataset)
    path_postfix = "no_path"

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    if args.add_path:
        if args.use_true:
            path_postfix = "ground_path"
        elif args.use_random:
            path_postfix = f"random_path_num_{args.path_num}_len_{args.path_len}"
        elif args.use_weight:
            path_postfix = f"weight_path_num_{args.path_num}_len_{args.path_len}"
        elif args.use_pred_path:
            logger.info(f"\n[num_beam]: {args.n_beam}\n")
            path_postfix = f"pred_{args.pred_path_type}_path_beam_{args.n_beam}"
            relation_path_dataset = io_file.read(args.pred_relation_path_path)
            triple_path_dataset = io_file.read(args.pred_triple_path_path)
            dataset = merge_pred_path_result(dataset, relation_path_dataset, triple_path_dataset)

    if args.cot:
        path_postfix += "_cot"
    if args.explain:
        path_postfix += "_explain"
    if args.filter_empty:
        path_postfix += "_filter_empty"
    if args.each_line:
        path_postfix += "_each_line"
        
    logger.info("Load dataset finished")
    output_dir = os.path.join(
        args.predict_path, args.dataset, args.model_type, args.model_name, args.split, path_postfix
    )
    logger.info(f"Save results to: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Predict
    if LLM is not None:
        model = LLM(args)
        input_builder = PromptBuilder(
            args.add_path,
            use_true=args.use_true,
            use_weight=args.use_weight,
            use_pred_path=args.use_pred_path,
            pred_path_type=args.pred_path_type,
            cot=args.cot,
            explain=args.explain,
            use_random=args.use_random,
            each_line=args.each_line,
            maximun_token=model.maximun_token,
            tokenize=model.token_len,
        )

        logger.info("Prepare pipline for inference...")
        model.prepare_for_inference()
    else:
        model = None
        # Directly return last entity as answer
        input_builder = PromptBuilder(args.add_path, use_true=args.use_true)

    # Save args file
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    output_file = os.path.join(output_dir, f"predictions.jsonl")
    fout, processed_list = get_output_file(output_file, force=args.force)

    logger.info(f"Already processed data num: {len(processed_list)}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda x: x)
    for data in tqdm(dataloader, desc="Evaluating..."):
        res = prediction(data, processed_list, input_builder, model)
        if res is not None:
            for r in res:  # Iterate over the list of dictionaries
                fout.write(json.dumps(r) + "\n")
                fout.flush()
    fout.close()
    
    is_tuned = False if args.model_type == "un_tuned" else True
    eval_result(output_file, is_tuned=is_tuned)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path", type=str, default="data"
    )
    argparser.add_argument("--dataset", "-d", type=str, default="webqsp")
    argparser.add_argument("--model_type", type=str, default="webqsp_cwq_tuned", help="webqsp_cwq_tuned or webqsp_tuned or un_tuned")
    argparser.add_argument("--split", type=str, default="test")
    argparser.add_argument("--predict_path", type=str, default="results/KGQA")


    argparser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="KG-TRACES",
    )
    argparser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default="model/KG-TRACES",
    )

    argparser.add_argument("--pred_relation_path_path", type=str, default="results/gen_predict_path/webqsp/test/KG-TRACES/type_relation/predictions_3_True.jsonl")
    argparser.add_argument("--pred_triple_path_path", type=str, default="results/gen_predict_path/webqsp/test/KG-TRACES/type_triple/predictions_3_True.jsonl")
    argparser.add_argument("--add_path", action="store_true")
    argparser.add_argument("--use_true", action="store_true")
    argparser.add_argument("--cot", action="store_true")
    argparser.add_argument("--explain", action="store_true")
    argparser.add_argument("--use_random", action="store_true")
    argparser.add_argument("--use_weight", action="store_true")
    argparser.add_argument("--use_pred_path", action="store_true")
    argparser.add_argument("--pred_path_type", type=str, default="relation", help="relation or triple")
    argparser.add_argument("--n_beam", type=int, default=3)
    argparser.add_argument("--each_line", action="store_true")
    argparser.add_argument("--path_num", type=int, default=3, help="random path num")
    argparser.add_argument("--path_len", type=int, default=3, help="random path len")

    argparser.add_argument("--batch_size", type=int, default=2, help="batch size to evaluate")
    argparser.add_argument("--skip_special_tokens", type=bool, default=True, help="Whether to skip special tokens when LLM decoding")
    argparser.add_argument("--force", "-f", action="store_true", help="force to overwrite the results")
    argparser.add_argument("-n", default=1, type=int, help="number of processes")
    argparser.add_argument("--filter_empty", action="store_true")
    argparser.add_argument("--debug", action="store_true")
    
    

    args, _ = argparser.parse_known_args()
    if args.model_name != "no-llm":
        LLM = get_registed_model(args.model_name)
        LLM.add_args(argparser)
    else:
        LLM = None
    args = argparser.parse_args()
    args.generation_mode = "greedy"


    logger.info(args)

    main(args, LLM)
