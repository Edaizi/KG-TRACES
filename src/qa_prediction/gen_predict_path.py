import json
import sys
import os
from datasets import load_dataset
import datasets
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import argparse

datasets.disable_progress_bar()
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
from tools.logger_factory import setup_logger

logger = setup_logger("gen_predict_path")


N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)

PATH_RE = r"<PATH>(.*)<\/PATH>"

INSTRUCTION_RELATION = """Please generate a valid reasoning relation path that can be helpful for answering the following question:"""



INSTRUCTION_TRIPLE = """Please generate a valid reasoning triple path that can be helpful for answering the following question
- The reasoning triples path should follow the format: <PATH>subject<SEP>relation<SEP>object</PATH>.
- If multiple triples are needed, output them in a logical sequence.
- If no meaningful relation path can be generated, return <PATH>NONE</PATH>.
"""



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


def parse_relation_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = set()
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.add(rel)
        results.append(list(rules))
    return results



def parse_triple_prediction(predictions):
    """
    Parse a list of predictions to a list of triples

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for prediction in predictions:
        prediction = prediction.strip().split("\n")
        for p in prediction:
            path = re.search(PATH_RE, p)
            if not path:
                continue
            path = path.group(1)
            parts = path.split("->")
            if len(parts) % 2 != 1:
                continue
            triples = set()
            for i in range(0, len(parts)-2, 2):
                subject = parts[i].strip()
                relation = parts[i+1].strip()
                obj = parts[i+2].strip()
                triples.add((subject, relation, obj))
            results.append(list(triples))
    
    return results


def generate_seq(
    model, input_texts, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100
):
    # tokenize the question
    inputs = tokenizer(input_texts, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    batch_size = input_ids.shape[0]
    # generate sequences
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    predictions = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    predictions = [p.strip() for p in predictions]
    predictions = [predictions[i*num_beam:(i+1)*num_beam] for i in range(batch_size)]

    if num_beam > 1:
        scores = output.sequences_scores.reshape([batch_size, -1]).tolist()
        norm_scores = torch.softmax(output.sequences_scores.reshape([batch_size, -1]), dim=-1).tolist()
    else:
        scores = [[1] for _ in range(batch_size)]
        norm_scores = [[1] for _ in range(batch_size)]

    return [{"paths": predictions[i], "scores": scores[i], "norm_scores": norm_scores[i]}
            for i in range(batch_size)]


def gen_prediction(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    input_file = os.path.join(args.data_path, args.dataset)
    output_dir = os.path.join(args.output_path, args.dataset, args.split, args.model_name, f"type_{args.path_type}")
    logger.info(f"Save results to: {output_dir}")

    # Load dataset
    dataset = load_dataset(input_file, split=args.split)

    # Load prompt template
    prompter = utils.InstructFormater(args.prompt_path)

    def prepare_dataset(sample):
        # Prepare input prompt
        question = sample["question"] if sample["question"].endswith("?") else sample["question"] + "?"

        # predict relation path
        if args.path_type == "relation":
            sample["text"] = prompter.format(
                system=INSTRUCTION_RELATION, query="**Question**:\n" + question)

        #  predict triple path
        elif args.path_type == "triple":    
            sample["text"] = prompter.format(
                system=INSTRUCTION_TRIPLE, query="**Question**:\n" + question)
        return sample

    dataset = dataset.map(
        prepare_dataset,
        num_proc=N_CPUS,
    )

    # Predict
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_file = os.path.join(
        output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl"
    )
    f, processed_results = get_output_file(prediction_file, force=args.force)
    filter_data = [data for data in dataset if data["id"] not in processed_results]

    logger.info(f"filtered {len(dataset) - len(filter_data)} processed samples")

    for batch_idx in tqdm(range(0, len(filter_data), args.batch_size), desc="Generating predict paths"):
        batch = filter_data[batch_idx:batch_idx+args.batch_size]
        batch_inputs = [d["text"] for d in batch]

        raw_outputs = generate_seq(
            model,
            batch_inputs,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            num_beam=args.n_beam,
            do_sample=args.do_sample,
        )

        for i in range(len(batch_inputs)):
            if args.path_type == "relation":
                parsed_paths = parse_relation_prediction(raw_outputs[i]["paths"])

            elif args.path_type == "triple":
                parsed_paths = parse_triple_prediction(raw_outputs[i]["paths"])

            output_data = {
                "id": batch[i]["id"],
                "question": batch[i]["question"],
                "answer": batch[i]["answer"],
                "prediction_paths": parsed_paths,
                "ground_paths": batch[i]["ground_paths"],
                "input": batch[i]["text"],
                "raw_output": raw_outputs[i],
            }

            f.write(json.dumps(output_data) + "\n")
            f.flush()
    f.close()

    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="data"
    )
    parser.add_argument("--dataset", "-dataset", type=str, default="webqsp")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--path_type", type=str, default="relation", help="relation or triple")
    parser.add_argument("--output_path", type=str, default="results/gen_predict_path")

    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="KG-TRACES",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default="model/KG-TRACES",
    )

    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/qwen2.5.txt"
    )
    
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--n_beam", type=int, default=3)
    parser.add_argument("--do_sample", action="store_true", help="do sampling")

    args = parser.parse_args()
    
    logger.info(args)

    gen_path = gen_prediction(args)