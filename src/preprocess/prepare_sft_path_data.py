import sys
import os
from transformers import AutoTokenizer
import datasets

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from tools.logger_factory import setup_logger

logger = setup_logger("preprocess_predict_path_sft")

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1


predict_type = "triple"  ## "triple" or "relation"
save_dir = "data/qa_aug/predict_path"
prompt_path = "prompts/qwen2.5.txt"
data_template = "data/qa_aug/predict_path/{}/{}_train_{}.jsonl"
data_list = ['webqsp', 'cwq']
model_name_or_path = "hug_ckpts/Qwen2.5-7b-instruct"
model_name = "qwen_2.5"
prompter = InstructFormater(prompt_path)

INSTRUCTION = f"""Please generate a valid reasoning {predict_type} path that can be helpful for answering the following question: """
SEP = '<SEP>'
BOP = '<PATH>'
EOP = '</PATH>'


tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

def formatting_prompts_func(example):
    if predict_type == "relation":
        output_label = rule_to_string(example["path"], sep_token=SEP, bop=BOP, eop=EOP)
    else:
        output_label = path_to_string(example["path"], bop=BOP, eop=EOP)
    
    question = example["question"].rstrip("?") + " ?"
    output_text = (
            prompter.format(system=INSTRUCTION.format(predict_type=predict_type), query="**Question**:\n" + question)
            + " "
            + output_label + tokenizer.eos_token
        )
    return {"text": output_text, "token_num":len(tokenizer.tokenize(output_text))}

for data_name in data_list:
    data_path = data_template.format(data_name, data_name, predict_type)
    logger.info(f"Loading data from {data_path}")
    save_path = os.path.join(save_dir, data_name, data_name + f"_sft_train-{model_name}-{predict_type}.jsonl")
    train_dataset = datasets.load_dataset('json', data_files=data_path, split="train")
    
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=["question", "path"],
        num_proc=N_CPUS,
    )
    train_dataset.to_json(save_path, orient="records", lines=True)
    logger.info(f"Save to: {save_path}")
