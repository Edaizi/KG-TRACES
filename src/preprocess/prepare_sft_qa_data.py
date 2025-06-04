import os
import sys
from tqdm import tqdm
from enum import Enum
from typing import List, Dict
from transformers import AutoTokenizer
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import utils

from tools.io_file import read, write
from tools.logger_factory import setup_logger

logger = setup_logger("prepare_sft_data")

class DataType(Enum):
    FULL = "full"          
    PROCESS_ONLY = "reasoning_process"    
    PATH_ONLY = "path"      
    QA_ONLY = "qa"            

def apply_rules(graph, rules, source_entities):
    results = []
    for entity in source_entities:
        for rule in rules:
            res = utils.bfs_with_rule(graph, entity, rule)
            results.extend(res)
    return results

class SFTDataGenerator:
    def __init__(self, 
                 model_name: str = "models/Qwen2.5-7b-instruct",
                 prompt_template: str = "prompts/qwen2.5.txt",
                 max_token_length: int = 2048,
                 min_paths: int = 3):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False
        )
        self.prompter = utils.InstructFormater(prompt_template)
        self.max_token_length = max_token_length
        self.instruction = {"full":(
            "Based on the reasoning paths, please answer the given question. "
            "Please generate both the reasoning process and answer. Keep the answer "
            "as simple as possible and return all the possible answers as a list.\n"),

            "reasoning_process": (
                "Please answer the given question. "
                "Please generate both the reasoning process and answer. Keep the answer "
                "as simple as possible and return all the possible answers as a list.\n"),
            
            "path": (
                "Based on the reasoning paths, please answer the given question. "
                "Keep the answer as simple as possible and return all the possible answers as a list.\n"),
        
            "qa":(
                "Please answer the given question. "
                "Keep the answer as simple as possible and return all the possible answers as a list.\n"
                )
            }
        self.min_paths = min_paths

    def _check_length(self, components: Dict[str, str]) -> str:
        full_text = "".join(components.values())
        tokens = self.tokenizer.tokenize(full_text)
        
        if len(tokens) > self.max_token_length and "paths" in components:
            remain_tokens = self.max_token_length - len(
                self.tokenizer.tokenize(
                    "".join([v for k,v in components.items() if k != "paths"])
                )
            )
            components["paths"] = self._truncate_paths(
                components["paths"], 
                max_token_allowance=remain_tokens
            )
        return components

    def _truncate_paths(self, paths: str, max_token_allowance: int) -> str:
        path_list = paths.split("\n")
        
        path_list.sort(key=lambda x: len(self.tokenizer.tokenize(x)))
        preserved_paths = path_list[:self.min_paths]
        candidate_paths = path_list[self.min_paths:]
        
        random.shuffle(candidate_paths)
        
        current_tokens = sum(
            len(self.tokenizer.tokenize(p)) 
            for p in preserved_paths
        )
        selected_paths = []
        
        for path in candidate_paths:
            path_tokens = len(self.tokenizer.tokenize(path))
            if current_tokens + path_tokens <= max_token_allowance:
                selected_paths.append(path)
                current_tokens += path_tokens
        
        final_paths = preserved_paths + selected_paths
        
        if len(final_paths) == 0:
            return ""
            
        while True:
            total_tokens = sum(
                len(self.tokenizer.tokenize(p)) 
                for p in final_paths
            )
            if total_tokens <= max_token_allowance or len(final_paths) <= self.min_paths:
                break
            final_paths.pop(
                max(range(len(final_paths)), 
                key=lambda i: len(self.tokenizer.tokenize(final_paths[i])))
            )
            
        return "\n".join(final_paths)



    def generate_entry(self,
                 idx: int,
                 question: str,
                 answer: str,
                 reasoning_paths: List[str],
                 reasoning_process: str,
                 data_type: DataType) -> Dict:
        components = {
            "instruction": "",
            "question_section": "\n**Question**:\n",
            "question_content": f"{question}",
            "paths_section": "",
            "paths_content": "",
            "process_section": "",
            "answer_section": "\n**Answer**:\n",
            "answer_content": f"{answer}"
        }

        if data_type in [DataType.FULL, DataType.PATH_ONLY] and reasoning_paths:
            components["paths_section"] = "\n**Potential Useful Reasoning Paths**:\n"
            components["paths_content"] = "\n".join(reasoning_paths)
            
        if data_type in [DataType.FULL, DataType.PROCESS_ONLY] and reasoning_process:
            components["process_section"] = reasoning_process
        
        if reasoning_paths and reasoning_process:
            components["instruction"] = f"{self.instruction[data_type.value]}\n"
        elif reasoning_process and not reasoning_paths:
            components["instruction"] = f"{self.instruction['reasoning_process']}"
        elif reasoning_paths and not reasoning_process:
            components["instruction"] = f"{self.instruction['path']}\n"
        elif not reasoning_paths and not reasoning_process:
            components["instruction"] = f"{self.instruction['qa']}\n"

        if data_type == DataType.QA_ONLY:
            components["paths_section"] = ""
            components["paths_content"] = ""
            components["process_section"] = ""
        
        components = self._check_length(components)

        
        text_parts = {
        "instruction": components["instruction"],
        "question": components["question_section"] + components["question_content"],
        "answer": components["answer_section"] + components["answer_content"]
        }

  
        text_parts["paths"] = components["paths_section"] + components["paths_content"] if components["paths_content"] else ""
        text_parts["process"] = components["process_section"] if components["process_section"] else ""

        system_prompt = "\n".join(line for k, part in text_parts.items() if k in ["instruction"]  for line in part.split("\n") if line.strip())
        query_prompt = "\n".join(line for k, part in text_parts.items() if k in ["question", "paths"]  for line in part.split("\n") if line.strip())
        assitant_prompt = "\n".join([line for k, part in text_parts.items() if k == "process"  for line in part.split("\n") if line.strip()] + [line for k, part in text_parts.items() if k == "answer"  for line in part.split("\n") if line.strip()]) 

        system_user_prompt = self.prompter.format(system=system_prompt, query=query_prompt)

        full_prompt = system_user_prompt + assitant_prompt + self.tokenizer.eos_token
        
        return {
            "id": idx,
            "text": full_prompt,
            "token_len": len(self.tokenizer.tokenize(full_prompt))
        }


def generate_sft_data(
    origin_path: str,
    reasoning_process_path: str,
    output_path: str,
    predict_triple_path_path: str,
    data_type: DataType = DataType.FULL,
    model_path: str = "models/Qwen2.5-7b-instruct",
    prompt_template: str = "prompts/qwen2.5.txt",
    max_token_length: int = 2048
):


    generator = SFTDataGenerator(model_name = model_path, prompt_template=prompt_template, max_token_length=max_token_length)
    origin_data = read(origin_path)
    reasoning_data = read(reasoning_process_path)
    predict_triple_path_data = read(predict_triple_path_path)

    question_to_triple_path = dict()
    for data in predict_triple_path_data:
        qid = data["id"]
        predicted_paths = data["prediction_paths"]
        ground_paths = data["ground_paths"]
        question_to_triple_path[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }
    
    sft_data = []
    stats = {"total": 0, "with_process": 0, "with_path": 0}

    for idx, (orig_item, reason_item) in tqdm(enumerate(zip(origin_data, reasoning_data)), total=len(origin_data)):
        question = orig_item["question"].rstrip("?") + "?"
        answer = "\n".join(orig_item["answer"])
        q_id = orig_item["id"].strip()
        
        if data_type not in [DataType.QA_ONLY, DataType.PROCESS_ONLY]:
            graph = utils.build_graph(orig_item['graph'])
            gt_paths = apply_rules(graph, orig_item["ground_paths"], orig_item["q_entity"])
            list_of_gt_paths = [utils.predict_relation_path_to_string(p) for p in gt_paths]
            predict_triple_paths = question_to_triple_path[q_id]["predicted_paths"]
            list_of_predict_paths = [utils.predict_triple_path_to_string(p) for p in predict_triple_paths]
            reasoning_paths = list_of_gt_paths + list_of_predict_paths
            random.shuffle(reasoning_paths)
        else:
            reasoning_paths = []
        
        if data_type not in [ DataType.QA_ONLY, DataType.PATH_ONLY]:
            has_valid_reasoning = reason_item["question"] not in ("", None) and reason_item["answer"]  not in ("", None)
            
            reasoning_process = reason_item["answer"] if has_valid_reasoning else ""
            if has_valid_reasoning:
                if reasoning_process.strip().lower().startswith("### output:"):
                    reasoning_process = reasoning_process.split("### Output:")[1].strip().strip("\n\n").strip("\n")
                if not reasoning_process.strip().lower().startswith("**reasoning process**:"):
                    reasoning_process = "**Reasoning Process**:\n" + reasoning_process
        
        else:
            has_valid_reasoning = False
            reasoning_process = ""
        
        entry = generator.generate_entry(
            idx=idx,
            question=question,
            answer=answer,
            reasoning_paths=reasoning_paths,
            reasoning_process=reasoning_process if has_valid_reasoning else "",
            data_type=data_type
        )
        
        stats["total"] += 1
        stats["with_process"] += 1 if has_valid_reasoning else 0
        stats["with_path"] += 1 if reasoning_paths else 0

        sft_data.append(entry)

    write(output_path, sft_data, mode="w")

if __name__ == "__main__":
    data_type = DataType.PROCESS_ONLY
    dataset = "webqsp"
    prompt_name = "qwen_2.5"
    model_path = "models/Qwen2.5-7b-Instruct"
    prompt_template = "prompts/qwen2.5.txt"
    max_token_length =4096 

    predict_triple_path_path = "results/gen_predict_path/cwq/KG-TRACES/type_triple/predictions_3_False.jsonl"

    generate_sft_data(
        origin_path=f"data/{dataset}/train.jsonl",
        reasoning_process_path=f"data/qa_aug/{dataset}/train.jsonl",
        output_path=f"data/qa_aug/{dataset}/{dataset}-sft_aug_train-{data_type.value}-{prompt_name}.jsonl",
        predict_triple_path_path=predict_triple_path_path,
        data_type=data_type,
        model_path=model_path,
        prompt_template=prompt_template,
        max_token_length=max_token_length
    )