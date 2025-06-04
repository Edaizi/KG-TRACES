import sys
import os
from typing import Callable
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from utils import *
from transformers import AutoTokenizer
import datasets

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  
p_project_root = os.path.dirname(project_root) 
sys.path.extend([project_root,p_project_root])

from tools.logger_factory import setup_logger

logger = setup_logger("build_process_dataset")


class PromptBuilder(object):
    
    BACKGROUND = """"""

    QUESTION = """### Question:\n{question}\n"""
    ANSWER = """### Answer:\n{answer}\n"""
    
    GRAPH_CONTEXT = """
### Potential useful reasoning path:
The following reasoning paths are provided to help you understand relationships among entities and derive an answer:\n{reasoning_paths_text}
"""
    INSTRUCTIONS = """
### Task Instructions:
1. **Goal**: 
    - Use the given reasoning paths and answer to generate a detailed reasoning process for the original question, explicitly indicating the source of knowledge (e.g., from KG or inferred by LLM).
    - Enhance the reasoning process by including special tokens to label each path's source and effectiveness:
        - <KG>: Knowledge directly from the knowledge graph.
        - <INFERRED>: Knowledge inferred by LLM without explicit KG support.
        - <EFFECTIVE> / <INEFFECTIVE>: Indicate whether the path effectively contributes to the final answer.
2. **Specific Requirements**:
    - Path Selection and Labeling:
        - Filter out unnecessary paths: Only select paths directly relevant to the question. Ignore paths marked as <INEFFECTIVE> when get final answer.
        - Label each selected path using the special tokens (<KG>, <INFERRED>, <EFFECTIVE>, <INEFFECTIVE>).
    Dynamic Knowledge Utilization:
        - If no path from the knowledge graph directly applies, allow LLM to infer logical connections using the <INFERRED> token, but clearly mark them as such.
3. **Output Format**:
    **Reasoning Process**:\n[Output reasoning process here]\n
"""

    EXAMPLES = """
### Example:
[Input]:
**Question**: Which film directed by Christopher Nolan starred Leonardo DiCaprio and was released in 2010?
**Answer**: Inception

**Potential useful reasoning path**:
1. Leonardo DiCaprio -> film.actor.film -> m.12345
2. m.12345 -> film.director -> Christopher Nolan
3. Christopher Nolan -> film.director_of -> m.67890
4. m.12345 -> film.release_date -> 2010
5. m.67890 -> film.release_date -> 2017
6. m.12345 -> film.name -> Inception
7. Leonardo DiCaprio -> person.awards -> Academy Award
8. Christopher Nolan -> person.birthplace -> London
9. Inception -> film.budget -> $160 million
10. Leonardo DiCaprio -> film.actor.film -> m.99999
11. m.99999 -> film.name -> Titanic
12. m.99999 -> film.release_date -> 1997
13. Leonardo DiCaprio -> film.actor.film -> m.88888
14. m.88888 -> film.name -> The Wolf of Wall Street
15. m.88888 -> film.release_date -> 2013
16. Christopher Nolan -> spouse -> Emma Thomas
17. Christopher Nolan -> film.director_of -> m.00000
18. m.00000 -> film.name -> Dunkirk
19. m.00000 -> film.release_date -> 2017

[Output]
**Reasoning Process**:
Step 1: Identify the film in which Leonardo DiCaprio starred.
- Relevant Triple: #1 (Leonardo DiCaprio -> film.actor.film -> m.12345) [<KG> <EFFECTIVE>]
- Note: Triples #10 and #13 also indicate films Leonardo DiCaprio acted in (m.99999 and m.88888), but they do not satisfy the release year requirement; hence, they are marked [<KG> <INEFFECTIVE>].
- Conclusion: Select film m.12345 for further analysis.
Step 2: Verify that m.12345 was directed by Christopher Nolan.
- Relevant Triple: #2 (m.12345 -> film.director -> Christopher Nolan) [<KG> <EFFECTIVE>]
- Note: Triple #3 (Christopher Nolan -> film.director_of -> m.67890) and Triple #17 (Christopher Nolan -> film.director_of -> m.00000) pertain to other films and are marked [<KG> <INEFFECTIVE>].
- Conclusion: m.12345 is confirmed to be directed by Christopher Nolan.
Step 3: Confirm the release year of m.12345.
- Relevant Triple: #4 (m.12345 -> film.release_date -> 2010) [<KG> <EFFECTIVE>]
- Note: Triple #5 indicates m.67890 was released in 2017 and is thus [<KG> <INEFFECTIVE>].
- Conclusion: m.12345 was released in 2010.
Step 4: Verify the film name.
- Relevant Triple: #6 (m.12345 -> film.name -> Inception) [<KG> <EFFECTIVE>]
- Note: Triples #11/#12 (Titanic) and #14/#15 (The Wolf of Wall Street) do not match; they are marked [<KG> <INEFFECTIVE>].
- Conclusion: m.12345 corresponds to the film "Inception."
Step 5: Additional Irrelevant Information.
- Note: Triples such as #7 (Leonardo DiCaprio -> person.awards -> Academy Award), #8 (Christopher Nolan -> person.birthplace -> London), #9 (Inception -> film.budget -> $160 million), and #16 (Christopher Nolan -> spouse -> Emma Thomas) provide extra data but do not contribute to the final answer [<KG> <INEFFECTIVE>].
- Conclusion: These are filtered out.

Final Inference:
Combining the effective reasoning steps, an inferred conclusion [<INFERRED> <EFFECTIVE>] confirms that the film meeting all criteria is "Inception."
Final Answer: Inception
"""

    def __init__(self, prompt_path, add_rule = False, use_true = False,use_random = False, maximun_token = 4096, max_answer_length=2048, tokenizer: Callable = None):
        self.prompt_template = self._read_prompt_template(prompt_path)
        self.add_rule = add_rule
        self.use_true = use_true
        self.use_random = use_random
        self.maximun_token = maximun_token
        self.max_answer_length = max_answer_length 
        self.tokenize = lambda x: tokenizer.encode(x, add_special_tokens=True)
        self.detokenize = lambda x: tokenizer.decode(x)
        
    def _read_prompt_template(self, template_file):
        with open(template_file) as fin:
            prompt_template = f"""{fin.read()}"""
        return prompt_template
    
    def apply_rules(self, graph, rules, srouce_entities):
        results = []
        for entity in srouce_entities:
            for rule in rules:
                res = utils.bfs_with_rule(graph, entity, rule)
                results.extend(res)
        return results
    
    def direct_answer(self, question_dict):
        graph = utils.build_graph(question_dict['graph'])
        entities = question_dict['q_entity']
        rules = question_dict['predicted_paths']
        prediction = []
        if len(rules) > 0:
            reasoning_paths = self.apply_rules(graph, rules, entities)
            for p in reasoning_paths:
                if len(p) > 0:
                    prediction.append(p[-1][-1])
        return prediction
    
    
    def process_input(self, question_dict):
        '''
        Take question as input and return the input with prompt
        '''
        question = question_dict['question']
        
        if not question.endswith('?'):
            question += '?'
        
        if self.add_rule:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity']
            if self.use_true:
                rules = question_dict['ground_paths']
            elif self.use_random:
                _, rules = utils.get_random_paths(entities, graph)
            else:
                rules = question_dict['predicted_paths']
            if rules is not None and len(rules) > 0:
                reasoning_paths = self.apply_rules(graph, rules, entities)
                lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
                
            else:
                lists_of_paths = []
            
        answer = "\n".join(question_dict['a_entity'])
        tokenized_answer = self.tokenize(answer)
        if len(tokenized_answer) > self.max_answer_length:
            tokenized_answer = tokenized_answer[:self.max_answer_length]
            answer = self.detokenize(tokenized_answer)

        input = self.BACKGROUND + self.QUESTION.format(question = question) + self.ANSWER.format(answer=answer)

        if self.add_rule:
            if lists_of_paths:
                reasoning_path = self.check_prompt_length(input, lists_of_paths, self.maximun_token)
                input = input + self.GRAPH_CONTEXT.format(reasoning_paths_text=reasoning_path) + self.INSTRUCTIONS + self.EXAMPLES
            else:
                input = ""
            
        return input, len(self.tokenize(input))
    
    def check_prompt_length(self, prompt, list_of_paths, maximun_token, truncate=False):
        # Shuffle the paths to avoid bias in path selection
        random.shuffle(list_of_paths)
        
        # Step 1: Precompute the token lengths for each path
        path_lengths = [(path, len(self.tokenize(path))) for path in list_of_paths]

        # Sort paths based on their token lengths (ascending)
        path_lengths.sort(key=lambda x: x[1])

        # Step 2: Compute the token count for the prompt itself
        prompt_length = len(self.tokenize(prompt))

        # Step 3: Iteratively add paths while ensuring the total token count does not exceed maximun_token
        selected_paths = []
        total_length = prompt_length

        for path, length in path_lengths:
            if total_length + length > maximun_token:
                remaining_space = maximun_token - total_length
                if truncate and remaining_space > 0:
                    # Truncate the path to fit within the remaining token limit
                    truncated_path = self.tokenize(path)[:remaining_space]
                    selected_paths.append(self.detokenize(truncated_path))
                # If not truncating, just skip the path entirely
                break  # Once we reach the limit, stop adding more paths
            else:
                # If adding the full path does not exceed the limit, add it
                selected_paths.append(path)
                total_length += length

        return "\n".join(selected_paths)



N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1
logger.info(f"N_CPUS: {N_CPUS}\n")

save_dir = "data"
prompt_path = "prompts/qwen2.5.txt"
split="train"
model_max_length = 36*1024  # 36 k
data_list = ['webqsp', 'cwq']
data_path = "data"

model_name_or_path = "models/Qwen2.5-7b-instruct"

prompter = InstructFormater(prompt_path)

tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

# Load prompt template
input_builder = PromptBuilder(
        prompt_path,
        add_rule = True,
        use_true= True,
        maximun_token=model_max_length,
        tokenizer=tokenizer,
    )

def formatting_prompts_func(example, idx):
    answer = example["answer"]
    ground_paths = example["ground_paths"]
    
    output_text, token_num = input_builder.process_input(example)
    return {"id": idx, "question": output_text, "token_num":token_num, "answer":answer}

for data_name in data_list:
    input_file = os.path.join(data_path, data_name)
    train_dataset = datasets.load_dataset(input_file, split="train")
    data_name = data_name.split("-")[-1]
    save_path = os.path.join(save_dir, data_name, data_name + "_to_aug_train.jsonl")

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_dataset = train_dataset.map(
        formatting_prompts_func,
        remove_columns=train_dataset.column_names,
        with_indices=True,
        num_proc=N_CPUS,

    )
    
    train_dataset.to_json(save_path, orient="records", lines=True)
    logger.info(f"\nSaved processed data to: {save_path}")

