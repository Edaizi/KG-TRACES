import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
import random
from typing import Callable

class PromptBuilder(object):
    PROMPT_TEMPLATE = """{instruction}\n{input}"""
    MCQ_INSTRUCTION = """Please answer the following questions. Please select the answers from the given choices and return the answer only."""
    SAQ_INSTRUCTION = """Please answer the following questions. Please keep the answer as simple as possible and return all the possible answer as a list."""

    MCQ_PATH_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please select the answers from the given choices and return the answers only."""
    SAQ_PATH_INSTRUCTION = """Based on the reasoning paths, please answer the given question. Please generate both the reasoning process and answer. Please keep the answer as simple as possible and return all the possible answers as a list."""

    SAQ_PRED_PATH_INSTRUCTION = """Based on the reasoning paths, please answer the given question. The paths with <KG> are from the real world **Knowledge Graph** (more reliable) and the paths with <INFERRED> are your predictions, you should recognize useful reasoning paths from **Potential Useful Reasoning Paths**. Please generate both the reasoning process and answer. Please keep the answer as simple as possible and return all the possible answers as a list."""


    COT = """ Let's think it step by step."""
    EXPLAIN = """ Please explain your answer."""
    QUESTION = """**Question**:\n{question}"""
    GRAPH_CONTEXT = """**Potential Useful Reasoning Paths**:\n{context}\n\n"""
    CHOICES = """\nChoices:\n{choices}"""
    EACH_LINE = """ Please return each answer in a new line."""
    def __init__(self, add_path = False, use_true = False, use_weight=False, use_random = False, use_pred_path = False, pred_path_type = "relation", each_line = False, cot = False, explain = False, path_num=3, path_len=2, maximun_token = 4096, tokenize: Callable = lambda x: len(x)):
        self.prompt_template = self.PROMPT_TEMPLATE
        self.add_path = add_path
        self.use_true = use_true
        self.use_random = use_random
        self.use_weight = use_weight
        self.use_pred_path = use_pred_path
        self.pred_path_type = pred_path_type
        self.cot = cot
        self.explain = explain
        self.maximun_token = maximun_token
        self.tokenize = tokenize
        self.each_line = each_line
        self.path_num = path_num
        self.path_len = path_len
        
    def apply_rules(self, graph, rules, source_entities):
        results = []
        for entity in source_entities:
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

        if self.add_path:
            graph = utils.build_graph(question_dict['graph'])
            entities = question_dict['q_entity'] if 'q_entity' in question_dict else question_dict['question']
            if self.use_true:
                rules = question_dict['ground_paths']
                reasoning_paths = []
            elif self.use_random:
                reasoning_paths, rules = utils.get_random_paths(entities, graph, n=self.path_num, hop=self.path_len)
            elif self.use_weight:
                reasoning_paths, scores, rules =  utils.get_weight_paths(entities, graph, top_k=self.path_num, hop=self.path_len)
            elif self.use_pred_path and self.pred_path_type == "relation":
                rules = question_dict['predicted_relation_paths'] 
                reasoning_paths = []
            elif self.use_pred_path and self.pred_path_type == "triple":
                rules = None
                reasoning_paths = question_dict['predicted_triple_paths']
            elif self.use_pred_path and self.pred_path_type == "relation_triple":
                rules = None 
                reasoning_paths = None

            lists_of_paths = []
            if reasoning_paths is not None and len(reasoning_paths) > 0: 
                if self.use_pred_path and self.pred_path_type == "triple":
                    lists_of_paths = [utils.predict_triple_path_to_string(p) for p in reasoning_paths]
                else:
                    lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]

            elif rules is not None and len(rules) > 0:             
                reasoning_paths = self.apply_rules(graph, rules, entities)
                if self.use_pred_path and self.pred_path_type == "relation":
                    lists_of_paths = [utils.predict_relation_path_to_string(p) for p in reasoning_paths]
                else:
                    lists_of_paths = [utils.path_to_string(p) for p in reasoning_paths]
            elif self.use_pred_path and self.pred_path_type == "relation_triple":
                relation_paths = self.apply_rules(graph, question_dict['predicted_relation_paths'], entities)
                triple_paths = question_dict['predicted_triple_paths']
                lists_of_relation_paths = [utils.predict_relation_path_to_string(p) for p in relation_paths]
                lists_of_triple_paths = [utils.predict_triple_path_to_string(p) for p in triple_paths]
                lists_of_paths = lists_of_relation_paths + lists_of_triple_paths
            
        input = self.QUESTION.format(question = question)
        # MCQ
        if len(question_dict['choices']) > 0:
            choices = '\n'.join(question_dict['choices'])
            input += self.CHOICES.format(choices = choices)
            if self.add_path:
                instruction = self.MCQ_PATH_INSTRUCTION
            else:
                instruction = self.MCQ_INSTRUCTION
        # SAQ
        else:
            if self.add_path:
                if self.use_pred_path:
                    instruction = self.SAQ_PRED_PATH_INSTRUCTION
                else:
                    instruction = self.SAQ_PATH_INSTRUCTION
            else:
                instruction = self.SAQ_INSTRUCTION
        
        if self.cot:
            instruction += self.COT
        
        if self.explain:
            instruction += self.EXPLAIN
            
        if self.each_line:
            instruction += self.EACH_LINE
        
        if self.add_path:
            other_prompt = self.prompt_template.format(instruction = instruction, input = self.GRAPH_CONTEXT.format(context = "") + input)
            context = self.check_prompt_length(other_prompt, lists_of_paths, self.maximun_token)
            
            input = self.GRAPH_CONTEXT.format(context = context) + input
        
        input = self.prompt_template.format(instruction = instruction, input = input)
            
        return input, lists_of_paths if self.add_path else input, []
    
    def check_prompt_length(self, prompt:str, list_of_paths:list[str], maximun_token:int):
        '''Check whether the input prompt is too long. If it is too long, remove the first path and check again.'''
        all_paths = "\n".join(list_of_paths)
        all_tokens = prompt + all_paths
        if self.tokenize(all_tokens) < maximun_token:
            return all_paths
        else:
            # Shuffle the paths
            random.shuffle(list_of_paths)
            new_list_of_paths = []
            # check the length of the prompt
            for p in list_of_paths:
                tmp_all_paths = "\n".join(new_list_of_paths + [p])
                tmp_all_tokens = prompt + tmp_all_paths
                if self.tokenize(tmp_all_tokens) > maximun_token:
                    return "\n".join(new_list_of_paths)
                new_list_of_paths.append(p)
                
            return "\n".join(new_list_of_paths)
            