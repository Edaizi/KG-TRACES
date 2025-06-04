import torch
from .base_language_model import BaseLanguageModel
import os
import dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

from tools.logger_factory import setup_logger

logger = setup_logger("base_hf_causal_model")

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


class HfCausalModel(BaseLanguageModel):
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    @staticmethod
    def add_args(parser):
        parser.add_argument("--maximun_token", type=int, help="max input token", default=4096)
        parser.add_argument("--max_output_tokens", type=int, help="max output token", default=786)

        parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
        parser.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
        parser.add_argument(
            "--attn_implementation",
            default="flash_attention_2",
            choices=["eager", "sdpa", "flash_attention_2"],
            help="enable flash attention 2",
        )
        parser.add_argument(
            "--generation_mode",
            type=str,
            default="greedy",
            choices=["greedy", "beam", "sampling", "group-beam", "beam-early-stopping", "group-beam-early-stopping"],
        )
        parser.add_argument(
            "--k", type=int, default=1, help="number of paths to generate"
        )
        parser.add_argument("--temperature", type=float, default=0.7)
        parser.add_argument("--top_p", type=float, default=0.85)
        parser.add_argument("--top_k", type=int, default=20)
        parser.add_argument("--repetition_penalty", type=float, default=1.05)
        parser.add_argument("--chat_model", default='true', type=lambda x: (str(x).lower() == 'true'))

    def __init__(self, args):
        self.args = args
        self.maximun_token = args.maximun_token
        self.model_path = args.model_path

    def token_len(self, text):
        return len(self.tokenizer.tokenize(text))

    def prepare_for_inference(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.args.quant != "none":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.args.quant == "8bit",  
                load_in_4bit=self.args.quant == "4bit",
                bnb_4bit_quant_type="nf4",          
                bnb_4bit_use_double_quant=True      
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            device_map="auto",
            torch_dtype=self.DTYPE.get(self.args.dtype, None),
            quantization_config=quantization_config if self.args.quant != "none" else None,
            attn_implementation=self.args.attn_implementation,
        )


        self.generation_cfg = GenerationConfig.from_pretrained(self.args.model_path)
            
        self.generation_cfg.max_new_tokens = self.args.max_output_tokens
        self.generation_cfg.temperature = self.args.temperature
        self.generation_cfg.top_k = self.args.top_k
        self.generation_cfg.top_p = self.args.top_p
        self.generation_cfg.repetition_penalty = self.args.repetition_penalty



        if self.args.generation_mode == "greedy":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_return_sequences = 1
        elif self.args.generation_mode == "sampling":
            self.generation_cfg.do_sample = True
            self.generation_cfg.num_return_sequences = self.args.k
        elif self.args.generation_mode == "beam":
            self.generation_cfg.do_sample = False
            self.generation_cfg.num_beams = self.args.k
            self.generation_cfg.num_return_sequences = self.args.k

    def prepare_model_prompt(self, query):
        if self.args.chat_model:
            chat_query = [
                {"role": "user", "content": query}
            ]
            return self.tokenizer.apply_chat_template(chat_query, tokenize=False, add_generation_prompt=True)
        else:
            return query
    
    @torch.inference_mode()
    def generate_sentence_batch(self, llm_input, *args, **kwargs) -> list[list[str]]:
        new_llm_input = []
        for input in llm_input:
            new_llm_input.append(self.prepare_model_prompt(input))

        if isinstance(new_llm_input, list):
            inputs = self.tokenizer(new_llm_input, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
        else:
            inputs = self.tokenizer([new_llm_input], return_tensors="pt",padding=True, truncation=True,).to(self.model.device)

        input_ids = inputs.input_ids

        generated_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_cfg,
            )

        responses = []

        for batch_idx in range(input_ids.shape[0]):
            batch_responses = []
            for seq_idx in range(self.args.k):
                sequence = generated_ids[batch_idx *self.args.k + seq_idx] 
                response_len = len(input_ids[batch_idx])
                decoded = "".join(self.tokenizer.batch_decode(sequence[response_len:], skip_special_tokens=self.args.skip_special_tokens))
                batch_responses.append(decoded.strip()) 
            responses.append(batch_responses)

        return responses

