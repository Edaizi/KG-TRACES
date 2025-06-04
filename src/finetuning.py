import sys
import os
import torch
from dataclasses import dataclass, field
from typing import Optional
import datasets

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from torch.distributed import get_rank
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import AutoPeftModelForCausalLM, LoraConfig

datasets.disable_progress_bar()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
p_project_root = os.path.dirname(project_root) 
sys.path.extend([project_root, p_project_root])

from utils import *
from tools.logger_factory import setup_logger
from data_loader import load_multiple_datasets


logger = setup_logger("finetuning")


@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(
        metadata={"help": "Path to the training data."}
    )
    eval_data_path_list: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Path to the evaluation data."}
    )
    model_name_or_path: Optional[str] = field(
        default="model/Meta-Llama-3___2-1B-Instruct", metadata={"help": "the model name"}
    )

    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the lora r parameter"}
    )



@dataclass
class CustomSFTConfig(SFTConfig):
    output_dir: str = field(
        default="saved_models/llama_sft",
        metadata={"help": "The output directory where the model checkpoints will be saved."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use for training."},
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use sequence packing when training."},
    )
    dataset_num_proc: int = field(
        default=1,
        metadata={"help": "Number of processes to use for dataset loading and processing."},
    )
    dataset_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for dataset loading and processing."},
    )
    dataset_kwargs: dict = field(
        default_factory=lambda: {},
        metadata={"help": "Additional keyword arguments for dataset loading."},
    )
    num_of_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences per batch when packing sequences."},
    )

    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Whether to find unused parameters when using Distributed Data Parallel (DDP)."},
    )

    dataset_text_field: str = field(
        default="text",
        metadata={"help": "The text field to use for training."},
    )

    has_kg_kg_weight: float = field(
        default=1.5,
        metadata={"help": "The loss weight of the special token <KG> when KG path is available."},
    )

    has_kg_inferred_weight: float = field(
        default=1.0,
        metadata={"help": "The loss weight of the special token <INFERRED> when KG path is available."},
    )

    has_kg_effective_weight: float = field(
        default=2.0,
        metadata={"help": "The loss weight of the special token <EFFECTIVE> when KG path is available."},
    )
    
    has_kg_ineffective_weight: float = field(
        default=0.5,
        metadata={"help": "The loss weight of the special token <INEFFECTIVE> when KG path is available."},
    )

    no_kg_kg_weight: float = field(
        default=0,
        metadata={"help": "The loss weight of the special token <KG> when KG path is not available."},
    )

    no_kg_inferred_weight: float = field(
        default=2.0,
        metadata={"help": "The loss weight of the special token <INFERRED> when KG path is not available."},
    )

    no_kg_effective_weight: float = field(
        default=2.0,
        metadata={"help": "The loss weight of the special token <EFFECTIVE> when KG path is not available."},
    )

    no_kg_ineffective_weight: float = field(
        default=0.5,
        metadata={"help": "The loss weight of the special token <INEFFECTIVE> when KG path is not available."},
    )




def train():
    parser = HfArgumentParser((ScriptArguments, CustomSFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if get_rank() == 0:
        logger.info(script_args)
        logger.info(training_args)
    
    # Load models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_auth_token=False,
    )
    
    # logger.info(model)

    model.config.use_cache = False
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    new_tokens = ["<KG>", "<INFERRED>", "<EFFECTIVE>", "<INEFFECTIVE>", "<SEP>", "<PATH>", "</PATH>"]        

    num_add_new_token = tokenizer.add_tokens(new_tokens)
    if get_rank() == 0:
        logger.info(f"successfully added {num_add_new_token} new tokens: {new_tokens}")
    model.resize_token_embeddings(len(tokenizer)) 


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_multiple_datasets(script_args.data_path_list, shuffle=True)
    eval_dataset = None
    if script_args.eval_data_path_list:
        eval_dataset = load_multiple_datasets(script_args.eval_data_path_list, shuffle=False)

    if get_rank() == 0:
        logger.info(f"Loaded {len(train_dataset)} training examples.")
        if eval_dataset:
            logger.info(f"Loaded {len(eval_dataset)} evaluation examples.")
    

    # ipdb.set_trace()

    # Prepare instruct tuning
    
    response_template = "<|im_start|>assistant\\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer, mlm=False
    )
    # ipdb.set_trace()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
    )

    # ipdb.set_trace()
    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            if get_rank() == 0:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
            pass
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    trainer.train(resume_from_checkpoint=checkpoint)

    if script_args.use_peft:
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        if script_args.save_merged:
            del model
            torch.cuda.empty_cache()
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir, device_map="auto", torch_dtype=torch.bfloat16
            )
            model = model.merge_and_unload()
            model.eval()
            model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
    
    logger.info(f"Training finished.\nModel saved to: {training_args.output_dir}")



        

if __name__ == "__main__":
    
    train()