#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import numpy as np

import datasets
import torch
import accelerate
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset,Features, Value
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
#from transformers.utils.versions import require_version
from prompter import Prompter
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    PeftModel
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0.dev0")

logger = get_logger(__name__)

#require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        '--peft', 
        default=False, 
        action="store_true", 
        help='use lora finetune or not')
    parser.add_argument(
        '--kg_pretrain', 
        default=False, 
        action="store_true", 
        help='pretraining with kg or not')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--peft_model_path",
        type=str,
        help="Path to peft model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        #choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        '--train_on_inputs', 
        default=False, 
        action="store_true", 
        help='Train on inputs. If False, masks out inputs in loss'
        )
    parser.add_argument(
        '--cutoff_len', 
        type=int, 
        default=1024, 
        help='cutoff length'
        )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        '--lora_r', 
        type=int, 
        default=8, 
        help='lora r'
    )
    parser.add_argument(
        '--lora_alpha', 
        type=int, 
        default=32, 
        help='lora alpha'
    )
    parser.add_argument(
        '--lora_dropout', 
        type=float, 
        default=0.1, 
        help='lora dropout'
    )
    parser.add_argument(
        '--lora_target_modules', 
        type=str, 
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", 
        help='lora target modules'
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # 新增参数：为DeepSeek R1添加
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Whether to trust remote code when loading the model."
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def get_model_target_modules(model_name_or_path, model_config):
    """
    根据模型类型自动确定LoRA target modules
    """
    model_type = getattr(model_config, 'model_type', '').lower()
    
    # DeepSeek R1 基于 Qwen 架构
    if 'deepseek' in model_name_or_path.lower() or 'qwen' in model_type:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # LLaMA/LLaMA2 架构
    elif 'llama' in model_type or 'llama' in model_name_or_path.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # ChatGLM 架构
    elif 'chatglm' in model_type:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    # Baichuan 架构
    elif 'baichuan' in model_type:
        return ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # 默认返回通用的模块
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]


def setup_tokenizer_for_model(tokenizer, model_name_or_path):
    """
    为不同模型设置合适的tokenizer配置
    """
    # DeepSeek R1 特殊处理
    if 'deepseek' in model_name_or_path.lower():
        # DeepSeek R1 通常使用与 Qwen 相似的 tokenizer 设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        # 通用设置
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = 0 if tokenizer.eos_token_id is None else tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    
    return tokenizer


def evaluate(args, accelerator, model, eval_dataloader):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        if args.report_to and args.report_to.lower() != "none":
            accelerator_log_kwargs["log_with"] = args.report_to
            accelerator_log_kwargs["project_dir"] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # ✅ 修复后的数据集加载代码：
    logger.info("Loading datasets...")
    
    if args.validation_file:
        # 当有单独的验证文件时
        logger.info(f"Loading training data from: {args.train_file}")
        logger.info(f"Loading validation data from: {args.validation_file}")
        
        train_dataset_dict = load_dataset('json', data_files=args.train_file)
        validation_dataset_dict = load_dataset('json', data_files=args.validation_file)
        
        raw_datasets = {
            'train': train_dataset_dict['train'],
            'validation': validation_dataset_dict['train']
        }
    else:
        # 当只有训练文件时，需要分割数据
        logger.info(f"Loading data from: {args.train_file}")
        full_dataset_dict = load_dataset('json', data_files=args.train_file)
        full_dataset = full_dataset_dict['train']
        
        # 使用train_test_split进行数据分割
        train_test_split = full_dataset.train_test_split(
            test_size=args.validation_split_percentage/100,
            shuffle=True,
            seed=args.seed if args.seed is not None else 42
        )
        
        raw_datasets = {
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        }
    
    # 打印数据集信息用于验证
    logger.info(f"✅ Dataset loaded successfully:")
    logger.info(f"  Training samples: {len(raw_datasets['train'])}")
    logger.info(f"  Validation samples: {len(raw_datasets['validation'])}")
    
    # 打印几个样本来检查数据格式
    logger.info("Sample training data:")
    for i in range(min(3, len(raw_datasets['train']))):
        sample = raw_datasets['train'][i]
        logger.info(f"Sample {i+1}: {dict(sample)}")
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, 
            trust_remote_code=args.trust_remote_code, 
            use_fast=not args.use_slow_tokenizer
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=args.trust_remote_code, 
            use_fast=not args.use_slow_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # 加载模型配置以确定架构
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
        if args.peft_model_path is not None:
            model = PeftModel.from_pretrained(model, args.peft_model_path)
    
    if args.peft:
        # Prepare For LoRA
        model = prepare_model_for_int8_training(model)
        
        # 自动确定target modules，但如果用户指定了就使用用户的
        if args.lora_target_modules:
            target_modules = [module.strip() for module in args.lora_target_modules.split(',')]
        else:
            target_modules = get_model_target_modules(args.model_name_or_path, model_config)
        
        logger.info(f"Using LoRA target modules: {target_modules}")
        
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,  # 使用动态确定的target modules
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        
        model.print_trainable_parameters()  

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # 根据模型类型设置tokenizer
    tokenizer = setup_tokenizer_for_model(tokenizer, args.model_name_or_path)

    prompter = Prompter()
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    
    def tokenize(prompt, add_eos_token=True):
        """修复后的tokenize函数 - 移除padding，让DataCollator处理"""
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,  # ✅ 改为False，让DataCollator统一处理padding
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        """修复后的数据处理函数"""
        try:
            if args.kg_pretrain:
                full_prompt = data_point['text']
            else:
                # ✅ 确保input和output字段存在且为字符串
                input_text = data_point.get("input", "")
                output_text = data_point.get("output", "")
                
                # ✅ 如果是列表，取第一个元素或转为字符串
                if isinstance(input_text, list):
                    input_text = input_text[0] if input_text else ""
                if isinstance(output_text, list):
                    output_text = output_text[0] if output_text else ""
                    
                full_prompt = prompter.generate_prompt(
                    str(input_text),
                    label=str(output_text),
                )
            
            tokenized_full_prompt = tokenize(full_prompt)
            
            if not args.train_on_inputs and not args.kg_pretrain:
                input_text = data_point.get("input", "")
                if isinstance(input_text, list):
                    input_text = input_text[0] if input_text else ""
                    
                user_prompt = prompter.generate_prompt(str(input_text))
                tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            
            # ✅ 确保返回的数据只包含必要字段
            return {
                "input_ids": tokenized_full_prompt["input_ids"],
                "attention_mask": tokenized_full_prompt["attention_mask"],
                "labels": tokenized_full_prompt["labels"]
            }
            
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
            logger.error(f"Data point keys: {list(data_point.keys()) if isinstance(data_point, dict) else 'Not a dict'}")
            logger.error(f"Data point sample: {str(data_point)[:200]}...")
            # 返回一个默认的空样本
            return {
                "input_ids": [tokenizer.eos_token_id],
                "attention_mask": [1],
                "labels": [tokenizer.eos_token_id]
            }

    # ✅ 修复后的数据集处理
    logger.info("Processing training dataset...")
    train_dataset = raw_datasets["train"].shuffle().map(
        generate_and_tokenize_prompt,
        remove_columns=list(raw_datasets["train"].column_names),  # ✅ 移除所有原始列
        desc="Processing train dataset",
        num_proc=args.preprocessing_num_workers
    )

    logger.info("Processing evaluation dataset...")
    eval_dataset = raw_datasets["validation"].shuffle().map(
        generate_and_tokenize_prompt,
        remove_columns=list(raw_datasets["validation"].column_names),  # ✅ 移除所有原始列
        desc="Processing eval dataset", 
        num_proc=args.preprocessing_num_workers
    )
    
    # ✅ 验证数据集
    logger.info(f"Train dataset sample: {train_dataset[0]}")
    logger.info(f"Train dataset features: {train_dataset.features}")
    logger.info(f"Eval dataset features: {eval_dataset.features}")
    
    # ✅ DataCollator设置 - 确保正确的padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True  # ✅ 确保这里是True
    )
 
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size
    )
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if hasattr(DistributedType, 'TPU') and accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        
        active_dataloader = train_dataloader
       
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ✅ 修复后的逻辑：只在梯度同步时执行检查点保存和计数
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                # 修复评估频率 - 改为每1000步评估一次
                if completed_steps % 29172 == 0:
                    perplexity, eval_loss = evaluate(args, accelerator, model, eval_dataloader)
                    logger.info(f"steps {completed_steps}: train_loss: {loss:.4f} eval_loss: {eval_loss:.4f} perplexity: {perplexity:.4f}")
                    
                    
                # ✅ 检查点保存逻辑移到 sync_gradients 内部
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        logger.info(f"Saved checkpoint at step {completed_steps}")
                
                if completed_steps >= args.max_train_steps:
                    break
                    
        # epoch结束后的评估
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if epoch < args.num_train_epochs - 1:
            output_dir = args.output_dir+'/step_'+str(epoch)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,safe_serialization=False
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                with open(os.path.join(output_dir, "all_results.json"), "w") as f:
                    json.dump({"perplexity": perplexity}, f)

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,safe_serialization=False
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)
    logger.info(f"End of training")

if __name__ == "__main__":
    main()