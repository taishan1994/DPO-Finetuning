#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
Supervised fine-tuning script for decoder language models.
"""

import logging
import os
import random
import sys
from typing import Optional, List

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from datasets.exceptions import DatasetGenerationError
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


from arguments import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    DPOConfig,
)
from utils import (
    get_checkpoint,
    get_kbit_device_map,  # 默认不用
    get_peft_config,  # 默认不用
    get_quantization_config,  # 默认不用
)

logger = logging.getLogger(__name__)

def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )
    return raw_datasets

def apply_chat_template(example,
                        system="You are a helpful assistant."):
    # print(example)
    messages = example["messages"]
    chosen_message = ""
    rejected_message = ""
    chosen_score = 0.99
    rejected_score = 0.01

    # DPOTrainer里面会加一个Bos
    # prompt = "<|im_start|>system\n{}<|im_end|>\n".format(system)
    prompt = "system\n{}<|im_end|>\n".format(system)
    for i, message in enumerate(messages):
        role = message["role"]
        # print(message)
        if role == "user":
            value = message["value"]
            _input = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(value)
            prompt += _input
        else:
            chosen_value = message["chosen_value"]
            rejected_value = message["rejected_value"]
            if i != len(messages) - 1:
                # 如果是多轮对话， 前面几轮对话的choen和rejected应该是一样的
                prompt += chosen_value  + "<|im_end|>\n"
            else:
                # 最后面不需要再加一个<|im_end|>，DPOTrainer里面会加一个Eos
                chosen_message = chosen_value
                rejected_message += rejected_value
                chosen_score = message["chosen_score"]
                rejected_score = message["rejected_score"]

    example["prompt"] = prompt
    example["chosen"] = chosen_message
    example["rejected"] = rejected_message
    example["reference_chosen_logps"] = chosen_score
    example["reference_rejected_logps"] = rejected_score
    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    dataset_mixer = data_args.dataset_mixer
    raw_datasets = mix_datasets(dataset_mixer, splits=data_args.dataset_splits, shuffle=True)
    # print(raw_datasets)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    # 需要设置bos_token和eos_token
    tokenizer.bos_token_id = tokenizer.get_vocab()["<|im_start|>"]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template,
                                    num_proc=data_args.preprocessing_num_workers,
                                    remove_columns=column_names,
                                    desc="Applying chat template",
                                    load_from_cache_file=False,
                                    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    eval_dataset = eval_dataset.select(range(100))

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['prompt']}")
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['chosen']}")
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['rejected']}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    logger.info("*** Model loaded! ***")

    ########################
    # 如果是基于参数有效微调训练的sft模型，需要先合并
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    ########################
    # Initialize the Trainer
    ########################
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    last_checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
