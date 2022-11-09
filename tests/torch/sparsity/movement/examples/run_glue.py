import argparse
import logging
from typing import Optional
from pathlib import Path
from collections import OrderedDict
from itertools import chain
from pprint import pformat

import torch
import torch.cuda
import numpy as np
from nncf import NNCFConfig
from nncf.torch import create_compressed_model
from nncf.torch.utils import is_main_process
from nncf.api.compression import CompressionAlgorithmController
from nncf.common.utils.tensorboard import prepare_for_tensorboard
import jstyleson as json

from datasets import load_dataset
from datasets import DatasetDict
import evaluate
import transformers
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import EvalPrediction
from transformers import HfArgumentParser
from transformers import enable_full_determinism
from transformers.trainer import Trainer
from transformers.trainer import TrainingArguments
from transformers.trainer import TrainerCallback

task_to_sample_keys = {
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
}

dataset_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask', 'position_ids']


def parse_args():
    parser = argparse.ArgumentParser('GLUE')
    parser.add_argument('--task_name', type=str, help=f'Task name for GLUE. Supported tasks: {list(task_to_sample_keys)}.')
    parser.add_argument('--model_name_or_path', type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum length for model input sequences.')
    parser.add_argument('--nncf_config', type=str, default=None, help='Path to NNCF configuration json file.')
    parser.add_argument('--no_cuda', action='store_true', help='Whether to disable cuda devices.')

    args, other_args = parser.parse_known_args()
    training_args, = HfArgumentParser(TrainingArguments).parse_args_into_dataclasses(other_args)

    # post parser checks and overrides
    assert args.task_name in task_to_sample_keys.keys(), f'Task name should be in {list(task_to_sample_keys)}.'
    training_args.no_cuda = args.no_cuda
    training_args.label_names = ["labels"]
    training_args.remove_unused_columns = False
    training_args.overwrite_output_dir = True
    training_args.report_to = []
    training_args.per_device_eval_batch_size = training_args.per_device_train_batch_size
    return args, training_args


class CompressionCallback(TrainerCallback):
    def __init__(self, compression_ctrl: CompressionAlgorithmController):
        self.compression_ctrl = compression_ctrl
        self.compression_stats_list = []
        self._global_step = 0

    def on_epoch_begin(self, *args, **kwargs):
        self.compression_ctrl.scheduler.epoch_step()

    def on_step_begin(self, *args, **kwargs):
        self._global_step += 1
        self.compression_ctrl.scheduler.step()

    def on_step_end(self, *args, **kwargs):
        stats = prepare_for_tensorboard(self.compression_ctrl.statistics())
        stats_dict = OrderedDict(step=self._global_step, **stats)
        self.compression_stats_list.append(stats_dict)
    
    # def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     return super().on_log(args, state, control, **kwargs)


class CompressionTrainer(Trainer):
    def __init__(self, compression_ctrl: Optional[CompressionAlgorithmController], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_ctrl = compression_ctrl

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        # print(inputs.keys())
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.compression_ctrl is not None:
            loss_compress = self.compression_ctrl.loss()
            loss = loss + loss_compress
        return (loss, outputs) if return_outputs else loss


def main():
    args, training_args = parse_args()
    if training_args.seed is not None:
        enable_full_determinism(training_args.seed)

    # datasets
    raw_datasets = load_dataset("glue", args.task_name)
    num_labels = len(raw_datasets["train"].features["label"].names)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(samples):
        sample_keys = task_to_sample_keys[args.task_name]
        max_length = min(args.max_seq_length, tokenizer.model_max_length)
        result = tokenizer(*(samples[key] for key in sample_keys),
                           padding="max_length",
                           max_length=max_length,
                           truncation=True)
        result['position_ids'] = list(range(max_length))
        return result

    with training_args.main_process_first():
        raw_datasets = raw_datasets.map(tokenize_fn)
        raw_datasets = raw_datasets.rename_column('label', 'labels')
        columns_to_remove = set(chain(*raw_datasets.column_names.values())) - set(dataset_columns)
        raw_datasets = raw_datasets.remove_columns(list(columns_to_remove))

    train_dataset = raw_datasets["train"] if training_args.do_train else None
    eval_dataset = raw_datasets["validation"] if training_args.do_eval else None

    # model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    compression_ctrl = None
    if args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(args.nncf_config)
        # nncf_config['compression'] = []
        if nncf_config.get('log_dir', None) is None:
            nncf_config['log_dir'] = training_args.output_dir
        compression_ctrl, model = create_compressed_model(model, nncf_config)

    # trainer
    metric = evaluate.load("glue", args.task_name)

    def compute_metrics(p: EvalPrediction):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    callback = None if compression_ctrl is None else CompressionCallback(compression_ctrl)
    trainer = CompressionTrainer(
        compression_ctrl=compression_ctrl,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=None if callback is None else [callback]
    )

    # do training & evaluation
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # log compression stats
    if callback is not None and is_main_process():
        with open(Path(training_args.output_dir, 'compression_stats.json'), 'w') as f:
            json.dump(callback.compression_stats_list, f, indent=2)


if __name__ == "__main__":
    main()
