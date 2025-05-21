# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import time
import gc
import random
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm, trange

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
from datasets import load_dataset

from llamafactory.hparams.data_args import DataArguments

from ...extras import logging
from ...data import get_template_and_fix_tokenizer
from ...model import load_tokenizer
from ...extras.constants import CHOICES, SUBJECTS
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ...eval.template import get_eval_template
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

from transformers.utils import cached_file
from transformers.trainer_utils import EvalLoopOutput

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, EvaluationArguments, ModelArguments
    from numpy.typing import NDArray
    
logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


class Seq2SeqTrainerWithBenchmark(CustomSeq2SeqTrainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[Dict[str, Any]] = None,
        eval_args: "EvaluationArguments" = None,
        model_args: "ModelArguments" = None,
        **kwargs
    ):
        super().__init__(finetuning_args, processor, gen_kwargs, **kwargs)
        
        self.eval_args = eval_args
        self.model_args = model_args
        self.data_args = DataArguments(template='fewshot')
        self.eval_tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.eval_tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.eval_tokenizer, self.data_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.eval_tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]
        
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        # loop_results = super().evaluate(
        #     eval_dataset=eval_dataset,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        #     **gen_kwargs
        # )
        
        model = self._wrap_model(self.model, training=False)
        
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)
        model.eval()
        
        eval_task = self.eval_args.task.split("_")[0]
        eval_split = self.eval_args.task.split("_")[1]

        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        track_memory('init')
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            track_memory('load ds')
            pbar.set_postfix_str(categorys[subject]["name"])

            batch_size = self.eval_args.batch_size
            num_samples = len(dataset[eval_split])

            outputs = np.empty(num_samples, dtype=object)
            labels = np.empty(num_samples, dtype=object)

            track_memory('init batch')
            for batch_start in trange(0, num_samples, batch_size, desc="Processing dataset", position=1, leave=False):
                batch_inputs = []
                batch_labels = []

                for i in range(batch_start, min(batch_start + batch_size, num_samples)):
                    track_memory(f'in batch {i}')
                    train_indices = list(range(len(dataset["train"])))
                    random.shuffle(train_indices)  # Shuffles indices, not the dataset
                    selected_indices = train_indices[:min(self.eval_args.n_shot, len(dataset["train"]))]
                    support_set = dataset["train"].select(selected_indices)
                    messages = self.eval_template.format_example(
                        target_data=dataset[eval_split][i],
                        support_set=support_set,
                        subject_name=categorys[subject]["name"],
                    )
                    track_memory(f'in batch {i} create support set')
                    input_ids, _ = self.template.encode_oneturn(tokenizer=self.eval_tokenizer, messages=messages)
                    batch_inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                    batch_labels.append(messages[-1]["content"])
                    track_memory(f'in batch {i} encode and append')

                batch_input = self.eval_tokenizer.pad(
                    batch_inputs, return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                track_memory(f'batch pad')

                labels[batch_start : batch_start + len(batch_labels)] = batch_labels

                with torch.no_grad():
                    track_memory(f'before inf')
                    logits = model(**batch_input).logits
                    torch.cuda.empty_cache()
                    # logits = self.prediction_step(model, batch_input, False, ignore_keys=ignore_keys, **gen_kwargs)[1]
                    track_memory(f'end inf')
                    lengths = torch.sum(batch_input['attention_mask'], dim=-1)
                    probs = logits[torch.arange(len(lengths)), lengths - 1]
                    probs = torch.nn.functional.softmax(probs[:, self.choice_inputs], dim=-1).detach()
                    preds = [chr(ord("A") + offset.item()) for offset in torch.argmax(probs, dim=-1)]
                    track_memory(f'cal prob')

                outputs[batch_start : batch_start + len(preds)] = preds
                track_memory(f'log pred')


                del batch_input, logits, probs, batch_inputs, batch_labels
                torch.cuda.empty_cache()
                gc.collect()

            corrects = np.array(outputs) == labels
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}
            
            del outputs, labels
            gc.collect()
            torch.cuda.empty_cache()

        pbar.close()
        outputs = EvalLoopOutput(predictions=None, label_ids=None, num_samples=None, metrics={
            f'{metric_key_prefix}_{category_name}': 100 * np.mean(category_correct)
                for category_name, category_correct in category_corrects.items() if len(category_correct)
        })
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, outputs.metrics)

        self._memory_tracker.stop_and_update_metrics(outputs.metrics)

    def _save_results(self, category_corrects: Dict[str, "NDArray"], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def track_memory(prefix: str = "Memory Check"):
    return
    if torch.cuda.current_device() == 1:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GiB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GiB
        free = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Get free memory in GiB
        total = torch.cuda.mem_get_info()[1] / (1024 ** 3)  # Get total memory

        if free > 40:
            print(f"[{prefix}] Free: {free:.2f} GiB")
        else:
            print(f"[{prefix}] Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB | Free: {free:.2f} GiB | Total: {total:.2f} GiB")