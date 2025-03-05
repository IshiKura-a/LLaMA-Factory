# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
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
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch.distributed as dist
import torch
import random
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file
from torch.utils.data import DataLoader, DistributedSampler

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import CHOICES, SUBJECTS
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.eval.template import get_eval_template


if TYPE_CHECKING:
    
    from numpy.typing import NDArray


class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        """Initialize model, tokenizer, and distributed settings"""
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.local_rank = int(os.getenv("LOCAL_RANK"))  # Rank of the current process on the node
        self.global_rank = int(os.getenv("NODE_RANK"))  # Global rank across all nodes
        self.world_size = int(os.getenv("WORLD_SIZE"))  # Total number of processes
        if self.local_rank == 0:
            print(f'Local Rank: {self.local_rank}, Global Rank: {self.global_rank}, World Size: {self.world_size}')
        torch.cuda.set_device(self.local_rank)  # Assign each process to a GPU
        dist.init_process_group(backend="nccl", init_method="env://")  # Initialize distributed training

        self.model = load_model(load_tokenizer(self.model_args)["tokenizer"], self.model_args, finetuning_args)
        self.model.to(self.local_rank)
        self.model.eval()

        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        """Perform batch inference on the model"""
        print("%"*20, self.local_rank, set([param.device for param in self.model.parameters()]))
        batch_input = {k: v.to(self.local_rank) for k, v in batch_input.items()}  # Ensure inputs are on correct GPU
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> None:
        """Evaluate the model across multiple GPUs using DataParallel"""
        if self.global_rank == 0:
            print(f"Starting Evaluation | Global Rank: {self.global_rank}, World Size: {self.world_size}")

        eval_task, eval_split = self.eval_args.task.split("_")

        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, encoding="utf-8") as f:
            categories: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categories.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=self.model_args.trust_remote_code,
            )
            pbar.set_postfix_str(categories[subject]["name"])
            # Ensure correct split selection
            train_set, eval_set = dataset["train"], dataset[eval_split]

            # Use DistributedSampler to split dataset across nodes/GPUs
            sampler = DistributedSampler(eval_set, num_replicas=self.world_size, rank=self.global_rank, shuffle=False)
            dataloader = DataLoader(eval_set, batch_size=self.eval_args.batch_size, sampler=sampler, collate_fn=self.collate_fn)

            outputs, labels = [], []

            for batch in tqdm(dataloader, desc=f"Rank {self.global_rank} - Processing Batches", position=1, leave=False):
                batch_inputs, batch_labels = [], []

                for ex in batch:
                    # Select 5-shot examples from the training set
                    train_indices = list(range(len(train_set)))
                    random.shuffle(train_indices)
                    selected_indices = train_indices[:min(self.eval_args.n_shot, len(train_set))]
                    support_set = train_set.select(selected_indices)

                    messages = self.eval_template.format_example(
                        target_data=ex,
                        support_set=support_set,
                        subject_name=categories[subject]["name"],
                    )

                    input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                    batch_inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                    batch_labels.append(messages[-1]["content"])

                batch_input = self.tokenizer.pad(batch_inputs, return_attention_mask=True, return_tensors="pt").to(self.local_rank)
                preds = self.batch_inference(batch_input)

                outputs.extend(preds)
                labels.extend(batch_labels)

                torch.cuda.empty_cache()  # Free unused memory

            # Gather results across all nodes
            gathered_outputs = [None] * self.world_size
            gathered_labels = [None] * self.world_size
            dist.all_gather_object(gathered_outputs, outputs)
            dist.all_gather_object(gathered_labels, labels)

            if self.global_rank == 0:
                # Flatten lists
                gathered_outputs = [item for sublist in gathered_outputs for item in sublist]
                gathered_labels = [item for sublist in gathered_labels for item in sublist]

                corrects = np.array(gathered_outputs) == np.array(gathered_labels)
                category_name = categories[subject]["category"]
                category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
                category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
                results[subject] = {str(i): gathered_outputs[i] for i in range(len(gathered_outputs))}

        dist.barrier()  # Ensure all nodes finish before proceeding
        if self.global_rank == 0:
            self._save_results(category_corrects, results)

    def collate_fn(self, batch):
        """Collate function for DataLoader"""
        return batch  # Return raw batch (each example is processed individually in eval loop)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        """Save evaluation results (only on rank 0)"""
        if self.global_rank == 0:
            score_info = "\n".join(
                [f"{category:>15}: {100 * np.mean(corrects):.2f}" for category, corrects in category_corrects.items() if len(corrects)]
            )
            print(score_info)
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=True)
                with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()

if __name__ == "__main__":
    run_eval()
