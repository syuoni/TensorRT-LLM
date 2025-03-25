import math
import random
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import List, Optional, Union

import click
import datasets
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorrt_llm.profiler as profiler
from tensorrt_llm._torch.llm import LLM as PyTorchLLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import (LLM, BuildConfig, KvCacheConfig, RequestOutput,
                                 SamplingParams)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.logger import logger, severity_map


class Evaluator(ABC):

    @abstractmethod
    def generate_samples(self):
        raise NotImplementedError()

    @abstractmethod
    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries):
        raise NotImplementedError()

    def evaluate(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None):
        profiler.start("trtllm exec")
        outputs, references, auxiliaries = [], [], []
        for prompt, reference, *aux in tqdm(self.generate_samples(),
                                            desc="Submitting requests"):
            output = llm.generate_async(prompt, sampling_params)
            outputs.append(output)
            references.append(reference)
            auxiliaries.append(aux)
        for output in tqdm(outputs, desc="Fetching responses"):
            output.result()
        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        print(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")

        score = self.compute_score(outputs, references, *zip(*auxiliaries))
        return score

    @abstractstaticmethod
    def command(ctx, *args, **kwargs):
        raise NotImplementedError()


class CnnDailymail(Evaluator):

    def __init__(self,
                 dataset_path: str = "ccdv/cnn_dailymail",
                 num_samples: int = None,
                 random_seed: int = 0,
                 rouge_path: str = "rouge"):
        self.data = datasets.load_dataset(dataset_path, "3.0.0", split="test")
        self.data = self.data.shuffle(random_seed)
        if num_samples is None:
            self.num_samples = self.data.num_rows
        else:
            self.num_samples = min(num_samples, self.data.num_rows)
        self.rouge = evaluate.load(rouge_path)

    def generate_samples(self):
        for i, sample in enumerate(self.data):
            if i >= self.num_samples:
                break
            prompt = sample["article"].strip().replace(" n't",
                                                       "n't") + " TL;DR: "
            reference = sample["highlights"].strip().replace(" n't", "n't")
            yield prompt, reference

    def compute_score(self, outputs: List[RequestOutput],
                      references: List[str]):
        metrics = self.rouge.compute(
            predictions=[output.outputs[0].text for output in outputs],
            references=references)
        for key in metrics.keys():
            logger.info(f"  {key}: {metrics[key]*100:.3f}")
        return metrics["rouge1"]

    @click.command("cnn_dailymail")
    @click.option("--dataset_path", type=str, default="ccdv/cnn_dailymail")
    @click.option("--num_samples", type=int, default=None)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--rouge_path", type=str, default="rouge")
    @click.option("--max_input_length", type=int, default=924)
    @click.option("--max_output_length", type=int, default=100)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=15)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, random_seed: int,
                rouge_path: str, max_input_length: int, max_output_length: int,
                check_accuracy: bool, accuracy_threshold: float):
        llm = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = CnnDailymail(dataset_path,
                                 num_samples=num_samples,
                                 random_seed=random_seed,
                                 rouge_path=rouge_path)
        accuracy = evaluator.evaluate(llm, sampling_params)
        llm.shutdown()

        if check_accuracy:
            assert accuracy > accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"


class Mmlu(Evaluator):
    CHOICES = ["A", "B", "C", "D"]
    SUBJECT_TO_SUBCATEGORIES = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }
    CATEGORY_TO_SUBCATEGORIES = {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }

    def __init__(self,
                 dataset_path: str,
                 num_samples: int = None,
                 num_train: int = 5,
                 random_seed: int = 0):
        self.dataset_path = dataset_path
        if num_samples is None:
            self.num_samples_per_subject = None
        else:
            self.num_samples_per_subject = math.ceil(
                num_samples / len(self.SUBJECT_TO_SUBCATEGORIES))
        self.num_train = num_train
        random.seed(random_seed)
        np.random.seed(random_seed)

    def format_subject(self, subject):
        line = subject.split("_")
        s = ""
        for entry in line:
            s += " " + entry
        return s

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.CHOICES[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def generate_samples(self):
        for subject in self.SUBJECT_TO_SUBCATEGORIES.keys():
            dev_df = pd.read_csv(f"{self.dataset_path}/dev/{subject}_dev.csv",
                                 header=None)
            train_prompt = self.gen_prompt(dev_df, subject, self.num_train)

            test_df = pd.read_csv(
                f"{self.dataset_path}/test/{subject}_test.csv", header=None)
            if self.num_samples_per_subject is not None and self.num_samples_per_subject < test_df.shape[
                    0]:
                test_df = test_df.sample(self.num_samples_per_subject)

            for i in range(test_df.shape[0]):
                prompt_end = self.format_example(test_df,
                                                 i,
                                                 include_answer=False)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]
                yield prompt, label, subject

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      subjects: List[str]):
        subject_corrections = {
            key: []
            for key in self.SUBJECT_TO_SUBCATEGORIES.keys()
        }
        for output, ref, sub in zip(outputs, references, subjects):
            correction = output.outputs[0].text.strip().startswith(ref)
            subject_corrections[sub].append(correction)

        subcategory_corrections = {
            key: []
            for subcats in self.SUBJECT_TO_SUBCATEGORIES.values()
            for key in subcats
        }
        category_corrections = {
            key: []
            for key in self.CATEGORY_TO_SUBCATEGORIES.keys()
        }
        all_corrections = []
        for sub, corrections in subject_corrections.items():
            for subcat in self.SUBJECT_TO_SUBCATEGORIES[sub]:
                subcategory_corrections[subcat].extend(corrections)
                for cat, subcats in self.CATEGORY_TO_SUBCATEGORIES.items():
                    if subcat in subcats:
                        category_corrections[cat].extend(corrections)
            all_corrections.extend(corrections)

        for subject, corrections in subject_corrections.items():
            acc = np.mean(corrections) * 100
            print(
                f"Average accuracy {acc:.2f} ({len(corrections)}) - {subject}")

        for subcat, corrections in subcategory_corrections.items():
            acc = np.mean(corrections) * 100
            print(f"Average accuracy {acc:.2f} ({len(corrections)}) - {subcat}")

        for cat, corrections in category_corrections.items():
            acc = np.mean(corrections) * 100
            print(f"Average accuracy {acc:.2f} ({len(corrections)}) - {cat}")

        weighted_acc = np.mean(all_corrections) * 100
        print(
            f"MMLU weighted average accuracy: {weighted_acc:.2f} ({len(all_corrections)})"
        )
        return weighted_acc

    @click.command("mmlu")
    @click.option("--dataset_path", type=str, required=True)
    @click.option("--num_samples", type=int, default=None)
    @click.option("--num_train", type=int, default=5)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--max_input_length", type=int, default=4094)
    @click.option("--max_output_length", type=int, default=2)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=30)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, num_train: int,
                random_seed: int, max_input_length: int, max_output_length: int,
                check_accuracy: bool, accuracy_threshold: float):
        llm = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = Mmlu(dataset_path,
                         num_samples=num_samples,
                         num_train=num_train,
                         random_seed=random_seed)
        accuracy = evaluator.evaluate(llm, sampling_params)
        llm.shutdown()

        if check_accuracy:
            assert accuracy > accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"


@click.group()
@click.option(
    "--model",
    required=True,
    type=str,
    help="model name | HF checkpoint path | TensorRT engine path",
)
@click.option("--tokenizer",
              type=str,
              default=None,
              help="Path | Name of the tokenizer."
              "Specify this value only if using TensorRT engine as model.")
@click.option("--backend",
              type=click.Choice(["pytorch", "tensorrt"]),
              default="pytorch",
              help="Set to 'pytorch' for pytorch path. Default is cpp path.")
@click.option('--log_level',
              type=click.Choice(severity_map.keys()),
              default='info',
              help="The logging level.")
@click.option("--max_beam_width",
              type=int,
              default=BuildConfig.max_beam_width,
              help="Maximum number of beams for beam search decoding.")
@click.option("--max_batch_size",
              type=int,
              default=BuildConfig.max_batch_size,
              help="Maximum number of requests that the engine can schedule.")
@click.option(
    "--max_num_tokens",
    type=int,
    default=BuildConfig.max_num_tokens,
    help=
    "Maximum number of batched input tokens after padding is removed in each batch."
)
@click.option(
    "--max_seq_len",
    type=int,
    default=BuildConfig.max_seq_len,
    help="Maximum total length of one request, including prompt and outputs. "
    "If unspecified, the value is deduced from the model config.")
@click.option("--tp_size", type=int, default=1, help='Tensor parallelism size.')
@click.option("--pp_size",
              type=int,
              default=1,
              help='Pipeline parallelism size.')
@click.option("--ep_size",
              type=int,
              default=None,
              help="expert parallelism size")
@click.option("--gpus_per_node",
              type=int,
              default=None,
              help="Number of GPUs per node. Default to None, and it will be "
              "detected automatically.")
@click.option("--kv_cache_free_gpu_memory_fraction",
              type=float,
              default=0.9,
              help="Free GPU memory fraction reserved for KV Cache, "
              "after allocating model weights and buffers.")
@click.option("--trust_remote_code",
              is_flag=True,
              default=False,
              help="Flag for HF transformers.")
@click.option("--extra_llm_api_options",
              type=str,
              default=None,
              help="Path to a YAML file that overwrites the parameters")
@click.pass_context
def main(ctx, model: str, tokenizer: Optional[str], log_level: str,
         backend: str, max_beam_width: int, max_batch_size: int,
         max_num_tokens: int, max_seq_len: int, tp_size: int, pp_size: int,
         ep_size: Optional[int], gpus_per_node: Optional[int],
         kv_cache_free_gpu_memory_fraction: float, trust_remote_code: bool,
         extra_llm_api_options: Optional[str]):
    logger.set_level(log_level)
    build_config = BuildConfig(max_batch_size=max_batch_size,
                               max_num_tokens=max_num_tokens,
                               max_beam_width=max_beam_width,
                               max_seq_len=max_seq_len)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=kv_cache_free_gpu_memory_fraction)

    if backend == "tensorrt":
        backend = None
    pytorch_backend_config = None
    if backend == "pytorch":
        pytorch_backend_config = PyTorchConfig(enable_overlap_scheduler=True)

    llm_args = {
        "model": model,
        "tokenizer": tokenizer,
        "tensor_parallel_size": tp_size,
        "pipeline_parallel_size": pp_size,
        "moe_expert_parallel_size": ep_size,
        "gpus_per_node": gpus_per_node,
        "trust_remote_code": trust_remote_code,
        "build_config": build_config,
        "kv_cache_config": kv_cache_config,
        "backend": backend,
        "pytorch_backend_config": pytorch_backend_config,
    }

    if extra_llm_api_options is not None:
        llm_args = update_llm_args_with_extra_options(llm_args,
                                                      extra_llm_api_options)

    profiler.start("trtllm init")
    if backend == 'pytorch':
        llm = PyTorchLLM(**llm_args)
    else:
        llm = LLM(**llm_args)
    profiler.stop("trtllm init")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm init")
    print(f"TRTLLM initialization time: {elapsed_time:.3f} seconds.")

    # Pass llm to subcommands
    ctx.obj = llm


main.add_command(CnnDailymail.command)
main.add_command(Mmlu.command)

if __name__ == "__main__":
    main()
