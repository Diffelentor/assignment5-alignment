#!/usr/bin/env python3
"""
sft_experiment.py

Usage example:
CUDA_VISIBLE_DEVICES=0,1 python utils/sft.py \
  --model_path models/Qwen2.5-Math-1.5B \
  --sft_jsonl data/math/train.jsonl \
  --val_jsonl data/math/test.jsonl \
  --out_dir ./out \
  --devices "cuda:0,cuda:1"
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

# Optional: wandb
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

hf_logging.set_verbosity_error()


# -----------------------
# Utilities / Interfaces
# -----------------------
from utils.sft_utils import run_tokenize_prompt_and_output,run_sft_microbatch_train_step,run_get_response_log_probs


R1_ZERO_PROMPT_TEMPLATE = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e.,<think> reasoning process here </think> <answer> answer here </answer>.\nUser: {problem}\nAssistant: <think>"

R1_ZERO_RESPONSE_TEMPLATE = "{think} </think> <answer> {answer} </answer>"

# -----------------------
# Dataset helpers
# -----------------------
class PromptResponseDataset(Dataset):
    def __init__(self, txts: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 1024):
        self.txts = txts
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.txts)

    def __getitem__(self, idx):
        txt = self.txts[idx]
        prompt = R1_ZERO_PROMPT_TEMPLATE.format(problem=txt["problem"])
        response = R1_ZERO_RESPONSE_TEMPLATE.format(think=txt["solution"], answer=txt["answer"])
        full = prompt + response
        return {"prompt": prompt, "response": response, "full": full}

    def collate_fn(self, batch):
        prompts = [b["prompt"] for b in batch]
        responses = [b["response"] for b in batch]
        full_texts = [b["full"] for b in batch]

        # tokenize prompts (for computing prompt lengths)
        tokenized_prompts = self.tokenizer(prompts, padding=True,  return_tensors="pt")
        prompt_lens = tokenized_prompts["attention_mask"].sum(dim=1)

        # tokenize full sequence
        # full_texts = [p + r for p, r in zip(prompts, responses)]
        tokenized_fulls = self.tokenizer(full_texts, padding=True,  return_tensors="pt")

        input_ids = tokenized_fulls["input_ids"]
        attention_mask = tokenized_fulls["attention_mask"]

        # shift input_ids / labels for causal LM
        labels = input_ids.clone()
        for i, p_len in enumerate(prompt_lens):
            labels[i, :p_len] = -100  # mask prompt
        response_mask = (labels != -100).long()

        # shift for self-regressive training
        input_ids = input_ids[:, :-1]
        labels    = labels[:, 1:]
        attention_mask = attention_mask[:, 1:]
        response_mask = response_mask[:, 1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "prompts": prompts,
            "responses": responses,
        }


# -----------------------
# vLLM helpers
# -----------------------
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# -----------------------
# Evaluation
# -----------------------
def evaluate_with_vllm(llm: LLM, val_txts: List[Dict[str, str]], max_new_tokens=128):
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
    correct = 0
    total = 0
    results = []
    
    prompts = [txt["prompt"] for txt in val_txts]
    outputs = llm.generate(prompts, sampling_params)
    
    for i, output in enumerate(outputs):
        txt = val_txts[i]
        gold = txt.get("answer", "").strip()
        out = output.outputs[0].text.strip()
        
        try:
            pred = out.strip().splitlines()[-1].strip().replace(",", "")
            if "." in pred:
                pred = str(round(float(pred)))
            else:
                pred = str(int(pred))
        except (ValueError, IndexError):
            pred = ""

        total += 1
        if pred == gold:
            correct += 1
        results.append({"prompt": txt["prompt"], "gold": gold, "pred": pred, "raw": out})
            
    acc = correct / max(1, total)
    return acc, results


# -----------------------
# Training loop
# -----------------------
def train_sft(
    model_path: str,
    sft_txts: List[Dict[str, str]],
    val_jsonl: str,
    out_dir: str,
    devices: str = "cuda:0,cuda:1",
    dataset_sizes: List[int] = [128, 256, 512, 1024, -1],
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    eval_every_steps: int = 200,
):
    os.makedirs(out_dir, exist_ok=True)
    device_policy, device_eval = devices.split(",")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    policy.to(device_policy)

    val_txts = []
    with open(val_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            # val_txts.append({"prompt": j.get("prompt", j.get("question", "")), "answer": str(j.get("answer", ""))})
            val_txts.append({"problem": j["problem"], "solution": j["solution"], "answer": str(j["answer"])})

    if _HAS_WANDB:
        wandb.init(project="sft-math", reinit=True)
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    llm = init_vllm(model_id=model_path, device=device_eval, seed=42, gpu_memory_utilization=0.85)

    for size in dataset_sizes:
        if size == -1:
            cur_txts = sft_txts
            tag = "full"
        else:
            cur_txts = sft_txts[:size]
            tag = str(size)
        print(f"Starting SFT with dataset size={tag}, n_txts={len(cur_txts)}")

        ds = PromptResponseDataset(cur_txts, tokenizer, max_length=1024)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        global_step = 0
        eval_step = 0

        for epoch in range(epochs):
            policy.train()
            for batch in dataloader:
                global_step += 1
                input_ids = batch["input_ids"].to(device_policy)
                attention_mask = batch["attention_mask"].to(device_policy)
                response_mask = batch["response_mask"].to(device_policy)
                labels = batch["labels"].to(device_policy)

                # outputs = policy(input_ids=input_ids, attention_mask=attention_mask)
                # logits = outputs.logits
                
                result = run_get_response_log_probs(policy,input_ids,labels,return_token_entropy=True,attention_mask=attention_mask)
                log_probs = result["log_probs"]
                token_entropy = result["token_entropy"]

                # shifted_logits = logits[:, :-1, :]
                # shifted_labels = input_ids[:, 1:]
                # shifted_response_mask = response_mask[:, 1:].to(dtype=torch.float32)

                # log_probs = torch.log_softmax(shifted_logits, dim=-1)
                # policy_log_probs = torch.gather(log_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)

                total_response_tokens = response_mask.sum().clamp(min=1.0) / batch_size

                micro_loss, metadata = run_sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    # normalize_constant=total_response_tokens.item(),
                )

                clip_grad_norm_(policy.parameters(), max_norm=1.0)

                if (global_step + 1 % gradient_accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                if _HAS_WANDB:
                    wandb.log({f"train/{tag}/masked_loss": metadata["masked_loss"].item(),
                               f"train/{tag}/normalized_loss": metadata["normalized_loss"].item(),
                               f"train/{tag}/micro_loss": metadata["micro_loss"].item(),
                               f"train/{tag}/num_response_tokens": metadata["num_response_tokens"].item(),
                               "train_step": global_step})

                if global_step + 1 % eval_every_steps == 0:
                    eval_step += 1
                    policy.eval()
                    load_policy_into_vllm_instance(policy, llm)
                    
                    eval_val_subset = val_txts[:512]
                    acc, _ = evaluate_with_vllm(llm, eval_val_subset)
                    print(f"[dataset={tag}] eval_step={eval_step} global_step={global_step} acc={acc:.4f}")
                    if _HAS_WANDB:
                        wandb.log({f"eval/{tag}/accuracy": acc, "eval_step": eval_step})
                    policy.train()

        ckpt_path = Path(out_dir) / f"policy_final_{tag}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if _HAS_WANDB:
        wandb.finish()


# -----------------------
# Dataset Filtering
# -----------------------
def filter_sft_txts_by_answer(txts: List[Dict[str, str]], llm_for_eval: LLM):
    filtered = []
    prompts = [txt["prompt"] for txt in txts]
    outputs = llm_for_eval.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024))
    
    for i, output in enumerate(outputs):
        txt = txts[i]
        gold = txt.get("answer", "").strip()
        out_text = output.outputs[0].text.strip()
        try:
            pred = out_text.strip().splitlines()[-1].strip().replace(",", "")
            if "." in pred:
                pred = str(round(float(pred)))
            else:
                pred = str(int(pred))
        except (ValueError, IndexError):
            pred = ""

        if pred == gold:
            filtered.append(txt)
            
    return filtered


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--sft_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--devices", type=str, default="cuda:0,cuda:1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--dataset_sizes", type=str, default="128,256,512,1024,-1")
    parser.add_argument("--filter_correct", action="store_true", help="Filter SFT data for correct answers.")
    args = parser.parse_args()

    all_txts = []
    with open(args.sft_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            all_txts.append({"prompt": j["prompt"], "response": j["response"], "answer": str(j.get("answer", ""))})

    if args.filter_correct:
        print("Filtering SFT dataset for correct answers...")
        _, device_eval = args.devices.split(",")
        llm = init_vllm(model_id=args.model_path, device=device_eval, seed=42)
        
        sft_txts_to_use = filter_sft_txts_by_answer(all_txts, llm)
        print(f"Filtered dataset size: {len(sft_txts_to_use)}")
        dsizes = [-1]
    else:
        sft_txts_to_use = all_txts
        dsizes = [int(x) for x in args.dataset_sizes.split(",")]

    train_sft(
        model_path=args.model_path,
        sft_txts=sft_txts_to_use,
        val_jsonl=args.val_jsonl,
        out_dir=args.out_dir,
        devices=args.devices,
        dataset_sizes=dsizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_steps=args.eval_every_steps,
    )


if __name__ == "__main__":
    main()