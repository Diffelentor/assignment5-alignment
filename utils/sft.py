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

from accelerate import Accelerator

# 可以选择 mixed_precision: "no", "fp16", "bf16"
accelerator = Accelerator(mixed_precision="bf16")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from utils.sft_utils import PromptResponseDataset
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
from utils.sft_utils import run_tokenize_prompt_and_output,run_sft_microbatch_train_step,run_get_response_log_probs,apply_r1_zero_prompt_template,evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from utils.evaluate_vllm import init_vllm,load_policy_into_vllm_instance

# -----------------------
# Training loop
# -----------------------
def train_sft(
    model_path: str,
    sft_txts: List[Dict[str, str]],
    val_jsonl: str,
    out_dir: str,
    devices: str = "cuda:0",
    dataset_sizes: List[int] = [128, 256, 512, 1024, -1],
    epochs: int = 3,
    micro_batch_size: int = 8,
    lr: float = 1e-5,
    gradient_accumulation_steps: int = 1,
    eval_every_steps: int = 200,
):
    device = devices
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    policy = AutoModelForCausalLM.from_pretrained(model_path)#, torch_dtype=torch.bfloat16)

    val_txts = {"prompts":[],"answers":[]}
    with open(val_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            # val_txts.append({"prompt": j.get("prompt", j.get("question", "")), "answer": str(j.get("answer", ""))})
            # val_txts.append({"prompt": apply_r1_zero_prompt_template(j["problem"]), "answer": str(j["answer"])})
            val_txts["prompts"].append(apply_r1_zero_prompt_template(j["problem"]))
            val_txts["answers"].append(str(j["answer"]))
            

    if _HAS_WANDB:
        wandb.init(project="qwen-2.5-math-1.5B-sft-math", reinit=True)
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # llm = init_vllm(model_id=model_path, device=device, seed=42, gpu_memory_utilization=0.85)
    # llm = init_vllm(model_id=model_path, seed=42, gpu_memory_utilization=0.85)

    for size in dataset_sizes:
        if size == -1:
            cur_txts = sft_txts
            tag = "full"
        else:
            cur_txts = sft_txts[:size]
            tag = str(size)
        print(f"Starting SFT with dataset size={tag}, n_txts={len(cur_txts)}")

        ds = PromptResponseDataset(cur_txts, tokenizer, max_length=1024)
        dataloader = DataLoader(ds, batch_size=micro_batch_size, shuffle=True, collate_fn=ds.collate_fn)

        optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

        policy, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)

        global_step = 0
        eval_step = 0

        for epoch in range(epochs):
            policy.train()
            for batch in dataloader:
                global_step += 1
                print(f"开始第{global_step}步训练")
                # input_ids = batch["input_ids"].to(device_policy)
                # attention_mask = batch["attention_mask"].to(device_policy)
                # response_mask = batch["response_mask"].to(device_policy)
                # labels = batch["labels"].to(device_policy)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                response_mask = batch["response_mask"]
                labels = batch["labels"]

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

                total_response_tokens = response_mask.sum().clamp(min=1.0) / micro_batch_size

                micro_loss, metadata = run_sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    # normalize_constant=total_response_tokens.item(),
                )

                clip_grad_norm_(policy.parameters(), max_norm=1.0)

                if (global_step % gradient_accumulation_steps) == 0:
                    accelerator.backward(micro_loss)
                    optimizer.step()
                    optimizer.zero_grad()

                if _HAS_WANDB:
                    wandb.log({f"train/{tag}/masked_loss": metadata["masked_loss"].item(),
                               f"train/{tag}/normalized_loss": metadata["normalized_loss"].item(),
                               f"train/{tag}/micro_loss": metadata["micro_loss"].item(),
                               f"train/{tag}/num_response_tokens": metadata["num_response_tokens"].item(),
                               "train_step": global_step})

                if global_step % eval_every_steps == 0:
                    eval_step += 1
                    print(f"开始第{eval_step}步测试")
                    policy.eval()
    
                    # load_policy_into_vllm_instance(policy, llm)
                    tmp_path = "/root/autodl-tmp/tmp"
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(policy)
                    unwrapped_model.save_pretrained(tmp_path, save_function=accelerator.save)
                    tokenizer.save_pretrained(tmp_path)

                    del policy
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    llm = LLM(model=tmp_path)
                    
                    # eval_val_subset = val_txts[:512]
                    acc, _ = evaluate_with_vllm(llm, eval_val_subset)
                    result, acc_format, acc_answer = evaluate_vllm(
                        vllm_model=llm,
                        reward_fn=r1_zero_reward_fn,
                        prompts=val_txts["prompts"],
                        answers=val_txts["answers"],
                        eval_sampling_params=SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True),
                        output_file=out_dir + f"/eval_results_{tag}_step{global_step}.jsonl",
                    )
                    print(f"[dataset={tag}] eval_step={eval_step} global_step={global_step} acc_format={acc_format:.4f} acc_answer={acc_answer:.4f}")
                    if _HAS_WANDB:
                        wandb.log({f"eval/{tag}/acc_answer": acc_answer, "acc_format": acc_format,  "eval_step": eval_step})
                    policy.train()
                    del llm
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    # tokenizer = AutoTokenizer.from_pretrained(tmp_path)
                    policy = AutoModelForCausalLM.from_pretrained(tmp_path)

                    

        # ckpt_path = Path(out_dir) / f"policy_final_{tag}.pt"
        # torch.save(policy.state_dict(), ckpt_path)
        # print(f"Saved checkpoint to {ckpt_path}")
        out_dir = out_dir + f"/policy_final_{tag}"
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(policy)
        unwrapped_model.save_pretrained(out_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(out_dir)

    if _HAS_WANDB:
        wandb.finish()
        
        
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
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--dataset_sizes", type=str, default="128,256,512,1024,-1")
    args = parser.parse_args()
    print(args)

    all_txts = []
    with open(args.sft_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            j = json.loads(line)
            all_txts.append({"problem": j["problem"], "solution": j["solution"], "answer": str(j.get("answer", ""))})
            
    dsizes = [int(x) for x in args.dataset_sizes.split(",")]

    train_sft(
        model_path=args.model_path,
        sft_txts=all_txts,
        val_jsonl=args.val_jsonl,
        out_dir=args.out_dir,
        devices=args.devices,
        dataset_sizes=dsizes,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_steps=args.eval_every_steps,
    )


if __name__ == "__main__":
    main()
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()