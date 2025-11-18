from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
   
    prompts = prompt_strs
    responses = output_strs

    # tokenize prompts (for computing prompt lengths)
    tokenized_prompts = tokenizer(prompts, padding=True,  return_tensors="pt")
    prompt_lens = tokenized_prompts["attention_mask"].sum(dim=1)

    # tokenize full sequence
    full_texts = [p + r for p, r in zip(prompts, responses)]
    tokenized_fulls = tokenizer(full_texts, padding=True,  return_tensors="pt")

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
    }

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""

    # logits: [B, S, V]
    # 使用 logsumexp 归一化
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # p = softmax(logits) = exp(log_probs)
    probs = torch.exp(log_probs)

    # entropy = -sum(p * log p)
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
   
    # Get logits from model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits # (batch_size, seq_len, vocab_size)

    # Compute log-probabilities for each token in the labels
    log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

    # 从 (batch_size, seq_len, vocab_size) 取出对应 labels 的 log-probs （因为我们计算误差时只针对真实label计算）
    # Result shape: (batch_size, seq_len)
    log_probs_for_labels = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {
        "log_probs": log_probs_for_labels
    }

    if return_token_entropy:
        token_entropy = run_compute_entropy(logits)  # (batch_size, seq_len)
        result["token_entropy"] = token_entropy

    return result

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size, seq_length = policy_log_probs.shape
    # 1. 计算 per-token loss：交叉熵 = -log prob
    per_token_loss = -policy_log_probs    # (batch, seq)

    # 2. 只对 response token 求和
    # 3. 归一化 normalize_constant
    normalized_loss_per_batch = run_masked_normalize(per_token_loss,response_mask,normalize_constant=normalize_constant) / batch_size

    # 4. 适配 gradient accumulation（除以 microbatch 个数）
    micro_loss = normalized_loss_per_batch / gradient_accumulation_steps

    # 5. backward() 但不 step（外部会累积）
    micro_loss.backward()

    # 6. metadata（作业要求一般需要返回这些）
    metadata = {
        "masked_loss": (per_token_loss * response_mask).sum(),
        "normalized_loss": normalized_loss_per_batch.detach(),
        "micro_loss": micro_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }

    return micro_loss.detach(), metadata

    
def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    
    masked_tensor = tensor * mask
    sum_vals = masked_tensor.sum(dim=dim)
    normalized = sum_vals / normalize_constant
    return normalized

