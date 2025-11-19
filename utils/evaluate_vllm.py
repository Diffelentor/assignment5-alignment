import json
import os
from tqdm import tqdm
from typing import Callable, List

from vllm import LLM, SamplingParams

# -----------------------------
# 1. 加载 reward 函数
# -----------------------------
# 该函数用于解析模型输出并判断是否正确
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
# def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            # device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# -----------------------------
# 2. r1_zero prompt 模板
# -----------------------------
R1_ZERO_PROMPT_TEMPLATE = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e.,<think> reasoning process here </think> <answer> answer here </answer>.\nUser: {}\nAssistant: <think>"

# -----------------------------
# 3. 评估函数定义
# -----------------------------
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams,
    output_file: str = "qwen2.5_math_eval_results.jsonl",
):
    """Evaluate a language model on a list of prompts, compute metrics, and serialize results to disk."""
    print(f"[INFO] Generating {len(prompts)} responses using vLLM...")

    # 模型批量生成
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    correct_format, correct_answer = 0, 0

    for output, gt in zip(outputs, answers):
        generated_text = output.outputs[0].text.strip()
        rewards = reward_fn(generated_text, gt)

        # 统计不同 reward 情况
        fmt_reward = rewards.get("format_reward", 0)
        ans_reward = rewards.get("answer_reward", 0)
        correct_format += fmt_reward
        correct_answer += ans_reward

        results.append({
            "prompt": output.prompt,
            "generated_text": generated_text,
            "gold_answer": gt,
            "rewards": rewards
        })

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved evaluation results to {output_file}")
    print(f"[METRICS] Format Reward = {correct_format / len(prompts):.3f}")
    print(f"[METRICS] Answer Reward = {correct_answer / len(prompts):.3f}")
    return results


# -----------------------------
# 4. 主脚本逻辑
# -----------------------------
if __name__ == "__main__":
    # 模型路径
    # MODEL_PATH = "/models/Qwen2.5-Math-1.5B"
    MODEL_PATH = "/root/autodl-fs/models/Qwen2.5-Math-1.5B"
    DATA_PATH = "/root/assignment5-alignment/data/math/test.jsonl"
    OUTPUT_PATH = "qwen2.5_math_eval_results.jsonl"

    # 载入数据
    print("[INFO] Loading validation data...")
    questions, answers = [], []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["problem"]
            a = item["answer"]
            questions.append(q)
            answers.append(a)

    # 构建 prompts
    prompts = [R1_ZERO_PROMPT_TEMPLATE.format(q) for q in questions]

    # vLLM 推理配置
    sampling_params = SamplingParams(
        temperature=0.0,
        # top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print("[INFO] Initializing vLLM model...")
    # llm = LLM(model=MODEL_PATH)
    llm = init_vllm(model_id=MODEL_PATH, device="cuda", seed=42)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    load_policy_into_vllm_instance(policy=None, llm=llm)
    

    # 运行评估
    results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        answers=answers,
        eval_sampling_params=sampling_params,
        output_file=OUTPUT_PATH,
    )
