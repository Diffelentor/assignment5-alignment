# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2.5-Math-1.5B /data/models/Qwen2.5-Math-1.5B
# # git lfs install
# # git clone https://huggingface.co/Qwen/Qwen2.5-Math-1.5B /data/models/Qwen2.5-Math-1.5B

# from transformers import AutoModelForCausalLM, AutoTokenizer
# # import torch, os

# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# # model = AutoModelForCausalLM.from_pretrained(
# #     "Qwen/Qwen2.5-Math-1.5B",
# #     cache_dir="/root/autodl-tmp/models"  # 指定缓存目录
# # )
# # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", cache_dir="/root/autodl-tmp/models")

# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-Math-1.5B",
#     cache_dir="/root/autodl-tmp/models",
#     trust_remote_code=True
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-Math-1.5B",
#     cache_dir="/root/autodl-tmp/models",
#     trust_remote_code=True
# )

#Load model directly

import torch
# torch.backends.cuda.matmul.allow_tf32 = True



# del model
# del tokenizer
# torch.cuda.empty_cache()

from vllm import LLM, SamplingParams
model_path = "/root/autodl-fs/models/Qwen2.5-Math-1.5B"
prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e.,<think> reasoning process here </think> <answer> answer here </answer>. User: What is $10.0000198\cdot 5.9999985401\cdot 6.9999852$ to the nearest whole number? Assistant: <think>"

# 创建 vLLM LLM 对象
llm = LLM(model=model_path)
# 采样参数：数学任务一般不采样，等价于 greedy
sampling_params = SamplingParams(
    temperature=0.0,
    # top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)

# 生成
outputs = llm.generate([prompt], sampling_params)

# 输出结果
print("以下是vllm的结果:\n", outputs[0].outputs[0].text.strip())

del llm
# del tokenizer
torch.cuda.empty_cache()


# import os
# os.environ["USE_FLASH_ATTN"] = "0"
# import os
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"



# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_path = "/root/autodl-fs/models/Qwen2.5-Math-1.5B"
# # device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # load model
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>".format(question="What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number?")
# # tokenize
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

# # generate
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=1024,  # 40 tokens太短，不够输出<think> + reasoning + <answer>
#     do_sample=False      # math任务一般不走随机采样
# )

# # decode
# print("以下是transformer的结果:\n", tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# del model
# del tokenizer
# torch.cuda.empty_cache()

# # 创建 vLLM LLM 对象
# llm = LLM(model=model_path)
# # 采样参数：数学任务一般不采样，等价于 greedy
# sampling_params = SamplingParams(
#     temperature=0.0,
#     # top_p=1.0,
#     max_tokens=1024,
#     stop=["</answer>"],
#     include_stop_str_in_output=True,
# )

# # 生成
# outputs = llm.generate([prompt], sampling_params)

# # 输出结果
# print("以下是vllm的结果:\n", outputs[0].outputs[0].text.strip())


# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]

# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)


# import torch

# def gpu_test():
#     # 1️⃣ 检查 CUDA
#     if not torch.cuda.is_available():
#         print("CUDA is not available. Please check your GPU environment.")
#         return
#     device = torch.device("cuda")
#     print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"PyTorch version: {torch.__version__}")

#     # 2️⃣ 打印 GPU 总显存和当前显存使用
#     print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
#     print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
#     print(f"Cached memory: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")

#     # 3️⃣ 小型张量运算测试
#     a = torch.randn((1024, 1024), device=device)
#     b = torch.randn((1024, 1024), device=device)
#     c = torch.matmul(a, b)  # 矩阵乘法
#     print("Matrix multiplication successful! Result shape:", c.shape)

#     # 4️⃣ 浮点运算测试
#     d = torch.sum(c ** 2)
#     print("Element-wise operations successful! Sum of squares:", d.item())
#     print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
#     print(f"Cached memory: {torch.cuda.memory_reserved(device) / 1e6:.2f} MB")

# if __name__ == "__main__":
#     gpu_test()

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unittest.mock import patch

model_path = "/root/autodl-fs/models/Qwen2.5-Math-1.5B"
prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>".format(question="What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number?")
device = "cuda" if torch.cuda.is_available() else "cpu"


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
# def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.85):
    # vllm_set_random_seed(seed)
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

# load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# generate
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,  # 40 tokens太短，不够输出<think> + reasoning + <answer>
    do_sample=False      # math任务一般不走随机采样
)

# decode
print("以下是transformer的结果:\n", tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

llm = init_vllm(model_id=model_path, device=device, seed=42, gpu_memory_utilization=0.85)

load_policy_into_vllm_instance(model, llm)
sampling_params = SamplingParams(
    temperature=0.0,
    # top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)

# 生成
outputs = llm.generate([prompt], sampling_params)

# 输出结果
print("以下是vllm的结果:\n", outputs[0].outputs[0].text.strip())



import torch.distributed as dist
if dist.is_initialized():
    dist.destroy_process_group()


