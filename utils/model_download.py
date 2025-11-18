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

# import torch
# torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "models/Qwen2.5-Math-1.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

prompt = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>".format(question="What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number?")
# tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# generate
outputs = model.generate(
    **inputs,
    max_new_tokens=200,  # 40 tokens太短，不够输出<think> + reasoning + <answer>
    do_sample=False      # math任务一般不走随机采样
)

# decode
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

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
