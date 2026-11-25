# Смоук-тест: Qwen3-0.6B на transformers — проверяем VM, GPU и загрузку с Hugging Face.
# Токен заранее заливается в /content/.hf_token (см. README).
import os
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
os.environ["HF_TOKEN"] = token = open("/content/.hf_token").read().strip()

gpu = torch.cuda.is_available()
print("устройство:", torch.cuda.get_device_name(0) if gpu else "CPU", "| transformers", transformers.__version__)

t = time.time()
tok = AutoTokenizer.from_pretrained(MODEL, token=token)
model = AutoModelForCausalLM.from_pretrained(MODEL, token=token, dtype=torch.float16 if gpu else torch.float32)
model.to("cuda" if gpu else "cpu")
print(f"загрузка: {time.time() - t:.1f} с")

msgs = [{"role": "user", "content": "Объясни в двух предложениях, что такое Google Colab."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt").to(model.device)
t = time.time()
out = model.generate(**inputs, max_new_tokens=128)[0][inputs.input_ids.shape[1]:]
dt = time.time() - t
print("ответ:", tok.decode(out, skip_special_tokens=True))
print(f"скорость: {len(out) / dt:.1f} ток/с")
