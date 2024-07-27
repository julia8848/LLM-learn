from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

model_name_or_path = "facebook/opt-6.7b"

# 使用 GPTQ 算法支持的默认数据集来量化
quantization_config = GPTQConfig(
     bits=4, # 量化精度
     group_size=128,
     dataset="wikitext2",
     desc_act=False,
)

# 逐层量化
quant_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map='auto')

# 保存模型权重
quant_model.save_pretrained("models/opt-6.7b-gptq")