import os
import sys

import torch
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
)
from model import MyModelConfig, MyModelForCausalLLM

warnings.filterwarnings("ignore", category=UserWarning)


def convert_torch2transformers_mymodel(
    torch_path, transformers_path, dtype=torch.bfloat16
):
    """
    pt模型转换为transformers模型

    :param torch_path: pt模型路径
    :param transformers_path: 转换后存放路径
    :param dtype: 模型精度
    """
    MyModelConfig.register_for_auto_class()
    MyModelForCausalLLM.register_for_auto_class("AutoModelForCausalLM")

    lm_model = MyModelForCausalLLM(lm_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict=state_dict, strict=False)  # 把参数放到神经网络
    lm_model = lm_model.to(dtype)  # 转换模型参数精度
    # 统计可训练参数有多少个
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f"模型参数有：{model_params/1e6}百万={model_params/1e9}B（Billion）")

    # safe_serialization False=.b True=safetensors
    lm_model.save_pretrained(transformers_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-MyModel 格式：{transformers_path}")


def convert_torch2transformers_llama(
    torch_path, transformers_path, dtype=torch.bfloat16
):
    # LlamaForCausalLM 兼容三方结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(torch_path, map_location=device)

    llama_config = LlamaConfig(
        vocab_size=lm_config.vocab_size,
        hidden_size=lm_config.hidden_size,
        intermediate_size=64 * ((int(lm_config.hidden_size))),
        num_hidden_layers=lm_config.num_hidden_layers,
        num_attention_heads=lm_config.num_attention_heads,
        num_key_value_heads=lm_config.num_key_value_heads,
        max_position_embeddings=lm_config.max_seq_len,
        rms_norm_eps=lm_config.rms_norm_eps,
        rope_theta=lm_config.rope_theta,
    )
    llama_model = LlamaForCausalLM(llama_config)
    llama_model.load_state_dict(state_dict=state_dict, strict=False)
    llama_model = llama_model.to(dtype)
    llama_model.save_pretrained(transformers_path, safe_serialization=True)

    llama_model.save_pretrained(transformers_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    tokenizer.save_pretrained(transformers_path)
    print(f"模型已保存为 Transformers-MyModel 格式：{transformers_path}")


if __name__ == "__main__":
    lm_config = MyModelConfig(
        hidden_size=512, num_hidden_layers=8, max_seq_len=512, use_moe=False
    )
    torch_path = "./out_back/pretrain_512_moe.pth"
    transformers_path = "./MyModel"

    # convert_torch2transformers_mymodel(torch_path, transformers_path)
    convert_torch2transformers_llama(torch_path, transformers_path)
