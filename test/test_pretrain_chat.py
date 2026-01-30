import os

import torch
from transformers import AutoTokenizer

from models.model import MyModelForCausalLM, MyModelConfig
from utils.my_llm import project_base_path


def init_model(lm_config):
    # 下面这行的意思是读取一个现成的分词器模型
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_base_path, 'datasets'))
    # 下面这行的意思不是去读取一个现成的大语言模型，而是用自己封装的类初始化一个自己的大语言模型
    model = MyModelForCausalLM(lm_config).to("cuda")
    state_dict = torch.load(os.path.join(project_base_path,'out/pretrain_512.pth'), map_location='cuda')
    model.load_state_dict(state_dict=state_dict, strict=False)
    print(f'LLM 模型加载成功：参数量{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

lm_config = MyModelConfig(hidden_size=512, num_hidden_layers=8,
                          use_moe=False)
init_model(lm_config)