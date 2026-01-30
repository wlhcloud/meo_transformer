import math
import os

import torch
from torch import nn, optim


# 定义LoRA模块
class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        """
        LoRA模块初始化
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param rank: LoRA的秩
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 初始化LoRA参数
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化参数
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # x: (batch_size, seq_len, in_features) or (batch_size, in_features)
        # 计算LoRA适配：x @ A^T @ B^T
        result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result


def apply_lora_to_model(model, rank=8, target_modules=None, alpha=32):
    """
    将LoRA模块应用到模型的指定线性层上

    :param model: 需要应用LoRA的模型
    :param rank: LoRA的秩
    :param target_modules: 需要替换的线性层名称关键词
    :param alpha: LoRA的缩放参数
    :return: 应用LoRA后的模型
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    # 收集需要处理的线性层
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                linear_layers.append((name, module))

    print(f"找到 {len(linear_layers)} 个目标线性层")

    for i, (name, module) in enumerate(linear_layers):
        # 创建LoRA模块
        lora_module = LoRAModule(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha
        )

        # 将LoRA模块作为子模块添加到原模块中
        module.lora = lora_module

        # 冻结原始权重
        for param in module.parameters():
            param.requires_grad = False

        # 保存原始的前向传播方法
        original_forward = module.forward

        # 创建新的前向传播方法
        def new_forward(self, x, original_forward=original_forward, lora_module=lora_module):
            return original_forward(x) + lora_module(x)

        # 绑定新的前向传播方法
        module.forward = new_forward.__get__(module, type(module))

        print(f"为 {name} 添加LoRA，参数数量: "
              f"A: {module.in_features * rank}, "
              f"B: {module.out_features * rank}, "
              f"总计: {(module.in_features + module.out_features) * rank}")

    return model


def save_lora_weights(model, path):
    """
    保存LoRA权重

    :param model: 模型
    :param path: 保存路径
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if hasattr(module, 'lora') and module.lora is not None:
            # 保存LoRA权重
            lora_state_dict[f"{name}.lora.lora_A"] = module.lora.lora_A.data
            lora_state_dict[f"{name}.lora.lora_B"] = module.lora.lora_B.data

    if lora_state_dict:
        torch.save(lora_state_dict, path)
        print(f"保存了 {len(lora_state_dict)} 个LoRA权重到 {path}")
    else:
        print("警告：没有找到LoRA权重")


def load_lora_weights(model, path):
    """
    加载LoRA权重

    :param model: 模型
    :param path: 权重文件路径
    """
    if not path or not os.path.exists(path):
        print(f"警告：权重文件不存在 {path}")
        return

    lora_state_dict = torch.load(path, map_location='cpu')

    loaded_count = 0
    for name, param in lora_state_dict.items():
        # 解析模块名和参数名
        parts = name.split('.')
        module_name = '.'.join(parts[:-2])  # 去掉最后的 "lora.lora_A/B"
        param_name = parts[-1]  # lora_A 或 lora_B

        # 获取模块
        module = model
        for part in module_name.split('.'):
            module = getattr(module, part)

        if hasattr(module, 'lora') and module.lora is not None:
            if param_name == "lora_A":
                module.lora.lora_A.data = param.to(module.lora.lora_A.device)
                loaded_count += 1
            elif param_name == "lora_B":
                module.lora.lora_B.data = param.to(module.lora.lora_B.device)
                loaded_count += 1

    print(f"加载了 {loaded_count} 个LoRA权重")


# 使用示例
if __name__ == "__main__":
    # 测试LoRA模块
    batch_size = 4
    seq_len = 128
    in_features = 512
    out_features = 1024

    # 创建测试线性层
    linear = nn.Linear(in_features, out_features)

    # 应用LoRA
    apply_lora_to_model(linear, rank=8, target_modules=["weight"])

    # 测试前向传播
    x = torch.randn(batch_size, seq_len, in_features)
    output = linear(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"LoRA参数数量: {sum(p.numel() for n, p in linear.named_parameters() if 'lora' in n)}")