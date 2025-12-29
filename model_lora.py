import torch
from torch import nn, optim


# 定义LoRA模块
class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        """
        LoRA模块初始化
        :param in_features: 输入特征维度
        :param out_features: 输出特征维度
        :param rank: LoRA的秩
        """
        super().__init__()
        self.rank = rank  # lora的秩，控制低秩矩阵的大小

        # 初始化LoRA参数
        self.lora_A = nn.Linear(in_features, rank, bias=False)  # 降维
        self.lora_B = nn.Linear(rank, out_features, bias=False)  # 升维

        # 初始化参数高斯分布
        self.lora_A.weight.data.normal_(mean=0.0, std=0.02)
        self.lora_B.weight.data.zero_()

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


def apply_lora_to_model(model, rank=8):
    """
    将LoRA模块应用到模型的指定线性层上

    :param model: 需要应用LoRA的模型
    :param target_modules: 需要替换的线性层名称列表
    :param rank: LoRA的秩
    """
    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        # module.weight.shape[0] == module.weight.shape[1] 确保是方阵
        # 但是Lora来说，其实不一定非得是方阵才能去用
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            lora_module = LoRAModule(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
                rank=rank,
            )
            setattr(model, f"lora", lora_module)

            # 替换原始线性层的前向传播方法
            original_forward = module.forward

            def lora_forward(
                x, original_forward=original_forward, lora_module=lora_module
            ):
                return original_forward(x) + lora_module(x)

            module.forward = lora_forward
    return model


def load_lora_weights(model, path):
    """
    加载LoRA权重到模型中

    :param model: 需要加载LoRA权重的模型
    :param path: LoRA权重文件路径
    """
    lora_state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in lora_state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)
    print(f"LoRA权重已加载：{path}")


def save_lora_weights(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # 这里存和上面取是对应的，也就是说这里对key加上前缀{name}.lora.,取的时候也要对应去掉
            lora_state = {
                f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)
    print(f"LoRA权重已保存：{path}")
