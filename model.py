# 导入一些必要的库
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN


from transformers import PretrainedConfig
from typing import Optional, List, Union
import warnings


class MyModelConfig(PretrainedConfig):
    """
    自定义模型的配置类，支持基础Transformer架构和MOE（混合专家网络）扩展。
    继承自Hugging Face的PretrainedConfig，实现配置的序列化、校验和默认值管理。

    Args:
        dropout (float, optional): 整体Dropout概率，适用于前馈网络、注意力等模块。默认值为0.0。
        bos_token_id (int, optional): 句子开始标记（BOS）的token ID。默认值为1。
        eos_token_id (int, optional): 句子结束标记（EOS）的token ID。默认值为2。
        hidden_act (str, optional): 隐藏层激活函数，支持'silu'/'swish'、'gelu'、'relu'等。默认值为'silu'。
        hidden_size (int, optional): 模型隐藏层维度，需为注意力头数的整数倍。默认值为512。
        intermediate_size (int, optional): 前馈网络中间层维度，若为None则自动按8/3*hidden_size计算并对齐64的倍数。默认值为None。
        max_position_embeddings (int, optional): 模型支持的最大序列长度（位置嵌入上限）。默认值为32768。
        num_attention_heads (int, optional): 注意力机制的总头数（查询头数）。默认值为8。
        num_hidden_layers (int, optional): Transformer隐藏层（编码器层）的数量。默认值为8。
        num_key_value_heads (int, optional): 注意力机制的键值（KV）头数，用于实现Multi-Query Attention (MQA)或Grouped-Query Attention (GQA)。默认值为2。
        vocab_size (int, optional): 词汇表大小（tokenizer的词典规模）。默认值为6400。
        rms_norm_eps (float, optional): RMSNorm归一化的极小值，防止分母为0。默认值为1e-05。
        rope_theta (float, optional): RoPE位置编码的基数（theta），控制位置编码的周期。默认值为1000000.0。
        flash_attn (bool, optional): 是否启用FlashAttention优化，提升注意力计算效率。默认值为True。
        use_moe (bool, optional): 是否启用混合专家网络（MOE）。默认值为False。
        num_experts_per_tok (int, optional): 每个token路由选择的专家数量（top-k），仅当use_moe=True时生效。默认值为2。
        n_routed_experts (int, optional): 路由专家的总数量（独立专家数），仅当use_moe=True时生效。默认值为4。
        n_shared_experts (int, optional): 共享专家的数量（所有token共享的专家），仅当use_moe=True时生效。默认值为1。
        scoring_func (str, optional): MOE路由的评分函数，支持'softmax'/'gumbel_softmax'/'sigmoid'，仅当use_moe=True时生效。默认值为'softmax'。
        aux_loss_alpha (float, optional): MOE负载均衡辅助损失的权重系数，仅当use_moe=True时生效。默认值为0.1。
        seq_aux (bool, optional): 是否在序列级别计算MOE辅助损失，仅当use_moe=True时生效。默认值为True。
        norm_topk_prob (bool, optional): 是否对MOE的top-k专家概率进行归一化，仅当use_moe=True时生效。默认值为True。
        **kwargs: 父类PretrainedConfig的额外参数（如layer_norm_eps、initializer_range等）。
    """

    model_type = "MyModel"

    # 定义配置类的字段类型（用于序列化/反序列化时的类型检查）
    attribute_map = {
        "n_embd": "hidden_size",  # 兼容GPT类模型的命名习惯
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_kv_head": "num_key_value_heads",
    }

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        flash_attn: bool = True,
        # MOE配置
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        # 调用父类构造函数，处理额外参数（如initializer_range、layer_norm_eps等）
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # 基础Transformer配置
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn

        # MOE相关配置
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

        # 配置合法性校验
        self._validate_config()

        # 自动补全intermediate_size的默认值（若未指定）
        self._set_default_intermediate_size()

    def _validate_config(self) -> None:
        """校验配置的合法性，抛出异常或警告以避免无效配置"""
        # 1. 隐藏层维度必须是注意力头数的整数倍
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须是 num_attention_heads ({self.num_attention_heads}) 的整数倍"
            )

        # 2. KV头数必须是查询头数的约数（GQA/MQA要求）
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) 必须是 num_key_value_heads ({self.num_key_value_heads}) 的整数倍"
            )

        # 3. 激活函数合法性检查
        valid_acts = ["silu", "swish", "gelu", "relu", "tanh", "gelu_new", "quick_gelu"]
        if self.hidden_act not in valid_acts:
            warnings.warn(
                f"隐藏层激活函数 {self.hidden_act} 并非推荐类型，支持的类型有: {valid_acts}",
                UserWarning,
            )

        # 4. MOE配置合法性校验（仅当use_moe=True时）
        if self.use_moe:
            # 每个token选择的专家数不能超过总路由专家数
            if self.num_experts_per_tok > self.n_routed_experts:
                raise ValueError(
                    f"num_experts_per_tok ({self.num_experts_per_tok}) 不能超过 n_routed_experts ({self.n_routed_experts})"
                )
            # 评分函数合法性
            valid_scoring_funcs = ["softmax", "gumbel_softmax", "sigmoid"]
            if self.scoring_func not in valid_scoring_funcs:
                raise ValueError(
                    f"MOE评分函数 {self.scoring_func} 不支持，支持的类型有: {valid_scoring_funcs}"
                )
            # 辅助损失权重必须为非负数
            if self.aux_loss_alpha < 0:
                raise ValueError(
                    f"aux_loss_alpha ({self.aux_loss_alpha}) 必须大于等于0"
                )
            # 专家数量必须为正整数
            if self.n_routed_experts <= 0:
                raise ValueError(
                    f"n_routed_experts ({self.n_routed_experts}) 必须大于0"
                )
            if self.n_shared_experts < 0:
                raise ValueError(
                    f"n_shared_experts ({self.n_shared_experts}) 必须大于等于0"
                )

        # 5. 最大序列长度建议为2的幂次（对FlashAttention友好）
        if not (self.max_position_embeddings & (self.max_position_embeddings - 1) == 0):
            warnings.warn(
                f"max_position_embeddings ({self.max_position_embeddings}) 并非2的幂次，可能影响FlashAttention性能",
                UserWarning,
            )

        # 6. Dropout概率范围校验
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout ({self.dropout}) 必须在0到1之间")

    def _set_default_intermediate_size(self) -> None:
        """自动计算intermediate_size的默认值（遵循LLaMA/Qwen的8/3规则，并对齐64的倍数）"""
        if self.intermediate_size is None:
            # 基础计算：8/3 * hidden_size
            intermediate_size = int(self.hidden_size * 8 / 3)
            # 向上取整到64的倍数（GPU Tensor Core优化）
            intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            self.intermediate_size = intermediate_size
            # 打印日志（可选）
            warnings.warn(
                f"intermediate_size未指定，自动计算为 {self.intermediate_size} (8/3 * hidden_size并对齐64的倍数)",
                UserWarning,
            )

    def to_dict(self) -> dict:
        """重写to_dict方法，确保MOE配置也被正确序列化"""
        config_dict = super().to_dict()
        # 过滤掉父类的冗余字段（若有）
        config_dict.pop("transformers_version", None)
        return config_dict


# 定义RMSNorm类，继承torch.nn.Model
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """
        __init__ 的 Docstring

        :param self: 说明
        :param dim: 初始化权重参数
        :type dim: int
        :param eps: 初始化epsilon值，防止公式中的分母为0
        :type eps: float
        """
        self.eps = eps
        self.weight = dim

    def _norm(self, x):
        # 计算RMD归一化 keepdim 保留维度，方便后面和 x 做广播乘法
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def format(self, x):
        # 前向传播，返回归一化之后的结果
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    precompute_freqs_cis 的 预先计算cos和sin的值（RoPE所需）

    :param dim: 每个token的embedding维度（必须是偶数，因为两两分组旋转）
    :type dim: int
    :param end: 预先计算的最大序列长度（即支持的最大token位置数）
    :type end: int
    :param theta: 旋转频率的衰减基数，控制频率随维度的衰减速度
    :type theta: float
    :return: 各位置各维度的cos值和sin值，shape均为[end, dim]
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    # 步骤1：计算每个维度分组的旋转频率（freqs）
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 步骤2：生成位置索引m（从0到end-1）
    m = torch.arange(end, device=freqs.device)
    # 步骤3：计算每个位置m在每个维度分组的旋转角度（m * freqs）
    freqs = torch.outer(m, freqs).float()
    # 步骤4：计算cos值，并扩展到所有维度（两两分组复用）
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    # 步骤5：计算sin值，同样扩展到所有维度
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_post_emb(q, k, freqs_cos, freqs_sin, unsqueeze_dim=1):
    """
    应用RoPE旋转位置编码到Q/K向量
    :param q: Query张量，shape [bsz, seq_len, n_heads, head_dim]
    :param k: Key张量，shape [bsz, seq_len, n_heads, head_dim]
    :param freqs_cos: 预计算的余弦值，shape [seq_len, head_dim]
    :param freqs_sin: 预计算的正弦值，shape [seq_len, head_dim]
    :param unsqueeze_dim: 扩展维度的位置，用于匹配Q/K维度
    :return: 注入位置信息的Q/K张量
    """

    def rotate_half(x):
        # 将张量最后一维拆分为两半，交换并取反后半部分
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 扩展cos/sin维度以匹配Q/K的维度，支持广播运算
    freqs_cos = freqs_cos.unsqueeze(unsqueeze_dim)
    freqs_sin = freqs_sin.unsqueeze(unsqueeze_dim)

    # 应用RoPE旋转公式
    q_embed = q * freqs_cos + rotate_half(q) * freqs_sin
    k_embed = k * freqs_cos + rotate_half(k) * freqs_sin
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA/MQA核心函数：将KV的头数扩展n_rep倍以匹配Query头数
    :param x: 输入KV张量，shape [bsz, seq_len, n_kv_heads, head_dim]
    :param n_rep: 每个KV头需要复制的次数
    :return: 扩展后的KV张量，shape [bsz, seq_len, n_kv_heads*n_rep, head_dim]
    """
    bsz, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # 新增维度用于复制: [bsz, slen, n_kv_heads, 1, head_dim]
        .expand(
            bsz, slen, n_kv_heads, n_rep, head_dim
        )  # 按新维度复制: [bsz, slen, n_kv_heads, n_rep, head_dim]
        .reshape(
            bsz, slen, n_kv_heads * n_rep, head_dim
        )  # 合并头维度: [bsz, slen, n_kv_heads*n_rep, head_dim]
    )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # args.num_attention_heads 指的是多头主力中多个heads
        # num_key_value_heads 指的是像group query这种，那么 num_attention_heads >= num_key_value_heads
        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        # 验证一下参数是否合理 注意力分数计算是 Q 和 K 的转置做点积，要求二者最后一维（head_dim）必须一致，因此hidden_size必须能被num_attention_heads整除。
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # 换个名字
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # group query 中的query会被分成几组
        # n_local_heads >=n_local_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 计算的是单个注意力头的维度（每个头分配到的特征维度）。
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 设置多头注意力的一些W矩阵
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.droupout)
        self.droupout = args.droupout

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings,
        past_key_value,
        use_cache=False,
        attention_mask=None,
    ):
        """
        Transformer自注意力层前向传播（融合RoPE/GQA/KV Cache/掩码机制）
            :param x: 输入特征张量，shape [bsz, seq_len, hidden_dim]
            :param position_embeddings: 预计算的RoPE编码，元组(pre_cos, pre_sin)，shape均为[max_seq_len, head_dim]
            :param past_key_value: 历史KV缓存，元组(past_k, past_v)或None，shape均为[bsz, past_seq_len, n_kv_heads, head_dim]
            :param use_cache: 是否开启KV缓存（自回归生成时设为True）
            :param attention_mask: 注意力掩码，shape [bsz, seq_len]，1表示有效token，0表示padding
            :return: output: 注意力层输出，shape [bsz, seq_len, hidden_dim]
                    past_kv: 新的KV缓存，元组(xk, xv)或None
        """
        # 解析输入维度：bsz=批次大小，seq_len=当前序列长度
        bsz, seq_len, _ = x.shape

        # 1. Q/K/V线性投影：将输入特征映射为Query/Key/Value向量
        # xq/xk/xv shape: [bsz, seq_len, total_dim] (total_dim = n_heads * head_dim)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. 张量形状重塑：将扁平的Q/K/V拆分为四维结构（批次-序列长度-头数-单头维度）
        # 修正原代码：添加赋值操作，否则形状重塑不生效
        xq = xq.view(
            bsz, seq_len, self.n_local_heads, self.head_dim
        )  # [bsz, seq_len, n_local_heads, head_dim]
        xk = xk.view(
            bsz, seq_len, self.n_kv_heads, self.head_dim
        )  # [bsz, seq_len, n_kv_heads, head_dim]
        xv = xv.view(
            bsz, seq_len, self.n_kv_heads, self.head_dim
        )  # [bsz, seq_len, n_kv_heads, head_dim]

        # 3. 应用RoPE旋转位置编码：为Q/K注入位置信息
        pre_cos, pre_sin = position_embeddings
        # 截取与当前序列长度匹配的cos/sin值，避免维度不匹配
        xq, xk = apply_rotary_post_emb(xq, xk, pre_cos[:seq_len], pre_sin[:seq_len])

        # 4. KV Cache缓存拼接：增量推理时复用历史KV，避免重复计算
        if past_key_value is not None:
            # past_key_value[0] = 历史Key，past_key_value[1] = 历史Value
            # 修正原代码：指定dim=1（序列长度维度）拼接，xv拼接时使用自身而非xk
            xk = torch.cat(
                [past_key_value[0], xk], dim=1
            )  # [bsz, past_seq_len+seq_len, n_kv_heads, head_dim]
            xv = torch.cat(
                [past_key_value[1], xv], dim=1
            )  # [bsz, past_seq_len+seq_len, n_kv_heads, head_dim]

        # 5. 更新KV缓存状态：若开启缓存则返回当前KV，否则返回None
        past_kv = (xk, xv) if use_cache else None

        # 6. GQA/MQA关键操作：扩展KV头数以匹配Query头数，并调整维度顺序
        # 转置维度1（序列长度）和维度2（头数），变为注意力计算标准形状：[bsz, n_heads, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),  # Q: [bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(
                1, 2
            ),  # K: 扩展后转置，[bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xv, self.n_rep).transpose(
                1, 2
            ),  # V: 扩展后转置，[bsz, n_local_heads, seq_len, head_dim]
        )

        # 7. 计算缩放注意力分数：Transformer自注意力核心公式
        # xk.transpose(-2, -1)：K最后两维转置，[bsz, n_local_heads, head_dim, seq_len]
        # 矩阵乘法后shape: [bsz, n_local_heads, seq_len, seq_len]，表示每个Q对每个K的注意力分数
        # 除以sqrt(head_dim)：缩放分数，避免softmax梯度消失
        scaled_scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 8. 应用前瞻掩码（Look-Ahead Mask）：防止模型看到未来的token（自回归生成必备）
        # 获取实际的序列长度（拼接历史后）
        actual_seq_len = xk.shape[2]
        # 生成下三角掩码，上三角填充负无穷，diagonal=1表示保留主对角线及以下
        look_ahead_mask = torch.tril(
            torch.full(
                (actual_seq_len, actual_seq_len),
                float("-inf"),
                device=scaled_scores.device,
            ),
            diagonal=1,
        )
        # 融合前瞻掩码与原始分数，未来token的注意力分数变为负无穷
        masked_scores = scaled_scores + look_ahead_mask

        # 9. 应用Padding掩码：忽略无效的padding token
        if attention_mask is not None:
            # 扩展掩码维度以匹配注意力分数的形状：[bsz, 1, 1, actual_seq_len]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将padding位置（0）转为-1e9，有效位置（1）转为0，不影响分数
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            # 融合padding掩码与已有的前瞻掩码
            masked_scores = masked_scores + extended_attention_mask

        # 10. Softmax归一化：将注意力分数转为0-1的权重
        # 转为float计算softmax避免精度损失，再转回原数据类型
        scores = F.softmax(masked_scores.float(), dim=-1).type_as(xq)
        # 注意力权重dropout：防止过拟合
        scores = self.attn_dropout(scores)

        # 11. 注意力加权求和：用注意力权重对Value向量加权
        # output shape: [bsz, n_local_heads, seq_len, head_dim]
        output = scores @ xv

        # 12. 输出维度调整：合并头维度，恢复为扁平特征向量
        # 转置头数和序列长度维度，再重塑为[bsz, seq_len, n_local_heads*head_dim]
        output = output.transpose(1, 2).reshape(bsz, actual_seq_len, -1)
        # 截取当前序列长度的输出（排除历史缓存部分），保证输出维度与输入一致
        output = output[:, -seq_len:, :]

        # 13. 输出投影与dropout：将注意力输出映射回模型隐藏维度
        output = self.o_proj(output)  # [bsz, seq_len, hidden_dim]
        output = self.resid_dropout(output)

        # 返回注意力层输出和新的KV缓存
        return output, past_kv


class FeedForward(nn.Module):
    """
    Transformer 中的前馈网络（FFN）
    这里实现的是：SwiGLU / Gated-MLP 结构
    """

    def __init__(self, config):
        super().__init__()

        # 如果没有指定 intermediate_size，则按经验公式自动计算
        # 8/3 * hidden_size 是 LLaMA / Qwen 常见设置
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)

            # 向上取整到 64 的倍数（对 GPU Tensor Core 友好）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        # 门控投影（Gate Projection）
        # 用于生成 gate 信号，决定哪些信息通过
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

        # 上投影（Up Projection）
        # 将 hidden_size → intermediate_size
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

        # 下投影（Down Projection）
        # 将 intermediate_size → hidden_size
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        # Dropout，用于正则化，防止过拟合
        self.dropout = nn.Dropout()

        # 激活函数（如 silu / gelu）
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前向传播：
        x: [batch, seq_len, hidden_size]
        """

        # Gated MLP / SwiGLU 结构：
        # FFN(x) = Down( Up(x) * Act( Gate(x) ) )
        return self.dropout(
            self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x)))
        )


class MOEGate(nn.Module):
    """
    MoE（Mixture of Experts）门控网络核心类
    功能：接收批次的隐藏状态，为每个token分配Top-K个专家，并计算专家负载均衡辅助损失
    核心逻辑：
    1. 线性变换将token特征映射为专家得分 → 2. Softmax归一化得分 → 3. 选Top-K专家及权重 → 4. 可选权重归一化 → 5. 计算负载均衡辅助损失
    """

    def __init__(self, config):
        """
        初始化MoE门控网络参数
        Args:
            config: 配置类/字典，需包含以下关键参数：
                - num_experts_per_tok: 每个token分配的专家数量（Top-K的K值）
                - n_routed_experts: 可路由的专家总数（门控需要分配的专家池大小）
                - scoring_func: 得分归一化函数（仅支持softmax）
                - aux_loss_alpha: 辅助损失的权重系数（平衡主任务损失和负载均衡损失）
                - seq_aux: 辅助损失计算粒度（True: 序列级 / False: token级）
                - norm_topk_prob: 是否对Top-K专家的权重做归一化（使总和为1）
                - hidden_dim: 输入隐藏状态的维度（门控网络输入特征维度）
        """
        super().__init__()
        # 基础配置参数
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token分配的专家数（Top-K的K）
        self.n_routed_experts = config.n_routed_experts  # 专家池总数
        self.scoring_func = config.scoring_func  # 得分归一化方式（仅支持softmax）
        self.alpha = config.aux_loss_alpha  # 辅助损失权重（控制负载均衡惩罚强度）
        self.seq_aux = config.seq_aux  # 辅助损失计算粒度：序列级/Token级
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化Top-K权重
        self.gating_dim = config.hidden_dim  # 输入token特征维度

        # 门控网络核心可训练参数：[专家数, 特征维度]
        # 作用：将token特征映射为对每个专家的原始得分
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )

        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化门控网络的权重参数
        使用Kaiming均匀初始化（He初始化），适配ReLU类激活函数的梯度传播特性
        a=math.sqrt(5) 是PyTorch Linear层默认值，兼容历史初始化逻辑
        """
        import torch.nn.init as init

        # Kaiming均匀初始化：从均匀分布U[-bound, bound]采样，bound=sqrt(6/((1+a²)*fan_in))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        MoE门控网络前向传播
        Args:
            hidden_states (torch.Tensor): 输入隐藏状态，形状[bsz, seq_len, hidden_dim]
                - bsz: 批次大小
                - seq_len: 序列长度（每个样本的token数）
                - hidden_dim: token特征维度
        Returns:
            topk_ids (torch.Tensor): 每个token分配的Top-K专家索引，形状[bsz*seq_len, top_k]
            topk_weight (torch.Tensor): 每个token分配的Top-K专家权重，形状[bsz*seq_len, top_k]
            aux_loss (torch.Tensor): 专家负载均衡辅助损失，标量（训练模式下计算，eval模式为0）
        """
        # 步骤1：维度解析与张量展平
        # 解析输入张量的维度
        bsz, seq_len, h = hidden_states.shape
        # 展平为二维张量：[bsz*seq_len, hidden_dim]
        # 目的：将每个token视为独立样本，统一计算所有token的专家得分（避免嵌套维度计算）
        hidden_states = hidden_states.view(-1, h)

        # 步骤2：计算token对专家的原始得分
        # 线性变换：将token特征（hidden_dim）映射为专家得分（n_routed_experts）
        # 公式：logits = hidden_states @ weight.T （无偏置）
        # 输出形状：[total_tokens, n_routed_experts]，total_tokens=bsz*seq_len
        logits = F.linear(hidden_states, self.weight, None)

        # 步骤3：得分归一化（Softmax）
        if self.scoring_func == "softmax":
            # Softmax归一化：将无界logits转为0~1的概率分布（每行总和为1）
            # dim=-1：对每个token的所有专家得分做归一化
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"不支持的MoE门控得分函数: {self.scoring_func}，仅支持softmax"
            )

        # 步骤4：选择Top-K专家及权重
        # 对每个token，在专家维度选得分最高的top_k个专家
        # 返回值：
        # - topk_weight: Top-K专家的权重，形状[total_tokens, top_k]
        # - topk_ids: Top-K专家的索引，形状[total_tokens, top_k]
        # sorted=False：不强制按得分排序（提升计算效率，若需排序可设为True）
        topk_weight, topk_ids = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 步骤5：Top-K权重归一化（可选）
        # 仅当分配多个专家（top_k>1）且启用归一化时执行
        # 目的：使每个token的Top-K权重总和为1，统一不同token的专家贡献总量
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算每个token的Top-K权重总和（保留维度，方便广播除法）
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # +1e-20防除零
            # 归一化：权重 / 总和 → 每行总和为1
            topk_weight = topk_weight / denominator

        # 步骤6：计算专家负载均衡辅助损失
        # 仅训练模式且辅助损失权重>0时计算（eval模式下aux_loss为0）
        if self.training and self.alpha > 0:
            scores_for_aux = scores  # 保留所有专家的得分（未取Top-K）
            aux_topk = self.top_k  # 复用Top-K参数
            # 将Top-K专家索引展平为[bsz, seq_len*top_k]，方便按样本/序列统计
            topk_ids_for_aux_loss = topk_ids.view(bsz, -1)

            if self.seq_aux:
                # 序列级辅助损失
                # 逻辑：将单条序列视为整体，惩罚序列内token集中选择少数专家的行为
                # 恢复scores的三维形状：[bsz, seq_len, n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)

                # 初始化专家选中频率矩阵：[bsz, n_routed_experts]，初始全0
                # ce[i][j]表示第i个样本（序列）中，第j个专家被选中的次数
                ce = torch.zeros(
                    (bsz, self.n_routed_experts), device=hidden_states.device
                )

                # 统计每个样本的专家选中次数（scatter_add_是高效的按索引累加操作）
                # dim=1：沿专家维度累加
                # index=topk_ids_for_aux_loss：要累加的专家索引
                # src=全1张量：每个索引位置累加1（统计选中次数）
                ce.scatter_add_(
                    1,
                    topk_ids_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                )
                # 归一化频率：使均匀分布下ce≈1
                # 公式：ce = ce / (seq_len * top_k / n_routed_experts)
                ce.div_(seq_len * aux_topk / self.n_routed_experts)

                # 计算序列级辅助损失：
                # 1. scores_for_seq_aux.mean(dim=1)：每个样本的所有token对专家的平均得分 → [bsz, n_routed_experts]
                # 2. ce * 平均得分：惩罚“高得分+高选中频率”的专家 → [bsz, n_routed_experts]
                # 3. sum(dim=1)：每个样本的损失求和 → [bsz]
                # 4. mean()：所有样本的损失求均值 → 标量
                # 5. *self.alpha：乘以辅助损失权重
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=-1
                ).mean() * self.alpha

            else:
                # Token级辅助损失
                # 逻辑：统计全局所有token的专家选择分布，惩罚全局层面少数专家被过度选中的行为
                # 将专家索引展平为一维：[bsz*seq_len*top_k]
                # 转为one-hot编码：[bsz*seq_len*top_k, n_routed_experts]（选中的专家为1，其余为0）
                make_ce = F.one_hot(
                    topk_ids_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                # 计算全局专家选中频率：沿token维度求均值 → [n_routed_experts]
                # ce[i]表示第i个专家被选中的全局概率（0~1）
                ce = make_ce.float().mean(0)

                # 计算每个专家的全局平均得分：[n_routed_experts]
                Pi = scores_for_aux.mean(0)
                # 归一化选中频率：使均匀分布下fi=1 → fi = ce * 专家总数
                fi = ce * self.n_routed_experts

                # 计算token级辅助损失：
                # 1. Pi * fi：惩罚“高得分+高选中频率”的专家 → [n_routed_experts]
                # 2. sum()：所有专家的损失求和 → 标量
                # 3. *self.alpha：乘以辅助损失权重
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            # 非训练模式/辅助损失权重为0时，辅助损失设为0
            aux_loss = torch.tensor(0.0, device=hidden_states.device)

        # -------------------------- 返回结果 --------------------------
        # topk_ids: 每个token的Top-K专家索引 [total_tokens, top_k]
        # topk_weight: 每个token的Top-K专家权重 [total_tokens, top_k]
        # aux_loss: 专家负载均衡辅助损失（标量）
        return topk_ids, topk_weight, aux_loss


# 定义MOEFeedFprward类
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            (FeedForward(config) for _ in range(config.n.routed_experts))
        )
        self.gate = MOEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                (FeedForward(config) for _ in range(config.n_shared_experts))
            )

    def forward(self, x):
        identity = x  # 做 skip connection
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制专家的选择
        topk_ids, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_ids.view(-1)

        if self.train:
            # 对每个tocken，复制 num_experts_per_tok 多份，
            # 这样做的目的是为了将每个tocken同时传入到topk个被选中的专家里面进行计算
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个与x形状相同但是类型为float16的空张量，用于存储每个tocken经过对应专家处理后的结果
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # flat_topk_idx是一个索引张量，表示每个tocken被分配给那个专家
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 将输出按照tocken和专家维度重新组织
            # 使用topk_weight 权重对每个专家的输出进行加权求和
            y = y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1).sum(dim=1)
            # 把最终的输出恢复为原始形状
            y = y.view(*orig_shape)
        else:
            # 在推理阶段会使用更高效的函数 moe_infer 处理MOE的部分
            # 通常为了减少内存冗余或计算冗余，例如合并多个tocken，一起处理
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )

        # 如果启用了共享专家，它们会作用在所有的tocken上
        if self.config.num_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(x)

        # 通常损失会加到total_loss = task_loss + config.aux.loss * model.aux_loss
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weight):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsom(0)
        # tokens_per_expert = [6,15,20,26] 这四个值分别代表四个专家的处理数量
        tokens_idxs = idxs // self.config.num_experts_per_tok
        # tokens_ids = [3,7,19,21,14,15...]代表着token_idxs[:6]
        # 属于0号专家的；每个token有可能被多个专家处理，取决于 config.num_experts_per_tok

        for i, end_ids in enumerate(tokens_per_expert):
            # 计算当前token的起始索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果没有token被分配给这个专家，就跳过该专家
            if start_idx == end_ids:
                continue
            expert = self.experts[i]
            exp_token_idx = tokens_idxs[start_idx:end_ids]
            # 从原始的x中获取这些token的嵌入
            expert_tokens = x[exp_token_idx]
            # 输入到当前专家网络中进行前向传播
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 对专家当前的输出进行加权
            expert_out.mul_(flat_expert_weight[idxs[start_idx:end_ids]])
            # 使用scatter_add_ 将专家的输出加到最终的输出张量上面去，加权之后的求和
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


# 定义my model block
class MyModelBlock(nn.Module):
    def __init__(self, layer_id: int, config: MyModelConfig):
        super().__init__()
        self.num_attaention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attaention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rws_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rws_norm_eps
        )
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual
        hidden_states += self.mlp(self.post_attention_layernorm, hidden_states)
        return hidden_states, present_key_value


class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # 相当于做好一层层block的stack堆叠
        self.layers = nn.ModuleList(
            [MyModelBlock(i, config) for i in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_cos, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        batch_size, seq_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            # layer 相当于 MyModelBlock 类对应的对象，layer() 相当于调用 MyModelBlock 中的 forward 方法
            hidden_states, present = layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            # 相当于把计算出来的attention 里面的key value 追加到列表中，回头再放到cache里面
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            # layer.mlp相当于把 block 中的 mlp 取出来，取出来就是MOEFeedForward 或者 FeedForward
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MyModelForCausalLLM(PreTrainedModel, GenerationMixin):
    config_class = MyModelConfig

    def __init__(
        self,
        config: MyModelConfig = None,
    ):
        self.config = config or MyModelConfig()
        super().__init__(self.config)

        self.model = MyModel(config=self.config)
        # 输出层
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # 优化，参数共享，减少被训练的的参数量
        # self.model.embed_tokens.weight = self.lm_head.weight
        self.model.embed_tokens.weight = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        # h是堆叠的多个block块，最后一个的输出，作为后面输出层的输入
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # logits_to_keep保存几个时刻，-logits_to_keep保存前几个时刻的logits
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(h[:, slice_indices, :])

        self.OUT.__setitem__("last_hidden_state", h)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", past_kvs)

        return self.OUT
