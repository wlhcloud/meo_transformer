from transformers import PretrainedConfig


# 定义一个名为MyModelConfig的类，继承自PretrainedConfig
class MyModelConfig(PretrainedConfig):
    model_type = "MyModel"

    # 初始化方法，定义了一系列模型参数及其默认值
    def __init__(
            self,
            dropout: float = 0.0,  # Dropout比例，默认为0.0
            bos_token_id: int = 1,  # 开始标记ID，默认为1
            eos_token_id: int = 2,  # 结束标记ID，默认为2
            hidden_act: str = 'silu',  # 隐藏层激活函数，默认为'silu'
            hidden_size: int = 512,  # 隐藏层大小，默认为512
            intermediate_size: int = None,  # 中间层大小，默认为None
            max_position_embeddings: int = 32768,  # 最大位置嵌入数，默认为32768
            num_attention_heads: int = 8,  # 注意力头数，默认为8
            num_hidden_layers: int = 8,  # 隐藏层数，默认为8
            num_key_value_heads: int = 2,  # 键值对头数，默认为2
            vocab_size: int = 6400,  # 词汇表大小，默认为6400
            rms_norm_eps: float = 1e-05,  # RMS归一化的epsilon值，默认为1e-05
            rope_theta: int = 1000000.0,  # RoPE的theta值，默认为1000000.0
            flash_attn: bool = True,  # 是否使用Flash Attention，默认为True
            ####################################################
            # 以下是MOE（混合专家网络）的具体配置
            # 当use_moe为False时，以下配置无效
            ####################################################
            use_moe: bool = False,  # 是否使用MOE，默认为False
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量，默认为2
            n_routed_experts: int = 4,  # 总的专家数量，默认为4
            n_shared_experts: int = 1,  # 共享专家数量，默认为1
            scoring_func: str = 'softmax',  # 评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,  # 辅助损失的alpha参数，默认为0.1
            seq_aux: bool = True,  # 是否在序列级别上计算辅助损失，默认为True
            norm_topk_prob: bool = True,  # 是否标准化top-k概率，默认为True
            **kwargs  # 其他参数
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 将传入的参数赋值给实例变量
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        ####################################################
        # 以下是MOE（混合专家网络）的具体配置
        # 当use_moe为False时，以下配置无效
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 导入一些必要的库和模块
import math
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# 定义RMSNorm类，继承torch.nn.Module
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # 初始化的epsilon值，防止公式中的分母为0
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数 gi

    def _norm(self, x):
        # 计算RMS归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 前向传播，返回归一化之后的结果
        return self.weight * self._norm(x.float()).type_as(x)


# 预先计算cos和sin值
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    # 把一个token的embedding对于的旋转量计算出来，写作 freqs
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 定义m
    m = torch.arange(end, device=freqs.device)
    # 得到的是 m*theta
    freqs = torch.outer(m, freqs).float()
    # cos(m*theta)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    # sin(m*theta)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


# 应用旋转位置编码
def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        # arr[..., 2:4] 相当于 arr[:, :, :, :, 2:4]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * freqs_cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * freqs_sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * freqs_cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * freqs_sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]  # python的基础语法
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        # num_attention_heads 指的是多头注意力中多少个heads
        # num_key_value_heads 指的是像grouped query这种，那么 num_attention_heads >= num_key_value_heads
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # 验证一下参数是否合理
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # 换个名字
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # grouped query 中的 query 会被分成几组
        # n_local_heads >= n_local_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 设置多头注意力的一些W矩阵
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

    def forward(self,
                x: torch.Tensor,
                position_embeddings,  # 接收预先计算的 cos 和 sin
                past_key_value=None,  # 之前时刻的 K 和 V
                use_cache=False,
                attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # reshape
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        pre_cos, pre_sin = position_embeddings
        # 在 Q 和 K 身上应用 ROPE
        xq, xk = apply_rotary_pos_emb(xq, xk, pre_cos[:seq_len], pre_sin[:seq_len])

        # 关于 kv_cache
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        if use_cache:
            past_kv = (xk, xv)
        else:
            past_kv = None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # 使用 self-attention 公式
        scaled_scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores + mask
        look_ahead_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=scaled_scores.device), diagonal=1
        )
        masked_scores = (scaled_scores + look_ahead_mask).unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # attention_mask 中的 0 值会变成非常小的负数 -1e9
            # 将 1 保持为 0 ，这样做在后续的 softmax 操作中，这些非常小的负数值会接近零
            # 从而在 softmax 之后几乎为零，实现忽略这些位置的效果
            extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
            masked_scores = masked_scores + extended_attention_mask

        scores = F.softmax(masked_scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.up_proj(x) * self.act_fn(self.gate_proj(x))))


# 定义MOEGate类
class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts  # 表示总的可选专家数量

        self.scoring_func = config.scoring_func  # 选择使用哪种评分方式（一般就是'softmax'）
        # 为了让MoE表现的更均衡，我们可以设置关于MoE的权重，回头加到total loss身上
        self.alpha = config.aux_loss_alpha  # 控制辅助损失项的权重
        self.seq_aux = config.seq_aux  # 计算关于MOE是否balance的损失时有两种方式（1，token level；2，sequence level）

        self.norm_topk_prob = config.norm_topk_prob  # 是否对 topK 的概率进行归一化
        self.gating_dim = config.hidden_size  # 输入向量的维度
        # 定义一个可学习的门控矩阵，形状为 [n_routed_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # 调用 初始化函数对上面这个 weight 进行初始化
        self.reset_parameters()

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # 这块是核心逻辑，输入是一个batch的隐藏状态，输出是每个token的专家分配结果和辅助损失
        bsz, seq_len, h = hidden_states.shape
        # 把输入展平成二维数组，方便处理每个token; 二维数组对应的形状就是 [bsz*seq_len, hidden_size]
        hidden_states = hidden_states.view(-1, h)
        # 计算每个token对每个专家expert的原始分数logits，形状是 [total_tokens, n_routed_experts]
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'unsupportted scoring fucntion for MOE gating: {self.scoring_func}')

        # 对每个token，在expert维度上选出 topK 个得分最高的专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 是否启用了norm_topk_prob，对topK的权重做归一化，使其总和为1，防止除以零加一个小数值
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 如果处于训练模式并且启用了辅助损失，则开始构建辅助损失项
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 所有expert的得分，也就还没取topK
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)  # 展平之后的topK专家索引

            if self.seq_aux:
                # 按照sequence级别计算辅助损失
                # 每条sequence看作一个整体，如果某条sequence所有token都只用了expert 0，那么则惩罚这条sequence，鼓励其使用其它多个expert
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # 构建一个专家被选择的频率矩阵 ce
                ce = torch.zeros((bsz, self.n_routed_experts), device=hidden_states.device)
                # 使用 scatter_add_ 来统计每个batch每个expert被选中了多少次
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk,
                                                                     device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts
                )
                # 然后做一个平均，并且与平均得分相乘，作为辅助损失
                # 目的是防止某些expert被频繁选中，造成负载不均
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 按照token级别计算辅助损失
                # 分布统计每个token选择了哪个expert，如果大部分token都选择expert 0，则惩罚它，鼓励选择其它expert
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                # 计算每个expert 的平均得分
                Pi = scores_for_aux.mean(0)
                # 计算每个expert被选中的频率
                fi = ce * self.n_routed_experts
                # 辅助损失是两者相乘的结果
                aux_loss = (Pi * fi).sum() * self.alpha

        # topk_idx: 每个token被分配到 topK 个 expert 的索引
        # topk_weight： 每个 expert 对应的权重
        # aux_loss： 辅助损失项，用于平衡专家之间的负载
        return topk_idx, topk_weight, aux_loss


# 定义MOEFeedForward类
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x  # 做 skip connection
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制专家的选择
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 对每个token，复制 num_experts_per_tok 多份，
            # 这样做的目的是为了将每个token同时传入其top-K个被选中的专家里面进行计算
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # 创建一个与x形状相同但是类型为 float16 的空张量，用于存储每个token经过对应专家处理后的结果
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # flat_topk_idx 是一个索引张量，表示每个token被分配给了哪个专家
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 将输出按照token和专家维度重新组织
            # 使用 topk_weight 权重对每个专家的输出进行加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # 把最终输出恢复成原始输入的形状
            y = y.view(*orig_shape)
        else:
            # 在推理阶段使用更高效的函数 moe_infer 处理 MOE 部分
            # 通常是为了减少内存冗余或计算冗余，例如合并多个token，一起处理
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 如果启用了共享专家，它们会作用在所有的token上
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)

        # 通常这个损失会加到 total_loss = task_loss + config.aux_loss_coeff * model.aux_loss
        self.aux_loss = aux_loss

        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # tokens_per_expert = [6, 15, 20, 26] 这四个数值分别代表4个专家处理的token数量
        tokens_idxs = idxs // self.config.num_experts_per_tok
        # token_idxs = [3, 7, 19, 21, 24, 25, 4, 5, 6, 10, 11, 12...] 代表着 token_idxs[:6]
        # 属于0号专家的；每个token有可能被多个专家处理，取决于 config.num_experts_per_tok

        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前专家处理token的起始索引
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # 如果没有token被分配给这个专家，跳过该专家
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = tokens_idxs[start_idx:end_idx]
            # 从原始的输入x中获取这些token的嵌入
            expert_tokens = x[exp_token_idx]
            # 输入到当前专家网络中进行前向传播；
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 对专家输出进行加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 将专家输出加到最终的输出张量上面去，加权之后的求和
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


# 定义my model block
class MyModelBlock(nn.Module):
    def __init__(self, layer_id: int, config: MyModelConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings,
                past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), position_embeddings,
                                                          past_key_value,
                                                          use_cache, attention_mask)
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MyModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 相当于做好一层层的block的stack堆叠
        self.layers = nn.ModuleList([MyModelBlock(i, config) for i in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # layer 相当于 MyModelBlock 类所对应的对象，layer() 相当于调用 MyModelBlock 里面的 forward 方法
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            # 相当于是把计算出来的attention里面的 key_value 追加到列表中，回头再放到cache里面
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            # layer.mlp 相当于是把 block块中的 mlp 取出来，取出来是 MOEFeedForward 或者 FeedForward
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MyModelForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig = None):
        self.config = config or MyModelConfig()
        super().__init__(self.config)

        self.model = MyModel(self.config)
        # 这里是输出层
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 下面这行其实属于优化了，参数的共享，减少了被训练的参数量
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        # h 是堆叠的多个block，最后一个的输出，作为后面输出层的输入
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # logits_to_keep 保存几个时刻的logits，-logits_to_keep 保存前几个时刻的logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])

        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
