import os
import sys
import argparse
import time
import math
import warnings

import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM
from model import MyModelForCausalLM, MyModelConfig
from dataset import SFTDataset
from model_lora import apply_lora_to_model, save_lora_weights

warnings.filterwarnings("ignore")


def Logger(content):
    # 如果是非分布式进行训练，就在单机上进行打印；如果是分布式进行训练，就在主节点进行打印
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(step, total_steps, base_lr, warmup_ratio=0.03):
    warmup_steps = int(total_steps * warmup_ratio)

    if step < warmup_steps:
        return base_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def init_model(lm_config):
    model_name = "/home/gybwg/ai-project/models/Qwen/Qwen3-Embedding-0___6B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        device_map="auto" if not ddp else {"": args.device}
    )

    Logger(
        f"LLM 可以被训练的参数量是：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    return model, tokenizer


def init_distributed_model():
    if not ddp:
        return

    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_rank = int(os.environ["WORLD_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"

    # 确定当前代码运行在当前节点的那张GPU显卡上
    torch.cuda.set_device(DEVICE)


def train_epoch(epoch):
    # reduction=none 意味着返回每条样本的损失，reduction='sum'，reduction = 'mean'
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # ctx 分两种情况，一种基于cpu,一种基于gpu，主要是为了混合精度训练
        with ctx:
            res = model(X)  # 正向传播得到预测结果
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            if hasattr(res, "aux_loss") and res.aux_loss is not None:
                loss = loss + res.aux_loss
            loss = loss / args.accumulation_steps  # 梯度的累计，一种优化手段
            # 添加 NaN 检查
            if torch.isnan(loss) or torch.isinf(loss):
                Logger(f"Warning: loss is {loss}, skipping step")
                optimizer.zero_grad()
                continue

            # 修改梯度缩放逻辑
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # 回头要使用混合精度训练(FP32,FP16)；容易出现梯度消失
        # scaler.scale(loss).backward()  # 把loss放大

        if (step + 1) % args.accumulation_steps == 0:
            # 梯度的累计意味着连续几次正向传播（loss）,反向传播求gradient，然后才把这几次的梯度拿来更新一次参数
            # 梯度是在optimizer优化器身上，为什么要缩小gradient梯度，是因为前面将 loss 放大了
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            # 做梯度的剪裁
            grad_norm  = torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            # 真正的把梯度应用到参数身上去更新参数

            if scaler.is_enabled():
                # 更新优化器参数
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            global global_step
            global_step += 1

            lr = get_lr(global_step, total_steps, args.learning_rate)
            for param_group in optimizer.param_groups:
                # 相当于是把优化器要去优化的每一层的学习率都设置一下
                param_group["lr"] = lr

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time

            Logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            lora_save_path = (
                f"{args.save_dir}/lora/{args.lora_name}_{lm_config.hidden_size}.pth"
            )
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
            # 只保持lora权重
            save_lora_weights(model, lora_save_path)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyModel SFT With LoRA Training")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument(
        "--epochs", type=int, default=1
    )  # 如果要效果好，可以训练2-6个轮次
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--ddp", action="store_true"
    )  # 如果这个参数出现了，就是True,否则就是False
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=int, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--data_path", default="./data/wenbo3.jsonl", type=str)
    parser.add_argument(
        "--lora_name", type=str, default="lora_sft_model"
    )  # LoRA模型名称

    args = parser.parse_args()

    lm_config = MyModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(
        args.save_dir, exist_ok=True
    )  # exist_ok =True 如果文件夹已经存在也不会报错
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # torch.cuda.amp.autocast()混合精度训练
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    # 如果我们想复现一些结果，可以设置随机种子
    base_seed = 42
    torch.manual_seed(base_seed)  # 如果基于CPU计算，这行起作用
    torch.cuda.manual_seed(base_seed)  # 如果基于GPU计算，这行起作用

    if ddp:
        init_distributed_model()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    # 初始化模型和分词器
    model, tokenizer = init_model(lm_config)
    apply_lora_to_model(model,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        rank=8)

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if "lora" in name)
    if not ddp or dist.get_rank() == 0:
        print(
            f"模型总参数量：{total_params / 1e6:.3f}百万，"
            f"LoRA参数量：{lora_params_count / 1e6:.3f}百万，"
            f"占比：{lora_params_count / total_params * 100:.3f}%"
        )
    # 设置那些参数是需要被优化的
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False  # 冻结不需要调的参数

    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)

    train_ds = SFTDataset(
        args.data_path, tokenizer=tokenizer, max_length=args.max_seq_len
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None

    # 一条条样本读取，一个批次一个批次数据返回
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    use_amp = (args.dtype in ["float16", "bfloat16"])
    scaler_enabled = use_amp and args.dtype != "bfloat16"
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    global_step = 0
    total_steps = args.epochs * iter_per_epoch

    for epoch in range(args.epochs):
        train_epoch(epoch)
