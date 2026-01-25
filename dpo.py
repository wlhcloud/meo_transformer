import os
import sys
import argparse
import time
import math
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model import MyModelForCausalLLM, MyModelConfig
from dataset import PretrainDataset, SFTDataset

warnings.filterwarnings("ignore")


def Logger(content):
    # 如果是非分布式进行训练，就在单机上进行打印；如果是分布式进行训练，就在主节点进行打印
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_step, lr):
    """
    在训练过程中对学效率进行调整函数，这里选择一个比较流行的cos scheduler
    Args:
        - current_step: 当前迭代到第几次了
        - total_step: 总共需要迭代多少次
        - lr: 学习率初始值
    """
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_step))


def logits_to_probs(logits, labels):
    # 从大模型给出的对数几率，变成概率分布
    # logits shape: (batch_size,seq_len,vocab_size)
    # labels shape:(batch_size,seq_len)
    # probs shape: (batch_size,seq_len)
    log_probs = F.log_softmax()  # 先进行softmax,然后再取对数log
    # gather取值
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    # ref_probs: refrenshce model 进行推理预测之后的结果
    # probs: 要通过dpo去进行训练的模型推理预测之后的结果
    # mask: 指的是 Question+Answer 需要计算loss被考虑的时刻，说白了就只和Anwser有关
    # beta: 就是dpo 损失函数公式中的超参数
    # ref_probs 和 probs 都是同样的形状
    seq_lengths = mask.sum(dim=1, keepdim=True)  # (batch_size,1)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将chosen数据和rejected数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[: batch_size // 2]
    rejected_ref_probs = ref_probs[batch_size // 2 :]
    chosen_probs = probs[: batch_size // 2]
    rejected_probs = probs[batch_size // 2 :]

    # DPO论文中的损失函数公式计算
    pi_logratios = chosen_probs - rejected_probs
    ref_logratios = chosen_ref_probs - rejected_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def init_model(lm_config):
    # 读取现成的分词器模型
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    # 使用自己封装的类初始化一个自己的大语言模型
    model = MyModelForCausalLLM(config=lm_config)  # 此时是随机初始化的参数
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict=state_dict, strict=False)
    model = model.to(args.device)
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

    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch["x_chosen"].to(args.device)
        y_chosen = batch["y_chosen"].to(args.device)
        mask_chosen = batch["mask_chosen"].to(args.device)
        x_rejected = batch["x_rejected"].to(args.device)
        y_rejected = batch["y_rejected"].to(args.device)
        mask_rejected = batch["mask_rejected"].to(args.device)

        x = torch.cat(
            [x_chosen, x_rejected], dim=0
        )  # 相当于样本的堆叠，只想要一个x数据集
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        loss_mask = loss_mask.to(args.device)

        lr = get_lr(
            current_step=epoch * iter_per_epoch + step,
            total_step=args.epochs * iter_per_epoch,
            lr=args.learning_rate,
        )

        for param_group in optimizer.param_groups:
            # 相当于是把优化器要去优化的每一层的学习率都设置一下
            param_group["lr"] = lr

        # ctx 分两种情况，一种基于cpu,一种基于gpu，主要是为了混合精度训练
        with ctx:
            with torch.no_grad():  # 这个with中的代码涉及到model不会被训练
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask

            outputs = model(x)  # 正向传播得到预测结果
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = ref_probs * mask

            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss = loss / args.accumulation_steps  # 梯度的累计，一种优化手段

        # 回头要使用混合精度训练(FP32,FP16)；容易出现梯度消失
        scaler.scale(loss).backward()  # 把loss放大

        if (step + 1) % args.accumulation_steps == 0:
            # 梯度的累计意味着连续几次正向传播（loss）,反向传播求gradient，然后才把这几次的梯度拿来更新一次参数
            # 梯度是在optimizer优化器身上，为什么要缩小gradient梯度，是因为前面将 loss 放大了
            scaler.unscale_(optimizer)
            # 做梯度的剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 真正的把梯度应用到参数身上去更新参数
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

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

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.gat_rank() == 0):
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            # 拼接一个保存模型的路径
            ckp = f"{args.save_dir}dpo_{lm_config.hidden_size}{moe_path}.pth"

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.moduile.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存，量化为FP16
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict)
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MyModel Pretraining")
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
    parser.add_argument("--accumulation_step", type=int, default=8)
    parser.add_argument("--grad_clip", type=int, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)
    parser.add_argument("--data_path", default="./data/sft_mini_512.jsonl", type=str)

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
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast()

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

    scaler = torch.amp.GradScaler(enabled=(args.dtpe in ["float16,bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        train_loader(epoch)
