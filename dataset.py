from torch.utils.data import Dataset, DataLoader
import json
import torch


# 重写DataCollator，避免调用tokenizer.pad()
class CustomDataCollator:
    def __init__(self, tokenizer, max_length=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # 提取所有样本的key值
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 手动填充到批次最大长度
        batch_max_len = min(max(len(ids) for ids in input_ids), self.max_length)

        # 初始化批量数据
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, mask, lbl in zip(input_ids, attention_masks, labels):
            # 填充input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * (batch_max_len - len(ids))
            # 填充attention_mask
            padded_mask = mask + [0] * (batch_max_len - len(mask))
            # 填充labels（-100保持不变）
            padded_lbl = lbl + [-100] * (batch_max_len - len(lbl))

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_lbl)

        # 转换为tensor
        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long)
        }
        return batch

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
        self.bos_id = tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 假设每行是json结构的数据
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def _create_chat_prompt(self, conversations):
        # 构建符合聊天的格式对话
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})

        # apply_chat_template() 它会加上一些特殊符号
        # 会返回一个长的字符串
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成损失掩码，其实就是 assistant 角色给出的content内容，才是需要计算loss
        :param self:
        :param input_ids:
        """

        loss_mask = [0] * len(input_ids)
        i = 0
        # 试图把提示词每个位置进行遍历
        while i < len(input_ids):
            # 看一下提示词第几个token是bos token, 它的索引就是 Start
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # 找到开始的那一时刻
                end = start
                # 找到结束位置
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 把我们关心需要计算loss的时刻设置为1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示prompt
        prompt = self._create_chat_prompt(sample["messages"])
        # 分词和截断
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        # 填充
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成损失掩码：训练的时候指明那些位置是我们不关心的，那些位置是我们关心的
        loss_mask = self._generate_loss_mask(input_ids)

        # 如果我们一条样本，是<start> i love you <end>，X = <start> i love you y=wx+b
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        # 对于预训练来说，每个位置都是下一个词的位置，都是我们关心的
        # 但是对于我们微调来说，比如chat_model或者叫问答模型来说
        # 只有答案所对应的位置是否预测准确，才是我们关心的
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class PretrainDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 假设每行是json结构的数据
                data = json.loads(line.strip())
                samples.append(data)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # 核心代码逻辑
        # 我们需要准备 Y 是 X 的下一个词
        # 构建输入的文本
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        # loss_mask 就是在计算的loss的时候那些时刻对应的输出是我们不关心的，是需要被mask掉
        loss_mask = input_ids != self.tokenizer.pad_token_id

        # 如果我们一条样本，是<start> i love you <end>，X = <start> i love you y=wx+b
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  # 一行数据，其中包含chosen和rejected字段
        chosen = item["chosen"]
        rejected = item["rejected"]
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # 下面才是会把文本变成token ids的过程
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.bfloat16)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.bfloat16)
        mask_chosen = torch.tensor(chosen_loss_mask, dtype=torch.bfloat16)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.bfloat16)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.bfloat16)
        mask_rejected = torch.tensor(rejected_loss_mask, dtype=torch.bfloat16)

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
        }

    def _generate_loss_mask(self, input_ids):
        # input_ids = question+answer(chosen) question+answer(rejected)
        loss_mask = [0] * len(input_ids)  # 初始化
        i = 0
        while i < len(input_ids):
            # 本质上就是字符串匹配，匹配句子中的 <|im_start|>assistant
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 本质上就是字符串匹配，匹配句子中的 <|im_end|>
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = (
                        1  # 就是将 question+answer(rejected) answer 对应的mask设置为1
                    )
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
