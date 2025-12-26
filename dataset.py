from torch.utils.data import Dataset, DataLoader
import json
import torch


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
                for i in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示prompt
        prompt = self._create_chat_prompt(sample["conversations"])
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
        input_ids = encoding.input_ids.squesze()
        # loss_mask 就是在计算的loss的时候那些时刻对应的输出是我们不关心的，是需要被mask掉
        loss_mask = input_ids != self.tokenizer.pad_token_id

        # 如果我们一条样本，是<start> i love you <end>，X = <start> i love you y=wx+b
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
