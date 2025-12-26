from torch.utils.data import Dataset, DataLoader
import json
import torch


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
