from datasets import Dataset
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

from dataset import CustomDataCollator
from my_llm import lora_data_path, original_model_path
# 加载并清洗数据
df = pd.read_json(lora_data_path)
df = df.dropna(subset=["instruction", "output"])
ds = Dataset.from_pandas(df)

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    original_model_path,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id  # 显式设置pad token id

template_vars = {
    "AUTHOR": "智信联行",
    "NAME": "小古"
}
# 样本处理函数（确保返回一维列表）
def process_func(example):
    MAX_LENGTH = 512

    output_text = example["output"]
    for var, value in template_vars.items():
        output_text = output_text.replace("{{" + var + "}}", value)

    # 构造对话模板
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": output_text}
    ]

    # 直接编码为input_ids（跳过chat_template的格式问题）
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    input_ids = tokenized["input_ids"]

    # 单独编码用户部分，用于构造labels掩码
    user_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["instruction"]}],
        tokenize=False
    )
    user_tokenized = tokenizer(
        user_prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    user_len = len(user_tokenized["input_ids"])

    # 构造labels
    labels = [-100] * user_len + input_ids[user_len:] if user_len < len(input_ids) else [-100] * len(input_ids)

    # 确保长度一致
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 映射处理函数
tokenized_id = ds.map(
    process_func,
    remove_columns=ds.column_names,
    batched=False
)


# 初始化自定义填充器
data_collator = CustomDataCollator(tokenizer=tokenizer, max_length=1000)

model = AutoModelForCausalLM.from_pretrained(
    original_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    tie_word_embeddings=False,
    attn_implementation="eager"
)
model.enable_input_require_grads()

# LoRA配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


args = TrainingArguments(
    output_dir="./output/qwen_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=20,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    optim="paged_adamw_8bit"
)

# 使用自定义填充器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存权重
peft_model_id = "./out/lora/wenbo_think_0.6b"
model.save_pretrained(peft_model_id, safe_serialization=True)
tokenizer.save_pretrained(peft_model_id)

print(f"训练完成，LoRA权重已保存至: {peft_model_id}")