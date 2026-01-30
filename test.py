import transformers
import torch

model_path = r"/home/gybwg/ai-project/models/Qwen/Qwen3-0.6B"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_path,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "我的牙齿疼怎么办?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])