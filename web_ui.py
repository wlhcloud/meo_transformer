import os

from utils.my_llm import original_model_path, peft_model_path

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

device_map = {"cuda": int(os.environ.get("LOCAL_RANK") or 0)}

# 全局变量
model = None
tokenizer = None
processor = None
is_finetuned = False  # 标记当前是否使用微调模型


# 加载模型和 tokenizer, processor
def load_model():
    global model, tokenizer, is_finetuned, processor
    # tokenizer = AutoTokenizer.from_pretrained(original_model_path, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     original_model_path, torch_dtype=torch.bfloat16, device_map=device_map
    # )
    # 1. 加载 tokenizer（带 trust_remote_code，因为 DeepSeek/Qwen 有自定义模板）
    tokenizer = AutoTokenizer.from_pretrained(
        original_model_path,
        trust_remote_code=True
    )

    # 2. 加载模型（建议使用 AutoModelForCausalLM 以确保兼容）
    model = AutoModelForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # 自动分配 GPU 或用 device="cuda"
        trust_remote_code=True
    )
    model.load_adapter(peft_model_path, adapter_name="lora")
    model.disable_adapters()

    is_finetuned = False  # 默认加载原始模型
    return "原始模型加载成功！"


# 切换模型模式
def toggle_model_mode():
    global model, is_finetuned
    if model is None:
        return "请先加载模型！"

    if is_finetuned:
        # 禁用 LoRA adapter，使用原始模型
        model.disable_adapters()
        is_finetuned = False
        return "切换到原始模型模式！"
    else:
        model.enable_adapters()  # 显式启用
        is_finetuned = True
        return "切换到微调模型模式！"


# 生成回答
def generate_response(prompt):
    if model is None or tokenizer is None:
        return "请先加载模型！"

    # 根据模型模式构建 messages
    if is_finetuned:
        # 微调模型使用特定的 system prompt
        messages = [
            {
                "role": "system",
                "content": "Answer the question truthfully, you are a medical professional.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ]
    else:
        # 原始模型使用默认的 system prompt
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    # 处理输入
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to("cuda:0")
    eos_token_ids = [tokenizer.eos_token_id]

    # 生成输出
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=eos_token_ids
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("# DeepSeek-R1 1.5B模型测试")

    with gr.Row():
        load_model_btn = gr.Button("加载模型")
        toggle_mode_btn = gr.Button("切换模型模式")

    with gr.Row():
        model_status = gr.Textbox(label="模型状态", interactive=False)

    with gr.Row():
        prompt_input = gr.Textbox(label="输入问题", placeholder="请输入你的问题...")
        generate_btn = gr.Button("生成回答")

    with gr.Row():
        response_output = gr.Textbox(label="模型回答", interactive=False)

    # 绑定按钮事件
    load_model_btn.click(load_model, outputs=model_status)
    toggle_mode_btn.click(toggle_model_mode, outputs=model_status)
    generate_btn.click(generate_response, inputs=prompt_input, outputs=response_output)

# 启动 Gradio 应用
demo.launch(share=True)