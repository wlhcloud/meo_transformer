import base64
import io
import uuid
from typing import List, Dict

import gradio as gr
from PIL import Image
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from load_env import LOCAL_BASE_URL

llm = ChatOpenAI(  # 调用私有化部署的大模型 (全模态的大模型)
    model="qwen-omni-3b",
    api_key="xx",
    base_url=LOCAL_BASE_URL,
)

#  第一步： 定义提示词模板，并且指定动态参数
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个多模态AI助手，可以处理文本、音频和图像输入"),
        MessagesPlaceholder(
            variable_name="messages"
        ),  #  代表：历史消息。 让大模型可以理解上下文语义
    ]
)

# 第二步： 保存历史消息记录的机制（内存）
store = {}  # 用来保存历史消息


def get_session_history(session_id: str):
    """从内存中的历史消息列表中 返回当前会话 的所有历史消息"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 第三步： 创建一个执行对象（根据上下文和用户当前的输入 ，调用大模型）

chain = prompt | llm

execute = RunnableWithMessageHistory(chain, get_session_history)

# 参数的格式固定的：
config = {
    "configurable": {"session_id": str(uuid.uuid4())}
}  # 通过UUID算法，生成一个随机的会话ID


# resp = execute.invoke([HumanMessage(content="你知道中国最大的淡水湖是哪个吗？")], config)
# print(resp.content)


# 定义 组件 交互的函数
def respond(text, chat_bot, audio=None, image=None):
    """处理输入的函数"""
    message = None
    # 处理语音输入
    if audio:
        print(audio)
        message = "[语音输入]"

    # 处理图像输入
    if image:
        message = "[图片输入] "
        # 实际应用中需将图像路径传递给多模态模型
        # 此处简化处理，实际需修改chain以支持图像

    if text:
        message = text

    # 生成输入消息
    if message:
        chat_bot.append({"role": "user", "content": message})

    return chat_bot


# 语音处理函数 =====
def transcribe_audio(audio_path):
    """使用Base64处理语音转为"""

    # 目前多模态大模型： 支持两个传参方式，1、base64（字符串）（本地）。2、网络访问的url地址（外网的服务器上） http://sxxxx.com/11.mp3
    try:
        with open(audio_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
        content = [
            {"type": "text", "text": "请处理语音"},
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/mp3;base64,{audio_data}",
                    "duration": 30,  # 单位：秒（帮助模型优化处理）
                },
            },
        ]
        return content
    except Exception as e:
        return f"[语音识别错误] {str(e)}"


def execute_chain_audio(chat_bot: List[Dict], audio=None):
    """执行大模型应用 （调用）"""

    response = None
    # 处理语音输入
    print(2222222)
    print(audio)
    if audio:
        result = transcribe_audio(audio)
        audio_message = HumanMessage(content=result)
        response = execute.invoke(
            [audio_message],
            config=config,
        )

    if response:
        chat_bot.append({"role": "assistant", "content": response.content})
    return chat_bot, None


def execute_chain_image(chat_bot: List[Dict], image=None):
    """执行大模型应用 （调用）"""

    response = None
    # 处理语音输入
    print(333333)
    print(image)
    if image:
        img = Image.open(image)
        # img.thumbnail((512, 512))   # 512k 依赖
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_message = HumanMessage(
            content=[
                {"type": "text", "text": "请简单描述这张图片，并提取里面的文字"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ]
        )
        response = execute.invoke(
            [image_message],
            config=config,
        )

    if response:
        chat_bot.append({"role": "assistant", "content": response.content})
    return chat_bot, None


def execute_chain_text(chat_bot: List[Dict], text=None):
    """执行大模型应用 （调用）"""

    response = None
    # 处理语音输入
    print(4444)
    print(text)
    if text:

        response = execute.invoke(
            [HumanMessage(content=text)],
            config=config,
        )

    if response:
        chat_bot.append({"role": "assistant", "content": response.content})
    return chat_bot, None


# 开发一个界面
with gr.Blocks(title="多模态大模型案例", theme=gr.themes.Soft()) as instance:

    # 聊天历史记录
    chatbot = gr.Chatbot(type="messages", height=500, label="历史对话")

    with gr.Row():
        # 文字输入区
        with gr.Column(scale=4):
            text_input = gr.Textbox(
                placeholder="输入文字...",
                label="文字输入",
                max_lines=5,  # 支持多行输入[5](@ref)
            )
            submit_btn = gr.Button("发送", variant="primary")

        # 多媒体输入区
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],  # 新版参数名[8](@ref)
                type="filepath",
                format="mp3",
                label="语音输入",
            )
            image_input = gr.Image(
                type="filepath", label="图片上传", height=150  # 优化图片预览高度
            )

    audio_input.change(
        respond, [text_input, chatbot, audio_input, image_input], chatbot
    ).then(execute_chain_audio, [chatbot, audio_input], [chatbot, audio_input])

    image_input.change(
        respond, [text_input, chatbot, audio_input, image_input], chatbot
    ).then(execute_chain_image, [chatbot, image_input], [chatbot, image_input])

    text_input.submit(
        respond, [text_input, chatbot, audio_input, image_input], chatbot
    ).then(execute_chain_text, [chatbot, text_input], [chatbot, text_input])


if __name__ == "__main__":
    instance.launch()
