from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from load_env import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LOCAL_BASE_URL

# llm = ChatOpenAI(  # 调用官方的deepseek 的大模型(R1)
#     model='deepseek-rea TRsoner',
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL,
# )


llm = ChatOpenAI(  # 调用私有化部署的大模型 (全模态的大模型)
    model="qwen-8b",
    api_key="xx",
    temperature=0.8,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    base_url=LOCAL_BASE_URL,
)


resp = llm.invoke([HumanMessage(content="中国的人口有多少？")])
print(resp.content)
