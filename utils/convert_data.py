import pandas as pd
import json
import re

# 配置路径
INPUT_JSONL = "./data/self_cognition.jsonl"    # 你的原始jsonl文件
OUTPUT_JSON = "./data/self_cognition.json"  # 输出的标准格式json

# 正则：去掉	response 里的思考过程（匹配  包裹的内容）
def clean_response(response: str) -> str:
    """清理response，只保留最终回答"""
    if pd.isna(response):  # 处理空值
        return ""
    # 去掉思考过程，只保留最终回答
    cleaned = re.sub(r'<think>.*?</think>\n', '', response, flags=re.DOTALL)
    return cleaned

def convert_with_pandas(input_path: str, output_path: str):
    # 步骤1：用pandas读取jsonl文件（自动处理每行json）
    # lines=True 表示每行是一个独立的json对象
    df = pd.read_json(input_path, lines=True, encoding="utf-8")

    # 步骤2：数据清洗（可选，根据你的数据情况调整）
    # 1. 过滤空值：query或response为空的行
    df = df.dropna(subset=["query", "response"])
    # 2. 去重：按query去重（避免重复数据）
    df = df.drop_duplicates(subset=["query"], keep="first")
    # 3. 清理空白字符
    df["query"] = df["query"].str.strip()
    df["response"] = df["response"].str.strip()

    # 步骤3：转换为目标格式
    standard_data = []
    for _, row in df.iterrows():
        # 提取字段并清理
        instruction = row["query"]
        # output_text = clean_response(row["response"])
        output_text = row["response"]

        # 构造标准微调格式
        standard_item = {
            "instruction": instruction,
            "input": "",  # input留空
            "output": output_text,
            "system": ""  # system留空
        }
        standard_data.append(standard_item)

    # 步骤4：保存为标准JSON文件（数组格式）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(standard_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息（方便核对）
    print(f"转换完成！")
    print(f"原始数据行数：{len(df)}")
    print(f"转换后数据行数：{len(standard_data)}")
    print(f"输出文件路径：{output_path}")

if __name__ == "__main__":
    convert_with_pandas(INPUT_JSONL, OUTPUT_JSON)
    # template_vars = {
    #     "AUTHOR": "智信联行",
    #     "NAME": "小古"
    # }
    # # 样本处理函数（确保返回一维列表）
    # output_text = "我是{{NAME}}，由{{AUTHOR}}训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？"
    # for var, value in template_vars.items():
    #     output_text = output_text.replace("{{" + var + "}}", value)
    # print(output_text)
