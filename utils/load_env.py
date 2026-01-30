import os

from dotenv import load_dotenv

load_dotenv()  # 加载环境变量


# 获取环境变量的值
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')

LOCAL_BASE_URL = os.getenv('LOCAL_BASE_URL')