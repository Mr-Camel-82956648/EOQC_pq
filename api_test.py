import os
from openai import OpenAI
from config import config
import numpy as np

client = OpenAI(
    api_key=config.GITEEAI_API_KEY,
    base_url="https://ai.gitee.com/v1"
)

input_text = "巴洛克艺术与洛可可风格相似"
dimensions = 128

# 获取embeddings
completion = client.embeddings.create(
    model="all-mpnet-base-v2",  # Qwen3-Embedding-8B
    input=input_text,
    dimensions=dimensions,
)

print(completion.data[0].embedding)
print(len(completion.data[0].embedding))