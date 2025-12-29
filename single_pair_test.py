import os
from openai import OpenAI
from config import config
import numpy as np

client = OpenAI(
    api_key=config.GITEEAI_API_KEY,
    base_url="https://ai.gitee.com/v1"
)

input_pairs = [

    ["巴洛克艺术",
     "洛可可风格"],

    ["巴洛克艺术",
     "光合作用"],

    ["巴洛克艺术与洛可可风格相似", 
    "巴洛克艺术与古希腊风格相似"],

    ["巴洛克艺术与洛可可风格相似", 
    "巴洛克艺术与光合作用相似"],
     
    ["巴洛克艺术与哪一项相似？洛可可风格", 
    "巴洛克艺术与哪一项相似？古希腊风格"],

    ["巴洛克艺术与哪一项相似？洛可可风格", 
    "巴洛克艺术与哪一项相似？光合作用"],

    ["下列关于巴洛克艺术的描述，哪一项是正确的？巴洛克艺术与洛可可风格相似", 
    "下列关于巴洛克艺术的描述，哪一项是正确的？巴洛克艺术与古希腊风格相似"],

    ["下列关于巴洛克艺术的描述，哪一项是正确的？巴洛克艺术与洛可可风格相似", 
    "下列关于巴洛克艺术的描述，哪一项是正确的？巴洛克艺术与光合作用相似"],

]

dimensions=256

print(f"\n{'='*60}")
print(f"dimensions: {dimensions}")
print(f"{'='*60}\n")

def calculate_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# 遍历所有input对，计算相似度并打印
for idx, pair in enumerate(input_pairs, 1):
    print(f"\n{'='*60}")
    print(f"Pair {idx}:")
    print(f"  Text 1: {pair[0]}")
    print(f"  Text 2: {pair[1]}")
    
    # 获取embeddings
    completion = client.embeddings.create(
        model="Qwen3-Embedding-8B",
        input=pair,
        dimensions=dimensions,
    )
    
    # 计算相似度
    similarity = calculate_cosine_similarity(
        np.array(completion.data[0].embedding),
        np.array(completion.data[1].embedding)
    )
    
    print(f"  Similarity: {similarity:.6f}")
    print(f"{'='*60}")